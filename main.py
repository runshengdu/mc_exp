import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import yaml
from tqdm import tqdm

from src.data_loader import DataLoader, sanitize_unusual_line_terminators
from src.evaluator import Evaluator
from src.models.openai import OpenAIModel
from src.result_parser import ResultParser
from src.utils import submit_with_mapping

def _resolve_path(file_path: str, default_dir: str) -> str:
    """Resolve file path: if not absolute and no directory, prepend default_dir."""
    if not os.path.isabs(file_path) and not os.path.dirname(file_path):
        return os.path.join(default_dir, file_path)
    return file_path

def _models_config_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.yaml')

def _evaluators_config_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluators.yaml')

def _resolve_env_vars(obj):
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    if isinstance(obj, str):
        pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

        def repl(match):
            key = match.group(1)
            val = os.getenv(key)
            if val is None:
                raise ValueError(f"Environment variable {key} is not set")
            return val

        return pattern.sub(repl, obj)
    return obj

def load_model_config(model_id: str, config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    models = cfg.get('models') or []
    match = next((m for m in models if m.get('name') == model_id), None)
    if match is None:
        raise ValueError(f"Model id {model_id} not found in {config_path}")

    return _resolve_env_vars(match)

def _parse_json_objects_strict(content: str, file_path: str):
    decoder = json.JSONDecoder()
    objs = []
    idx = 0
    length = len(content)
    while idx < length:
        while idx < length:
            if content[idx].isspace():
                idx += 1
                continue
            # Some result files may accidentally contain literal escape sequences like "\\n"
            # between JSON objects (or at EOF). A JSON value cannot start with a backslash, so
            # treating these sequences as whitespace is safe and makes parsing more robust.
            if content[idx] == "\\" and idx + 1 < length and content[idx + 1] in {"n", "r", "t"}:
                idx += 2
                continue
            break
        if idx >= length:
            break
        try:
            obj, end_idx = decoder.raw_decode(content, idx)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON starting at char {idx + 1} in {file_path}: {e}") from e
        objs.append(obj)
        idx = end_idx
    return objs

def _load_results_file(file_path: str):
    summary_obj = None
    records_by_qid = {}
    if not os.path.exists(file_path):
        return summary_obj, records_by_qid
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    for obj in _parse_json_objects_strict(content, file_path):
        if not isinstance(obj, dict):
            continue
        if 'summary' in obj:
            summary_obj = obj
            continue
        qid = obj.get('question_id')
        if qid is None:
            continue
        records_by_qid[qid] = obj
    return summary_obj, records_by_qid

def _record_has_nonempty_evaluations(record: dict) -> bool:
    if not isinstance(record, dict):
        return False
    evaluations = record.get('evaluations')
    if evaluations is None:
        return False
    if isinstance(evaluations, list):
        return len(evaluations) > 0
    return True

def _is_legacy_attempt_row(record: dict) -> bool:
    if not isinstance(record, dict):
        return False
    legacy_markers = {'attempt_number', 'model_response', 'judge_verdict', 'final_result'}
    return any(k in record for k in legacy_markers)

def _write_results_file(file_path: str, summary_obj, records_by_qid: dict) -> None:
    output_dir = os.path.dirname(file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    tmp_path = file_path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        if summary_obj is not None:
            f.write(json.dumps(sanitize_unusual_line_terminators(summary_obj), ensure_ascii=False, indent=4) + "\n")
        for qid in sorted(records_by_qid.keys()):
            obj = records_by_qid[qid]
            f.write(json.dumps(sanitize_unusual_line_terminators(obj), ensure_ascii=False, indent=4) + "\n")
    for attempt in range(5):
        try:
            os.replace(tmp_path, file_path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2 * (2 ** attempt))

def _parse_evaluators(evaluator_arg: str) -> list:
    evaluators = [e.strip() for e in (evaluator_arg or '').split(',') if e.strip()]
    if not evaluators:
        raise ValueError("At least one evaluator must be provided")
    if len(evaluators) not in {1, 3, 5}:
        raise ValueError("Number of evaluators must be 1, 3, or 5")
    return evaluators

def main():

    parser = argparse.ArgumentParser(description="Run LLM benchmark for conversational ability.")
    
    parser.add_argument('--responses-file', type=str,
                        help="Path to the JSONL results file to write (responses + empty evaluations).")
    parser.add_argument('--model-id', type=str, default='kimi-k2.5',
                        help="Model id to use for response generation (must exist in models.yaml).")
    parser.add_argument('--evaluator', type=str,
                        default='doubao-seed-1-8-251228,glm-4.7,kimi-k2.5',
                        help="Comma-separated evaluator model ids (1, 3, or 5 models; must exist in evaluators.yaml).")
    parser.add_argument('--num-tasks', type=int,
                        help="If provided, only run the first k tasks from the benchmark dataset.")
    parser.add_argument('--gen-max-workers', type=int, default=50,
                        help="Number of parallel workers to use for response generation.")
    parser.add_argument('--eval-max-workers', type=int, default=50,
                        help="Number of parallel workers to use for evaluation.")
    parser.add_argument('--evaluate-file', type=str,
                        help="Results file for evaluation (in-place update).")

    args = parser.parse_args()
    try:
        evaluator_ids = _parse_evaluators(args.evaluator)
    except ValueError as e:
        parser.error(str(e))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_id_safe = args.model_id.replace('/', '_')

    if args.responses_file and args.evaluate_file:
        parser.error("The --responses-file argument cannot be used together with --evaluate-file")

    input_file = './data/benchmark_questions.jsonl'

    data_loader = DataLoader(input_file)
    data_loader.load_data(num_tasks=args.num_tasks)

    if args.evaluate_file:
        evaluate_file = _resolve_path(args.evaluate_file, 'results')

        if not os.path.exists(evaluate_file):
            parser.error("The --evaluate-file path does not exist")

        summary_obj, records_by_qid = _load_results_file(evaluate_file)
        if any(_is_legacy_attempt_row(r) for r in records_by_qid.values()):
            parser.error("The --evaluate-file must be a generation results file (aggregated records), not a legacy evaluation JSONL")
        data_loader.load_responses(evaluate_file)

        config_path = _evaluators_config_path()
        evaluator_cfgs = [load_model_config(eid, config_path) for eid in evaluator_ids]

        evaluated_question_ids = {qid for qid, r in records_by_qid.items() if _record_has_nonempty_evaluations(r)}

        responses = data_loader.get_responses()
        token_counts = data_loader.get_token_counts()
        conversations = data_loader.get_conversations()

        response_qids = set(responses.keys())
        conversations_in_scope = [conv for conv in conversations if conv.question_id in response_qids]
        missing_in_benchmark = response_qids - {conv.question_id for conv in conversations_in_scope}
        if missing_in_benchmark:
            print(f"Evaluation: {len(missing_in_benchmark)} QUESTION_IDs in responses file are not in the loaded benchmark and will be ignored")

        evaluator = Evaluator(conversations_in_scope, responses, evaluator_configs=evaluator_cfgs)

        conversations_to_eval = [
            conv for conv in conversations_in_scope
            if conv.question_id not in evaluated_question_ids
        ]

        skipped_eval = len(conversations_in_scope) - len(conversations_to_eval)
        remaining_eval = len(conversations_to_eval)
        print(f"Evaluation: skipped {skipped_eval} conversations, remaining {remaining_eval} conversations")

        if conversations_to_eval:
            with ThreadPoolExecutor(max_workers=args.eval_max_workers) as executor:
                def _eval_one(conv):
                    return evaluator.evaluate_conversation(conv, responses.get(conv.question_id, []))

                future_to_conv = submit_with_mapping(executor, conversations_to_eval, _eval_one)

                for future in tqdm(as_completed(future_to_conv), total=len(future_to_conv), desc="Evaluating"):
                    conv = future_to_conv[future]

                    try:
                        record = future.result()
                    except Exception as e:
                        print(f"Error evaluating question {conv.question_id}: {e}")
                        raise SystemExit(1)

                    if record.get('_failed_evaluators'):
                        failed = record.pop('_failed_evaluators')
                        backup = record.pop('_backup_used', [])
                        tqdm.write(f"[{conv.question_id}] Failed: {failed}, Backup: {backup}")

                    existing = records_by_qid.get(conv.question_id, {})
                    updated = dict(existing) if isinstance(existing, dict) else {}
                    updated.update(record)
                    updated['original_conversation'] = conv.conversation
                    if 'token_count' not in updated and conv.question_id in token_counts:
                        updated['token_count'] = token_counts.get(conv.question_id)
                    records_by_qid[conv.question_id] = updated
                    _write_results_file(evaluate_file, summary_obj, records_by_qid)
                    evaluated_question_ids.add(conv.question_id)

        ResultParser.prepend_summary_jsonl(evaluate_file)
        print(f"Evaluation results updated in {evaluate_file}")
        return

    results_output_file = None
    if args.responses_file:
        results_output_file = _resolve_path(args.responses_file, 'results')
    else:
        results_output_file = os.path.join('results', model_id_safe, f"{timestamp}.jsonl")
    completed_question_ids = set()
    if os.path.exists(results_output_file):
        data_loader.load_responses(results_output_file)
        completed_question_ids = set(data_loader.get_responses().keys())

    all_question_ids = {conv.question_id for conv in data_loader.get_conversations()}
    missing_question_ids = all_question_ids - completed_question_ids

    completed_in_scope = all_question_ids & completed_question_ids
    skipped_gen = len(completed_in_scope)
    remaining_gen = len(all_question_ids) - skipped_gen
    print(f"Response generation: skipped {skipped_gen} tasks, remaining {remaining_gen} tasks")

    if missing_question_ids:

        config_path = _models_config_path()
        model_cfg = load_model_config(args.model_id, config_path)

        model_provider = OpenAIModel.from_model_config(args.model_id, model_cfg)

        def _write_generation_record(output_file, conversation, responses, token_count):
            data_loader.append_result_record(
                output_file=output_file,
                conversation=conversation,
                responses=responses,
                token_count=token_count,
                evaluations=[],
                final_status="PENDING",
            )

        data_loader.generate_responses_with_checkpoint(
            model_provider=model_provider,
            output_file=results_output_file,
            max_workers=args.gen_max_workers,
            skip_question_ids=completed_question_ids,
            record_writer=_write_generation_record,
        )

    print(f"Results saved to {results_output_file}")

if __name__ == '__main__':
    main()
