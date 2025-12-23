import argparse
import os
import json
import re
import yaml
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.data_loader import DataLoader
from src.evaluator import Evaluator
from src.models.openai import OpenAIModel
from src.result_parser import ResultParser

def _resolve_path(file_path: str, default_dir: str) -> str:
    """Resolve file path: if not absolute and no directory, prepend default_dir."""
    if not os.path.isabs(file_path) and not os.path.dirname(file_path):
        return os.path.join(default_dir, file_path)
    return file_path

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

def main():

    parser = argparse.ArgumentParser(description="Run LLM benchmark for conversational ability.")
    
    parser.add_argument('--responses-file', type=str,
                        help="Path to the JSONL file containing model responses.")
    parser.add_argument('--model-id', type=str, default='minimax-m2.1',
                        help="Model id to use for response generation (must exist in models.yaml).")
    parser.add_argument('--evaluator-1', type=str, default='deepseek-chat',
                        help="Model id to use for evaluation (must exist in models.yaml).")
    parser.add_argument('--evaluator-2', type=str, default='glm-4.6',
                        help="Model id to use for evaluation (must exist in models.yaml).")
    parser.add_argument('--evaluator-3', type=str, default='kimi-k2-0905-preview',
                        help="Model id to use for evaluation (must exist in models.yaml).")
    parser.add_argument('--num-tasks', type=int,
                        help="If provided, only run the first k tasks from the benchmark dataset.")
    parser.add_argument('--max-workers', type=int, default=30,
                        help="Number of parallel workers to use for all LLM calls (response generation and evaluation).")
    parser.add_argument('--evaluate-file', type=str,
                        help="Responses file for evaluation.")
    parser.add_argument('--output-file', type=str,
                        help="Path to save evaluation output (responses + evaluations).")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_id_safe = args.model_id.replace('/', '_')

    if args.responses_file and (args.evaluate_file or args.output_file):
        parser.error("The --responses-file argument cannot be used together with --evaluate-file or --output-file")

    input_file = './data/benchmark_questions.jsonl'

    data_loader = DataLoader(input_file)
    data_loader.load_data(num_tasks=args.num_tasks)

    if args.evaluate_file:
        evaluate_file = _resolve_path(args.evaluate_file, 'responses')

        if not os.path.exists(evaluate_file):
            parser.error("The --evaluate-file path does not exist")

        data_loader.load_responses(evaluate_file)

        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.yaml')
        evaluator_cfgs = [
            load_model_config(args.evaluator_1, config_path),
            load_model_config(args.evaluator_2, config_path),
            load_model_config(args.evaluator_3, config_path),
        ]

        if args.output_file:
            output_file = _resolve_path(args.output_file, 'results')
        else:
            eval_base = os.path.basename(evaluate_file)
            eval_stem, _ = os.path.splitext(eval_base)
            output_file = os.path.join('results', f"{eval_stem}_evaluation.jsonl")
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        evaluated_question_ids = set()
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
            for item in ResultParser._parse_json_objects(content):
                if isinstance(item, dict) and 'SUMMARY' in item:
                    continue
                qid = None
                if isinstance(item, dict):
                    qid = item.get('question_id') or item.get('QUESTION_ID')
                if qid is not None:
                    evaluated_question_ids.add(qid)

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
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_to_conv = {
                    executor.submit(
                        evaluator.evaluate_conversation,
                        conv,
                        responses.get(conv.question_id, [])
                    ): conv
                    for conv in conversations_to_eval
                }

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

                    ResultParser.append_evaluation_record(output_file, conv, record, token_counts.get(conv.question_id))
                    evaluated_question_ids.add(conv.question_id)

        ResultParser.prepend_summary_jsonl(output_file)
        print(f"Evaluation output saved to {output_file}")
        return

    responses_output_file = None
    if args.responses_file:
        responses_output_file = _resolve_path(args.responses_file, 'responses')
    else:
        responses_output_file = os.path.join('responses', f"{model_id_safe}_{timestamp}.jsonl")
    completed_question_ids = set()
    if os.path.exists(responses_output_file):
        data_loader.load_responses(responses_output_file)
        completed_question_ids = set(data_loader.get_responses().keys())

    all_question_ids = {conv.question_id for conv in data_loader.get_conversations()}
    missing_question_ids = all_question_ids - completed_question_ids

    completed_in_scope = all_question_ids & completed_question_ids
    skipped_gen = len(completed_in_scope)
    remaining_gen = len(all_question_ids) - skipped_gen
    print(f"Response generation: skipped {skipped_gen} tasks, remaining {remaining_gen} tasks")

    if missing_question_ids:

        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.yaml')
        model_cfg = load_model_config(args.model_id, config_path)

        model_provider = OpenAIModel.from_model_config(args.model_id, model_cfg)

        data_loader.generate_responses_with_checkpoint(
            model_provider=model_provider,
            output_file=responses_output_file,
            max_workers=args.max_workers,
            skip_question_ids=completed_question_ids
        )

    print(f"Responses saved to {responses_output_file}")

if __name__ == '__main__':
    main()