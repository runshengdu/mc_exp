import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

# Ensure repository root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main as benchmark_main
from src.conversation import Conversation
from src.data_loader import DataLoader
from src.models.openai import OpenAIModel

TERMINAL_BATCH_STATES = {"completed", "failed", "expired", "cancelled"}
KIMI_BATCH_FORBIDDEN_PARAMS = {
    "temperature",
    "max_tokens",
    "top_p",
    "n",
    "presence_penalty",
    "frequency_penalty",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark responses with Batch API.")
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["all", "prepare", "upload", "create", "wait", "collect", "submit", "poll"],
        help="Pipeline step to run. Recommended flow: prepare/upload/create/wait/collect.",
    )
    parser.add_argument("--model-id", type=str, default="kimi-k2.6", help="Model id in models.yaml")
    parser.add_argument(
        "--input-file",
        type=str,
        default="./data/benchmark_questions.jsonl",
        help="Benchmark questions JSONL file path.",
    )
    parser.add_argument(
        "--responses-file",
        type=str,
        default=None,
        help="Output JSONL path for generation records. If omitted, use main.py default style.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="If provided, only run the first k tasks from benchmark input.",
    )
    parser.add_argument(
        "--completion-window",
        type=str,
        default="24h",
        help="Batch completion window. Default is 24h.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=10,
        help="Batch status polling interval in seconds.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="batch_api/moonshot/artifacts",
        help="Directory to store intermediate artifacts.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Existing run directory, required by upload/create/wait/collect steps.",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Batch ID override for wait/collect steps.",
    )
    return parser.parse_args()


def make_client(model_id: str) -> OpenAI:
    cfg = benchmark_main.load_model_config(model_id, benchmark_main._models_config_path())
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def extract_text_from_file_content(content_obj: Any) -> str:
    text_attr = getattr(content_obj, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    if callable(text_attr):
        return text_attr()
    read_method = getattr(content_obj, "read", None)
    if callable(read_method):
        raw = read_method()
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return str(raw)
    return str(content_obj)


def resolve_responses_output_path(args: argparse.Namespace) -> str:
    if args.responses_file:
        return benchmark_main._resolve_path(args.responses_file, "results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_id_safe = benchmark_main._sanitize_path_component(args.model_id)
    return str(Path("results") / model_id_safe / f"{timestamp}.jsonl")


def build_run_dir(artifacts_dir: Path, model_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = benchmark_main._sanitize_path_component(model_id)
    run_dir = artifacts_dir / safe_model / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _coerce_token_int(value: Any, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return default


def _build_conversation_index(data_loader: DataLoader) -> dict[str, Conversation]:
    return {str(conv.question_id): conv for conv in data_loader.get_conversations()}


def _load_completed_question_ids(input_file: str, results_output_file: str) -> set[str]:
    completed = set()
    if not Path(results_output_file).exists():
        return completed
    loader = DataLoader(input_file)
    loader.load_responses(results_output_file)
    for qid in loader.get_responses().keys():
        completed.add(str(qid))
    return completed


def _openai_api_kwargs(model_cfg: dict[str, Any]) -> dict[str, Any]:
    kwargs = OpenAIModel.build_api_params(model_cfg)
    kwargs.pop("model", None)
    kwargs.pop("messages", None)
    kwargs.pop("temperature", None)
    kwargs.pop("stream", None)
    extra_body = kwargs.pop("extra_body", None)
    if isinstance(extra_body, dict):
        kwargs.update(extra_body)
    return kwargs


def build_batch_request_body(
    model_id: str,
    messages: list[dict[str, Any]],
    model_cfg: dict[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": float(model_cfg.get("temperature", 0.0)),
    }
    body.update(_openai_api_kwargs(model_cfg))

    if model_id in {"kimi-k2.5", "kimi-k2.6"}:
        for k in KIMI_BATCH_FORBIDDEN_PARAMS:
            body.pop(k, None)
    return body


def build_batch_input_file(
    conversations: list[Conversation],
    pending_question_ids: set[str],
    model_id: str,
    model_cfg: dict[str, Any],
    input_path: Path,
) -> dict[str, dict[str, Any]]:
    question_payloads: dict[str, dict[str, Any]] = {}

    with open(input_path, "w", encoding="utf-8", newline="\n") as f:
        for conv in conversations:
            qid = str(conv.question_id)
            if qid not in pending_question_ids:
                continue
            request_obj = {
                "custom_id": f"question-{qid}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": build_batch_request_body(model_id, conv.conversation, model_cfg),
            }
            f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")
            question_payloads[qid] = {
                "question_id": conv.question_id,
                "axis": conv.axis,
                "target_question": conv.target_question,
                "pass_criteria": conv.pass_criteria,
            }
    return question_payloads


def parse_question_id(custom_id: str) -> str | None:
    if not isinstance(custom_id, str):
        return None
    if not custom_id.startswith("question-"):
        return None
    value = custom_id[len("question-") :]
    if value == "":
        return None
    return value


def parse_batch_output(
    output_text: str,
    question_payloads: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    question_results: dict[str, dict[str, Any]] = {}
    failed_question_ids: set[str] = set()

    for raw_line in output_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except Exception:
            continue

        question_id = parse_question_id(data.get("custom_id"))
        if question_id is None or question_id not in question_payloads:
            continue

        error = data.get("error")
        response = data.get("response") or {}
        status_code = response.get("status_code")
        body = response.get("body") or {}

        if error is not None or status_code not in {0, 200}:
            failed_question_ids.add(question_id)
            continue

        choices = body.get("choices") or []
        if not choices:
            failed_question_ids.add(question_id)
            continue

        message = choices[0].get("message") or {}
        content = message.get("content")
        usage = body.get("usage") or {}
        prompt_tokens = _coerce_token_int(usage.get("prompt_tokens"), 0)
        completion_tokens = _coerce_token_int(usage.get("completion_tokens"), 0)
        total_tokens = prompt_tokens + completion_tokens

        if not isinstance(content, str):
            failed_question_ids.add(question_id)
            continue

        question_results[question_id] = {
            "responses": [content],
            "token_count": total_tokens,
        }
    return question_results, failed_question_ids


def upload_input_file(client: OpenAI, input_path: Path) -> str:
    with open(input_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    print(f"输入文件已上传: {file_obj.id}")
    return file_obj.id


def create_batch(client: OpenAI, input_file_id: str, completion_window: str) -> str:
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
    )
    print(f"Batch 任务已创建: {batch.id}")
    return batch.id


def poll_batch(client: OpenAI, batch_id: str, poll_interval_seconds: int) -> Any:
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = getattr(batch, "request_counts", None)
        completed = getattr(counts, "completed", 0) if counts else 0
        total = getattr(counts, "total", 0) if counts else 0
        print(f"状态: {batch.status} ({completed}/{total})")

        if batch.status in TERMINAL_BATCH_STATES:
            return batch
        time.sleep(max(1, int(poll_interval_seconds)))


def stage_prepare(args: argparse.Namespace) -> Path:
    model_id = str(args.model_id)
    input_file = str(args.input_file)
    results_output_file = resolve_responses_output_path(args)

    data_loader = DataLoader(input_file)
    data_loader.load_data(num_tasks=args.num_tasks)
    conversations = data_loader.get_conversations()
    conversations_by_qid = _build_conversation_index(data_loader)

    completed_question_ids = _load_completed_question_ids(input_file, results_output_file)
    all_question_ids = {str(conv.question_id) for conv in conversations}
    pending_question_ids = all_question_ids - completed_question_ids

    if not pending_question_ids:
        print("输出文件中已包含所有任务的响应，无需继续生成。")
        raise SystemExit(0)

    artifacts_root = Path(args.artifacts_dir)
    run_dir = build_run_dir(artifacts_root, model_id)
    input_jsonl = run_dir / "batch_input.jsonl"
    output_jsonl = run_dir / "batch_output.jsonl"
    error_jsonl = run_dir / "batch_error.jsonl"
    question_payloads_json = run_dir / "question_payloads.json"
    meta_json = run_dir / "meta.json"

    model_cfg = benchmark_main.load_model_config(model_id, benchmark_main._models_config_path())
    question_payloads = build_batch_input_file(
        conversations=conversations,
        pending_question_ids=pending_question_ids,
        model_id=model_id,
        model_cfg=model_cfg,
        input_path=input_jsonl,
    )
    submitted_question_ids = set(question_payloads.keys())

    if not submitted_question_ids:
        print("没有可提交的问题（可能都已完成）。")
        raise SystemExit(0)

    save_json(question_payloads_json, question_payloads)
    metadata = {
        "version": 2,
        "model": model_id,
        "input_file": input_file,
        "responses_file": results_output_file,
        "num_tasks": args.num_tasks,
        "run_dir": str(run_dir),
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "error_jsonl": str(error_jsonl),
        "question_payloads_json": str(question_payloads_json),
        "input_file_id": None,
        "batch_id": None,
        "batch_status": None,
        "completion_window": str(args.completion_window),
        "poll_interval_seconds": int(args.poll_interval_seconds),
        "submitted_questions": len(submitted_question_ids),
        "submitted_question_ids": sorted(submitted_question_ids),
        "completed_questions_before_run": len(completed_question_ids & all_question_ids),
        "questions_in_input": len(conversations_by_qid),
    }
    save_json(meta_json, metadata)
    print(
        f"prepare 完成，run_dir: {run_dir}，"
        f"已完成跳过: {metadata['completed_questions_before_run']}，"
        f"待提交: {metadata['submitted_questions']}"
    )
    return run_dir


def load_meta_or_fail(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    meta_json = run_dir / "meta.json"
    if not meta_json.exists():
        raise FileNotFoundError(f"meta.json 不存在: {meta_json}")
    return meta_json, load_json(meta_json)


def stage_upload(args: argparse.Namespace, run_dir: Path) -> None:
    meta_json, metadata = load_meta_or_fail(run_dir)
    model_id = str(metadata["model"])
    input_jsonl = Path(metadata["input_jsonl"])
    if not input_jsonl.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_jsonl}")

    client = make_client(model_id)
    input_file_id = upload_input_file(client, input_jsonl)

    metadata["input_file_id"] = input_file_id
    metadata["batch_id"] = None
    metadata["batch_status"] = None
    save_json(meta_json, metadata)
    print(f"upload 完成，input_file_id: {input_file_id}")


def stage_create(args: argparse.Namespace, run_dir: Path) -> str:
    meta_json, metadata = load_meta_or_fail(run_dir)
    model_id = str(metadata["model"])
    input_file_id = metadata.get("input_file_id")
    if not input_file_id:
        raise ValueError("未找到 input_file_id。请先执行 upload。")

    completion_window = str(args.completion_window or metadata.get("completion_window") or "24h")
    client = make_client(model_id)
    batch_id = create_batch(client, str(input_file_id), completion_window=completion_window)
    metadata["batch_id"] = batch_id
    metadata["batch_status"] = "validating"
    metadata["completion_window"] = completion_window
    save_json(meta_json, metadata)
    print(f"create 完成，batch_id: {batch_id}")
    return batch_id


def stage_wait(args: argparse.Namespace, run_dir: Path) -> Any:
    meta_json, metadata = load_meta_or_fail(run_dir)
    model_id = str(metadata["model"])
    batch_id = args.batch_id or metadata.get("batch_id")
    if not batch_id:
        raise ValueError("未找到 batch_id。请先执行 create，或通过 --batch-id 指定。")

    poll_interval = int(args.poll_interval_seconds or metadata.get("poll_interval_seconds") or 10)
    client = make_client(model_id)
    batch = poll_batch(client, str(batch_id), poll_interval_seconds=poll_interval)
    metadata["batch_id"] = str(batch_id)
    metadata["batch_status"] = batch.status
    save_json(meta_json, metadata)
    print(f"wait 完成，最终状态: {batch.status}")
    return batch


def stage_collect(args: argparse.Namespace, run_dir: Path, batch_obj: Any | None = None) -> None:
    meta_json, metadata = load_meta_or_fail(run_dir)
    model_id = str(metadata["model"])
    input_file = str(metadata["input_file"])
    batch_id = args.batch_id or metadata.get("batch_id")
    if not batch_id:
        raise ValueError("未找到 batch_id。请先执行 create，或通过 --batch-id 指定。")

    client = make_client(model_id)
    batch = batch_obj if batch_obj is not None else client.batches.retrieve(str(batch_id))
    metadata["batch_id"] = str(batch_id)
    metadata["batch_status"] = batch.status

    if batch.status != "completed":
        save_json(meta_json, metadata)
        raise RuntimeError(f"batch 状态不是 completed，当前为: {batch.status}")

    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        save_json(meta_json, metadata)
        raise RuntimeError("Batch 已完成但 output_file_id 为空")

    output_jsonl = Path(metadata["output_jsonl"])
    error_jsonl = Path(metadata["error_jsonl"])
    results_output_file = str(metadata["responses_file"])
    question_payloads_json = Path(metadata["question_payloads_json"])
    if not question_payloads_json.exists():
        raise FileNotFoundError(f"question_payloads.json 不存在: {question_payloads_json}")

    question_payloads = load_json(question_payloads_json)
    output_content = client.files.content(output_file_id)
    output_text = extract_text_from_file_content(output_content)
    save_text(output_jsonl, output_text)
    metadata["output_file_id"] = output_file_id

    error_file_id = getattr(batch, "error_file_id", None)
    if error_file_id:
        error_content = client.files.content(error_file_id)
        error_text = extract_text_from_file_content(error_content)
        save_text(error_jsonl, error_text)
        metadata["error_file_id"] = error_file_id
    else:
        metadata["error_file_id"] = None

    question_results, failed_from_output = parse_batch_output(output_text, question_payloads)
    submitted_question_ids = set(str(v) for v in metadata.get("submitted_question_ids", []))
    if not submitted_question_ids:
        submitted_question_ids = set(question_payloads.keys())
    missing_question_ids = submitted_question_ids - set(question_results.keys())
    failed_question_ids = set(failed_from_output) | missing_question_ids
    success_question_ids = sorted(set(question_results.keys()) - failed_question_ids)

    data_loader = DataLoader(input_file)
    data_loader.load_data(num_tasks=metadata.get("num_tasks"))
    conversations_by_qid = _build_conversation_index(data_loader)

    written_question_ids: list[str] = []
    for question_id in success_question_ids:
        conversation = conversations_by_qid.get(question_id)
        if conversation is None:
            failed_question_ids.add(question_id)
            continue
        result = question_results[question_id]
        data_loader.append_result_record(
            output_file=results_output_file,
            conversation=conversation,
            responses=result["responses"],
            token_count=result.get("token_count"),
            evaluations=[],
            final_status="PENDING",
        )
        written_question_ids.append(question_id)

    if failed_question_ids:
        print(f"[SKIP] 以下问题请求失败，不写入结果文件: {sorted(failed_question_ids)}")

    metadata["success_question_ids"] = written_question_ids
    metadata["failed_question_ids"] = sorted(failed_question_ids)
    metadata["written_questions"] = len(written_question_ids)
    save_json(meta_json, metadata)

    print(f"已写入 generation JSONL: {results_output_file}")
    print(f"本次成功写入: {len(written_question_ids)} 题，失败跳过: {len(failed_question_ids)} 题")
    print(f"中间产物目录: {run_dir}")


def main() -> None:
    args = parse_args()
    step = str(args.step)
    if step == "submit":
        step = "upload"
    elif step == "poll":
        step = "wait"

    if step in {"upload", "create", "wait", "collect"} and not args.run_dir:
        raise ValueError(f"--step {step} 需要传入 --run-dir")

    if step == "prepare":
        stage_prepare(args)
        return
    if step == "upload":
        stage_upload(args, Path(args.run_dir))
        return
    if step == "create":
        stage_create(args, Path(args.run_dir))
        return
    if step == "wait":
        stage_wait(args, Path(args.run_dir))
        return
    if step == "collect":
        stage_collect(args, Path(args.run_dir))
        return

    run_dir = stage_prepare(args)
    stage_upload(args, run_dir)
    stage_create(args, run_dir)
    batch = stage_wait(args, run_dir)
    stage_collect(args, run_dir, batch_obj=batch)


if __name__ == "__main__":
    main()
