#!/usr/bin/env python3
"""Analyze failed benchmark cases with deepseek-v4-pro and save a structured summary."""

import argparse
import json
import os
import re
import sys
from datetime import datetime

import tiktoken

from main import _models_config_path, load_model_config
from src.models.openai import OpenAIModel
from src.result_parser import ResultParser

ANALYZER_MODEL_ID = "google/gemini-3.5-flash"
BATCH_TOKEN_THRESHOLD = 200_000
OUTPUT_DIR = "bad_case_analysis"
_RESULTS_MODEL_RE = re.compile(r"(?:^|[\\/])results[\\/]([^/\\]+)[\\/]", re.IGNORECASE)
_TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")

OUTPUT_FORMAT_INSTRUCTIONS = """
请严格按以下 Markdown 结构用中文回答（不要省略章节标题）：

## 一、总体概况
- 失败案例数：（写明是本批还是全部）
- 涉及 axis 及数量：（列表）

## 二、失败原因分类
（按重要性排序，每类一节）
### 类别N：<类别名称>
- 数量：
- 占比：（相对本批或全部失败案例）
- 典型表现：（1-3 句）
- 代表 question_id：（最多 3 个，附 axis）

## 三、按 axis 归纳
（每个 axis 一小段：主要失败模式）

## 四、其他发现
（可选；无则写「无」）
"""

SYSTEM_PROMPT = (
    "你是一位 LLM 评测分析专家。根据提供的失败案例完整 JSON 记录，"
    "归纳模型未通过判题的根本原因（关注模型行为与判题标准之间的差距，"
    "不要简单复述 judge 原文）。"
)


def count_tokens(text: str) -> int:
    return len(_TIKTOKEN_ENC.encode(text))


def resolve_results_path(path: str) -> str:
    if os.path.isfile(path):
        return os.path.abspath(path)
    root = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(root, path)
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)
    raise FileNotFoundError(f"Results file not found: {path}")


def is_summary_record(obj: dict) -> bool:
    if obj.get("question_id") is not None:
        return False
    return "SUMMARY" in obj or "summary" in obj


def load_records(file_path: str) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    objects = ResultParser._parse_json_objects(content)
    return [o for o in objects if isinstance(o, dict) and not is_summary_record(o)]


def extract_bad_cases(records: list) -> list:
    """Each record with passed=false counts as one bad case (flat or nested format)."""
    bad_cases = []
    for obj in records:
        if obj.get("question_id") is None:
            continue

        evaluations = obj.get("evaluations")
        if isinstance(evaluations, list) and evaluations:
            for ev in evaluations:
                if not isinstance(ev, dict) or ev.get("passed") is not False:
                    continue
                attempt = ev.get("attempt", 0)
                idx = int(attempt) if attempt is not None else 0
                case = dict(obj)
                case["_failed_attempt_index"] = idx
                case["_failed_evaluation"] = ev
                bad_cases.append(case)
            continue

        if obj.get("passed") is False:
            bad_cases.append(obj)

    return bad_cases


def cases_payload(cases: list) -> str:
    return json.dumps(cases, ensure_ascii=False, indent=2)


def build_batch_prompt(cases: list, batch_index: int, batch_total: int) -> str:
    return (
        f"以下为本评测结果文件中第 {batch_index}/{batch_total} 批失败案例的完整 JSON 记录"
        f"（共 {len(cases)} 条）。请归纳本批失败原因。\n\n"
        f"{OUTPUT_FORMAT_INSTRUCTIONS}\n\n"
        f"```json\n{cases_payload(cases)}\n```"
    )


def build_single_prompt(cases: list) -> str:
    return (
        f"以下是全部 {len(cases)} 条失败案例的完整 JSON 记录。请归纳所有失败原因。\n\n"
        f"{OUTPUT_FORMAT_INSTRUCTIONS}\n\n"
        f"```json\n{cases_payload(cases)}\n```"
    )


def build_merge_prompt(batch_summaries: list) -> str:
    parts = []
    for i, summary in enumerate(batch_summaries, start=1):
        parts.append(f"### 第 {i} 批归纳\n\n{summary}")
    joined = "\n\n".join(parts)
    return (
        f"以下是对同一评测文件分 {len(batch_summaries)} 批分析后得到的归纳结果。"
        f"请合并为一份全局报告：统一分类名称、合并计数、去重代表案例。\n\n"
        f"{OUTPUT_FORMAT_INSTRUCTIONS}\n\n"
        f"{joined}"
    )


def split_into_batches(cases: list, threshold: int) -> list:
    """Greedy pack cases so each batch's JSON payload stays within threshold."""
    batches = []
    current = []

    for case in cases:
        trial = current + [case]
        if current and count_tokens(cases_payload(trial)) > threshold:
            batches.append(current)
            current = [case]
        else:
            current = trial

    if current:
        batches.append(current)
    return batches


def estimate_input_tokens(case_tokens: int, num_batches: int) -> int:
    system_tokens = count_tokens(SYSTEM_PROMPT)
    if num_batches <= 1:
        prompt_shell = count_tokens(build_single_prompt([]).split("```json")[0])
        return system_tokens + prompt_shell + case_tokens

    batch_shell = count_tokens(
        build_batch_prompt([], 1, num_batches).split("```json")[0]
    )
    merge_shell = count_tokens(build_merge_prompt(["（批次摘要占位）"]).split("###")[0])
    per_batch_cases = case_tokens // num_batches
    merge_body = num_batches * 2500
    return (
        num_batches * (system_tokens + batch_shell + per_batch_cases)
        + system_tokens
        + merge_shell
        + merge_body
    )


def confirm_run(api_calls: int, estimated_input_tokens: int, bad_count: int) -> bool:
    print("--- 运行确认 ---")
    print(f"结果文件失败案例数: {bad_count}")
    print(f"分析模型: {ANALYZER_MODEL_ID}")
    print(f"分批阈值（tiktoken）: {BATCH_TOKEN_THRESHOLD:,}")
    print(f"预计 API 调用次数: {api_calls}")
    print(f"输入 token 估算: {estimated_input_tokens:,}")
    resp = input("确认继续？[y/N]: ").strip().lower()
    return resp in ("y", "yes")


def run_analysis(model: OpenAIModel, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return model.generate(messages)


def project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def extract_model_name_from_path(path: str) -> str:
    """Extract model folder name from paths like results/minimax-m3/foo.jsonl."""
    normalized = path.replace("\\", "/")
    match = _RESULTS_MODEL_RE.search(normalized)
    if match:
        return match.group(1)
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return parent or "unknown"


def sanitize_filename_component(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name)


def build_markdown_report(
    report: str,
    *,
    source_file: str,
    bad_count: int,
    total_records: int,
    analyzed_at: datetime,
) -> str:
    return (
        "# Bad Case 分析报告\n\n"
        f"- 源文件: `{source_file}`\n"
        f"- 分析时间: {analyzed_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- 分析模型: {ANALYZER_MODEL_ID}\n"
        f"- 任务记录数: {total_records}\n"
        f"- 失败案例数: {bad_count}\n\n"
        "---\n\n"
        f"{report.strip()}\n"
    )


def save_report(markdown: str, model_name: str, analyzed_at: datetime) -> str:
    out_dir = os.path.join(project_root(), OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    safe_model = sanitize_filename_component(model_name)
    filename = f"{safe_model}_{analyzed_at.strftime('%Y%m%d_%H%M')}.md"
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze passed=false cases in a results jsonl file using deepseek-v4-pro."
    )
    parser.add_argument(
        "results_path",
        help='Path to results jsonl, e.g. results\\model\\file_evaluation.jsonl',
    )
    args = parser.parse_args()

    file_path = resolve_results_path(args.results_path)
    records = load_records(file_path)
    bad_cases = extract_bad_cases(records)

    if not bad_cases:
        print("未找到 passed=false 的失败案例。")
        return 0

    payload_tokens = count_tokens(cases_payload(bad_cases))
    print(f"文件: {file_path}")
    print(f"任务记录数: {len(records)}")
    print(f"失败案例数: {len(bad_cases)}")
    print(f"失败案例 JSON token 估算: {payload_tokens:,}")

    if payload_tokens <= BATCH_TOKEN_THRESHOLD:
        batches = [bad_cases]
        api_calls = 1
    else:
        batches = split_into_batches(bad_cases, BATCH_TOKEN_THRESHOLD)
        api_calls = len(batches) + 1

    est_input_tokens = estimate_input_tokens(payload_tokens, len(batches))
    if not confirm_run(api_calls, est_input_tokens, len(bad_cases)):
        print("已取消。")
        return 0

    model_cfg = load_model_config(ANALYZER_MODEL_ID, _models_config_path())
    model = OpenAIModel.from_model_config(ANALYZER_MODEL_ID, model_cfg)

    if len(batches) == 1:
        print("\n正在调用 API（单次完整分析）...\n")
        report = run_analysis(model, build_single_prompt(bad_cases))
    else:
        print(f"\n输入超过阈值，分 {len(batches)} 批分析...\n")
        batch_summaries = []
        for i, batch in enumerate(batches, start=1):
            print(f"  批次 {i}/{len(batches)}（{len(batch)} 条）...")
            summary = run_analysis(
                model, build_batch_prompt(batch, i, len(batches))
            )
            batch_summaries.append(summary)
        print("  正在合并各批归纳...")
        report = run_analysis(model, build_merge_prompt(batch_summaries))

    analyzed_at = datetime.now()
    model_name = extract_model_name_from_path(file_path)
    markdown = build_markdown_report(
        report,
        source_file=file_path,
        bad_count=len(bad_cases),
        total_records=len(records),
        analyzed_at=analyzed_at,
    )
    out_path = save_report(markdown, model_name, analyzed_at)

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    print(f"\n报告已保存: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
