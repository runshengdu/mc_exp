# MultiChallenge

MultiChallenge evaluates large language models on realistic multi-turn conversations.  
The pipeline supports:

- response generation from benchmark conversations
- in-place multi-judge evaluation
- optional Batch API generation flow for Qwen and Moonshot models

## Project Structure

```text
├── data/
│   ├── benchmark_questions.jsonl        # Benchmark dataset
│   └── response_template.jsonl          # Reference output schema
├── src/
│   ├── models/
│   │   ├── base.py                      # Model provider interface
│   │   └── openai.py                    # OpenAI-compatible client wrapper
│   ├── conversation.py                  # Conversation dataclass
│   ├── data_loader.py                   # Dataset/response I/O and generation checkpointing
│   ├── evaluator.py                     # Multi-judge evaluation and voting
│   ├── result_parser.py                 # Summary aggregation
│   └── utils.py                         # Shared concurrency helpers
├── batch_api/
│   ├── batch.py                         # Qwen / Moonshot Batch API pipeline
│   └── artifacts/                       # Intermediate batch run artifacts
├── main.py                              # Main CLI (generate/evaluate)
├── models.yaml                          # Generation model configs
├── evaluators.yaml                      # Evaluator model configs
└── requirements.txt                     # Python dependencies
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Export API keys used by your selected entries in `models.yaml` and `evaluators.yaml`.

Example:

```bash
export OPENROUTER_API_KEY=...
export DEEPSEEK_API_KEY=...
export GLM_API_KEY=...
export MOONSHOT_API_KEY=...
export MINIMAX_API_KEY=...
export ARK_API_KEY=...
export DASHSCOPE_API_KEY=...
```

`main.py` resolves `${ENV_VAR}` placeholders in YAML and raises an error if a referenced env var is missing.

## Main CLI Usage (`main.py`)

The main workflow has two modes.

### 1) Generate Responses

```bash
python main.py --model-id kimi-k2.6
```

- Default output path: `results/<model_id>/<YYYYmmdd_HHMM>.jsonl`
- If output file already exists, completed `question_id`s are skipped (checkpoint resume)

Custom output file:

```bash
python main.py --model-id openai/gpt-5.2 --responses-file results/my_responses.jsonl
```

Limit benchmark size:

```bash
python main.py --model-id deepseek-v4-flash --num-tasks 10
```

### 2) Evaluate Existing Responses

```bash
python main.py --evaluate-file results/my_responses.jsonl
```

- Evaluates only records without non-empty `evaluations`
- Updates the same file in place
- Prepends a top-level summary object after evaluation

Custom evaluator panel (must be 1, 3, or 5 models):

```bash
python main.py --evaluate-file results/my_responses.jsonl --evaluator qwen3.6-flash,google/gemini-3-flash-preview,deepseek-v4-flash
```

### Common Flags

- `--model-id` (default: `kimi-k2.5`)
- `--responses-file`
- `--evaluate-file`
- `--evaluator` (default: `qwen3.6-flash,google/gemini-3-flash-preview,deepseek-v4-flash`)
- `--num-tasks`
- `--gen-max-workers` (default: `50`)
- `--eval-max-workers` (default: `30`)

## Evaluation Logic

- Each response attempt is judged by an odd-numbered evaluator panel (1/3/5).
- Each judge returns `YES`/`NO`.
- Final verdict uses majority voting.
- `passed = (final_verdict == pass_criteria)`.
- `final_status` is `PASS` if any attempt passes, otherwise `FAIL`.

If an evaluator call fails, the pipeline retries by reusing successful evaluators as backups. If all evaluators fail for a question, evaluation exits with error.

## Output File Format

Generation writes one record per question with:

- `question_id`, `axis`, `original_conversation`, `target_question`, `pass_criteria`
- `responses`
- `evaluations` (initially empty)
- `final_status` (initially `PENDING`)
- optional `token_count`

After evaluation, the same file includes:

- a first JSON object: `{"summary": {"overall_score": ..., "axis_scores": {...}}}`
- per-question records with populated `evaluations` and final pass/fail status

## Batch API Workflow

`batch_api/batch.py` supports Qwen and Moonshot models via `--model-id`.

Staged execution:

- `prepare` -> build batch input and metadata
- `upload` -> upload input file
- `create` -> create batch job
- `wait` -> poll until terminal status
- `collect` -> download results and append successful responses into generation JSONL
- `cancel` -> cancel an in-progress batch job

Or run all stages in one command:

```bash
python batch_api/batch.py --step all --model-id qwen3.6-flash
python batch_api/batch.py --step all --model-id kimi-k2.6
```

Notes:

- `--model-id` is required.
- Intermediate artifacts are stored under `batch_api/artifacts/<model_id>/...`.
- `collect` appends only successful questions; failed/missing ones are logged in `meta.json`.
- For staged runs, pass `--run-dir` for `upload/create/wait/collect/cancel`.

## Model Configuration

`models.yaml` and `evaluators.yaml` use:

```yaml
models:
  - name: model-name
    temperature: 1.0
    base_url: https://api.example.com/v1
    api_key: "${ENV_VAR_NAME}"
    max_tokens: 60000
    extra_body: {}  # optional provider-specific params
```

The implementation uses OpenAI-compatible chat completions (`/v1/chat/completions`).

## Dependencies

From `requirements.txt`:

- `pydantic`
- `tqdm`
- `openai`
- `pyyaml`
- `json-repair`
