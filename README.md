# MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs

MultiChallenge is a novel benchmark designed to evaluate large language models (LLMs) on their ability to handle multi-turn conversations with human users—an essential but underexplored capability for their real-world applications. MultiChallenge focuses on four key categories of challenges that are common, realistic, and highly demanding in current human-LLM interactions. These challenges require LLMs to excel simultaneously in accurate context allocation, in-context reasoning, and instruction-following.

## **Project Structure**

```
├── data/
│   ├── benchmark_questions.jsonl      # Benchmark conversation dataset
│   ├── response_template.jsonl        # Template for response file format
│   └── final_model_responses/         # Pre-generated model responses
├── results/                           # Generated responses + evaluation output
├── src/
│   ├── models/
│   │   ├── base.py                    # Abstract ModelProvider base class
│   │   └── openai.py                  # OpenAI-compatible API model provider
│   ├── conversation.py                # Conversation dataclass definition
│   ├── data_loader.py                 # Data loading and response generation
│   ├── evaluator.py                   # Multi-judge evaluation logic
│   └── result_parser.py               # Result parsing and summary generation
├── main.py                            # Main entry point
├── models.yaml                        # Model configurations (API keys, endpoints)
└── requirements.txt                   # Python dependencies
```

## **Setup Instructions**

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd multi-challenge
   ```

2. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**
   
   Create a `.env` file in the root directory with your API keys. The keys are referenced in `models.yaml`:
   ```plaintext
   OPENROUTER_API_KEY=your-openrouter-api-key
   DEEPSEEK_API_KEY=your-deepseek-api-key
   GLM_API_KEY=your-glm-api-key
   MOONSHOT_API_KEY=your-moonshot-api-key
   MINIMAX_API_KEY=your-minimax-api-key
   ARK_API_KEY=your-ark-api-key
   ```

## **Usage**

The benchmark operates in two modes: **Response Generation** and **Evaluation**.

### **1. Generate Model Responses**

Generate responses for benchmark conversations using a model defined in `models.yaml`:

```bash
python main.py --model-id doubao-seed-1-8-251228
```

- Responses are saved to `results/<model_id>/<timestamp>.jsonl`
- Supports checkpoint resumption: if the output file exists, already-completed questions are skipped

Specify a custom output file:
```bash
python main.py --model-id openai/gpt-5.2 --responses-file results/my_responses.jsonl
```

Limit the number of tasks:
```bash
python main.py --model-id deepseek-chat --num-tasks 10
```

### **2. Evaluate Responses**

Evaluate pre-generated responses using a multi-judge system (3 evaluator models by default):

```bash
python main.py --evaluate-file results/my_responses.jsonl
```

- Evaluation uses comma-separated evaluator ids via `--evaluator` (allow 1, 3, or 5 models; default `deepseek-chat,glm-4.7,kimi-k2.5`)
- Final verdict requires at least half-plus-one YES votes (majority for odd-sized panels)
- Results are written **in-place** back to the same JSONL file

Specify custom evaluators:
```bash
python main.py --evaluate-file results/responses.jsonl \
    --evaluator deepseek-chat,glm-4.7,kimi-k2.5
```

### **3. Parallel Processing**

Use multiple workers for faster processing:
```bash
python main.py --model-id anthropic/claude-sonnet-4.5 --gen-max-workers 10
python main.py --evaluate-file results/responses.jsonl --eval-max-workers 5
```

### **Command-Line Arguments**

| Argument | Description | Default |
|----------|-------------|---------|
| `--model-id` | Model ID for response generation (must exist in `models.yaml`) | `kimi-k2.5` |
| `--responses-file` | Custom path for generation results file (JSONL) | `results/<model_id>/<timestamp>.jsonl` |
| `--evaluate-file` | Path to generation results file for evaluation (in-place update) | - |
| `--evaluator` | Comma-separated evaluator model IDs (1, 3, or 5 entries; must exist in `evaluators.yaml`) | `deepseek-chat,glm-4.7,kimi-k2.5` |
| `--num-tasks` | Limit to first N tasks from benchmark | All tasks |
| `--gen-max-workers` | Number of parallel workers for response generation | `100` |
| `--eval-max-workers` | Number of parallel workers for evaluation | `40` |

### **Evaluation System**

The evaluation uses a **multi-judge voting** system:

1. Each response is evaluated by an odd-sized panel of judge models (1, 3, or 5) configured via `--evaluator`
2. Each judge outputs a `YES` or `NO` verdict based on the pass criteria
3. Final verdict requires at least half-plus-one YES votes (e.g., 1/1, 2/3, or 3/5)
4. A conversation passes if the final verdict matches the expected `PASS_CRITERIA`

### **Output Format**

**Generation Results File** (`results/<model_id>/<timestamp>.jsonl`):
```json
{
  "question_id": 1,
  "axis": "REFINEMENT",
  "original_conversation": ["..."],
  "target_question": "...",
  "pass_criteria": "YES",
  "responses": ["model response text"],
  "evaluations": [],
  "final_status": "PENDING",
  "token_count": 1234
}
```

**Evaluation Results (same JSONL file, updated in-place)**:
```json
{
  "summary": {
    "overall_score": 75.5,
    "axis_scores": {"REFINEMENT": 80.0, "COHERENCE": 70.0, "...": 0.0}
  }
}
{
  "question_id": 1,
  "axis": "REFINEMENT",
  "responses": ["..."],
  "evaluations": [
    {
      "attempt": 0,
      "verdict": "YES",
      "pass_criteria": "YES",
      "passed": true,
      "judge_1_verdict": "YES",
      "judge_1_reasoning": "..."
    }
  ],
  "final_status": "PASS (1/1 attempts passed)"
}
```

## **Model Configuration**

Models for **response generation** are configured in `models.yaml`. Evaluators are configured in `evaluators.yaml`. Each model entry includes:

```yaml
models:
  - name: model-name
    temperature: 0.0
    base_url: https://api.example.com/v1
    api_key: "${ENV_VAR_NAME}"
    extra_body: {}  # Optional API-specific parameters
    cost_1m_token_dollar:
      prompt_price: 2.00
      completion_price: 4.00
```

## **Benchmark Axes**

MultiChallenge evaluates models across four axes:
- **REFINEMENT**: Ability to refine responses based on user feedback
- **EXPLICIT IF**: Handling explicit conditional instructions
- **COHERENCE**: Maintaining coherent context across turns
- **RECOLLECTION**: Recalling information from earlier in the conversation

## **Project Dependencies**

```
pydantic
python-dotenv
tqdm
openai
pyyaml
json-repair
```

See `requirements.txt` for the complete list.
