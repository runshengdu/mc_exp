# MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs

MultiChallenge is a novel benchmark designed to evaluate large language models (LLMs) on their ability to handle multi-turn conversations with human users—an essential but underexplored capability for their real-world applications. MultiChallenge focuses on four key categories of challenges that are common, realistic, and highly demanding in current human-LLM interactions. These challenges require LLMs to excel simultaneously in accurate context allocation, in-context reasoning, and instruction-following.

## **Project Structure**

```
├── data/
│   ├── benchmark_questions.jsonl      # Benchmark conversation dataset
│   ├── response_template.jsonl        # Template for response file format
│   └── final_model_responses/         # Pre-generated model responses
├── output/                            # Generated model responses output
├── results/                           # Evaluation results output
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
   ```

## **Usage**

The benchmark operates in two modes: **Response Generation** and **Evaluation**.

### **1. Generate Model Responses**

Generate responses for benchmark conversations using a model defined in `models.yaml`:

```bash
python main.py --model-id doubao-seed-1-8-251228
```

- Responses are saved to `responses/<model_id>_<timestamp>.jsonl`
- Supports checkpoint resumption: if the output file exists, already-completed questions are skipped

Specify a custom output file:
```bash
python main.py --model-id openai/gpt-5.2 --responses-file my_responses.json
```

Limit the number of tasks:
```bash
python main.py --model-id deepseek-chat --num-tasks 10
```

### **2. Evaluate Responses**

Evaluate pre-generated responses using a multi-judge system (3 evaluator models by default):

```bash
python main.py --evaluate-file output/my_responses.json
```

- Evaluation uses comma-separated evaluator ids via `--evaluator` (allow 1, 3, or 5 models; default `deepseek-chat,glm-4.6,kimi-k2-0905-preview`)
- Final verdict requires at least half-plus-one YES votes (majority for odd-sized panels)
- Results are saved to `results/<responses_file_stem>_evaluation.jsonl` by default

Specify custom evaluators:
```bash
python main.py --evaluate-file output/responses.json \
    --evaluator deepseek-chat,glm-4.6,kimi-k2-0905-preview
```

### **3. Parallel Processing**

Use multiple workers for faster processing:
```bash
python main.py --model-id anthropic/claude-sonnet-4.5 --max-workers 10
python main.py --evaluate-file output/responses.json --max-workers 5
```

### **Command-Line Arguments**

| Argument | Description | Default |
|----------|-------------|---------|
| `--model-id` | Model ID for response generation (must exist in `models.yaml`) | `doubao-seed-1-8-251228` |
| `--responses-file` | Custom path for response output file | Auto-generated |
| `--evaluate-file` | Path to responses file for evaluation | - |
| `--output-file` | Path to save evaluation results | Auto-generated |
| `--evaluator` | Comma-separated evaluator model IDs (1, 3, or 5 entries; must exist in `models.yaml`) | `deepseek-chat,glm-4.6,kimi-k2-0905-preview` |
| `--num-tasks` | Limit to first N tasks from benchmark | All tasks |
| `--max-workers` | Number of parallel workers for LLM calls | `30` |

### **Evaluation System**

The evaluation uses a **multi-judge voting** system:

1. Each response is evaluated by an odd-sized panel of judge models (1, 3, or 5) configured via `--evaluator`
2. Each judge outputs a `YES` or `NO` verdict based on the pass criteria
3. Final verdict requires at least half-plus-one YES votes (e.g., 1/1, 2/3, or 3/5)
4. A conversation passes if the final verdict matches the expected `PASS_CRITERIA`

### **Output Format**

**Response File** (`output/*.json`):
```json
{"QUESTION_ID": 1, "RESPONSE": ["model response text"], "TOKEN_COUNT": 1234}
```

**Evaluation File** (`results/*_evaluation.json`):
```json
{
    "SUMMARY": {
        "overall_score": 75.5,
        "axis_scores": {"REFINEMENT": 80.0, "COHERENCE": 70.0, ...}
    }
}
{"question_id": 1, "axis": "REFINEMENT", "passed": true, "judge_1_verdict": "YES", ...}
```

## **Model Configuration**

Models are configured in `models.yaml`. Each model entry includes:

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
