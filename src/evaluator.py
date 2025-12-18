from pydantic import BaseModel
from typing import List, Dict, Any, Literal, Optional
import threading
import json
import random
from json_repair import repair_json
from src.models.openai import OpenAIModel
from src.data_loader import ensure_list

class JudgeResponse(BaseModel):
    reasoning: str
    verdict: Literal["YES", "NO"]

JUDGE_PROMPT = '''You are tasked with evaluating a model response to see if it meets a specific criteria.
The criteria will always be YES/NO evaluation.

The model response is as follows:
<MODEL_RESPONSE>
{}
</MODEL_RESPONSE>

The criteria that the model response must meet is as follows. Be VERY STRICT!:
<CRITERIA>
{}
</CRITERIA>

Output your response in the following JSON format (no other text):
{{"reasoning": "your detailed reasoning here", "verdict": "YES or NO"}}'''

class Evaluator:
    def __init__(
        self,
        conversations: List[Any],
        responses: Dict[int, List[str]],
        evaluator_configs: Optional[List[Dict[str, Any]]] = None,
    ):
        self.conversations = conversations
        self.responses = responses
        self._evaluator_configs = evaluator_configs or [{}]
        self._thread_local = threading.local()

    def _parse_judge_response(self, text: str) -> JudgeResponse:
        """Parse judge response: try JSON first, then json_repair, then regex fallback."""
        text = (text or "").strip()
        
        # Try direct JSON parse
        try:
            data = json.loads(text)
            return JudgeResponse(
                reasoning=data.get('reasoning', text),
                verdict=data.get('verdict', 'NO').upper()
            )
        except json.JSONDecodeError:
            pass
        
        # Try json_repair
        try:
            repaired = repair_json(text)
            data = json.loads(repaired)
            return JudgeResponse(
                reasoning=data.get('reasoning', text),
                verdict=data.get('verdict', 'NO').upper()
            )
        except (json.JSONDecodeError, Exception):
            pass
        
        # Fallback: regex-based YES/NO extraction
        upper = text.upper()
        if "YES" in upper and "NO" in upper:
            last_yes = upper.rfind("YES")
            last_no = upper.rfind("NO")
            verdict = "YES" if last_yes > last_no else "NO"
        elif "YES" in upper:
            verdict = "YES"
        elif "NO" in upper:
            verdict = "NO"
        else:
            verdict = "NO"
        return JudgeResponse(reasoning=text, verdict=verdict)

    def _get_eval_models(self) -> List[OpenAIModel]:
        models = getattr(self._thread_local, 'evaluation_models', None)
        if models is None:
            models = [
                OpenAIModel.from_model_config(cfg.get('name', 'gpt-4o'), cfg, max_tokens=4096)
                for cfg in self._evaluator_configs
            ]
            self._thread_local.evaluation_models = models
        return models

    def evaluate_conversation(self, conversation: Any, responses: Any) -> Dict:
        responses = ensure_list(responses)

        evaluations = []
        if not responses:
            evaluations.append({
                'attempt': 0,
                'reasoning': 'NA - Question ID not found in responses',
                'verdict': 'NO',
                'pass_criteria': conversation.pass_criteria,
                'passed': False
            })
        else:
            for attempt_idx, response in enumerate(responses):
                target_question = conversation.target_question
                pass_criteria = conversation.pass_criteria
                prompt = JUDGE_PROMPT.format(response, target_question)

                models = self._get_eval_models()
                model_names = [cfg.get('name', f'evaluator_{i+1}') for i, cfg in enumerate(self._evaluator_configs)]
                judge_results = []
                failed_count = 0
                successful_model_indices = []
                failed_evaluators = []
                backup_used = []
                
                for idx, model in enumerate(models):
                    try:
                        judgement = model.generate([{"role": "user", "content": prompt}])
                        parsed = self._parse_judge_response(str(judgement))
                        judge_results.append(parsed)
                        successful_model_indices.append(idx)
                    except Exception as e:
                        failed_count += 1
                        judge_results.append(None)
                        failed_evaluators.append(model_names[idx])
                
                if failed_count == len(models):
                    raise RuntimeError("All evaluators failed")
                
                while failed_count > 0 and successful_model_indices:
                    backup_idx = random.choice(successful_model_indices)
                    backup_model = models[backup_idx]
                    try:
                        judgement = backup_model.generate([{"role": "user", "content": prompt}])
                        parsed = self._parse_judge_response(str(judgement))
                        for i, r in enumerate(judge_results):
                            if r is None:
                                judge_results[i] = parsed
                                failed_count -= 1
                                backup_used.append(model_names[backup_idx])
                                break
                    except Exception:
                        successful_model_indices.remove(backup_idx)
                        if not successful_model_indices:
                            raise RuntimeError("All evaluators failed during retry")
                
                valid_results = [r for r in judge_results if r is not None]
                yes_count = sum(1 for r in valid_results if r.verdict == "YES")
                no_count = len(valid_results) - yes_count
                final_verdict = "YES" if yes_count > no_count else "NO"

                eval_entry = {
                    'attempt': attempt_idx,
                    'verdict': final_verdict,
                    'pass_criteria': pass_criteria,
                    'passed': final_verdict == pass_criteria
                }
                for i, jr in enumerate(judge_results):
                    if jr is not None:
                        eval_entry[f'judge_{i+1}_verdict'] = jr.verdict
                        eval_entry[f'judge_{i+1}_reasoning'] = jr.reasoning
                    else:
                        eval_entry[f'judge_{i+1}_verdict'] = 'ERROR'
                        eval_entry[f'judge_{i+1}_reasoning'] = 'Evaluator failed'

                evaluations.append(eval_entry)

        attempts = len(evaluations)
        passes = sum(1 for e in evaluations if e.get('passed'))
        final_status = f"{'PASS' if passes > 0 else 'FAIL'} ({passes}/{attempts} attempts passed)"

        result = {
            'QUESTION_ID': conversation.question_id,
            'AXIS': conversation.axis,
            'TARGET_QUESTION': conversation.target_question,
            'PASS_CRITERIA': conversation.pass_criteria,
            'RESPONSES': responses,
            'EVALUATIONS': evaluations,
            'FINAL_STATUS': final_status
        }
        if responses and 'failed_evaluators' in locals() and failed_evaluators:
            result['_failed_evaluators'] = failed_evaluators
            result['_backup_used'] = backup_used
        return result