from collections import defaultdict
from typing import List, Dict
import json
import os
import time
from src.data_loader import ensure_list, sanitize_unusual_line_terminators

class ResultParser:
    def __init__(self, evaluation_results):
        self.evaluation_results = evaluation_results

    def calculate_scores(self):
        # Group results by question_id
        question_results = defaultdict(list)
        for result in self.evaluation_results:
            question_results[result['question_id']].append(result['passed'])  # Use 'passed' instead of checking verdict

        # Initialize axis_counts with a set for tracking counted questions
        axis_counts = defaultdict(lambda: {'passed': 0, 'total': 0, 'counted': set()})
        
        for result in self.evaluation_results:
            question_id = result['question_id']
            axis = result['axis']
            
            # Only count each question once per axis
            if question_id not in axis_counts[axis]['counted']:
                axis_counts[axis]['total'] += 1
                axis_counts[axis]['counted'].add(question_id)
                
                # If any attempt for this question passed, count it as passed
                if any(r['passed'] for r in self.evaluation_results 
                      if r['question_id'] == question_id and r['axis'] == axis):
                    axis_counts[axis]['passed'] += 1

        axis_scores = {axis: (scores['passed'] / scores['total']) * 100 for axis, scores in axis_counts.items()}
        overall_score = sum(axis_scores.values())/len(axis_scores.keys()) if axis_scores else 0.0

        return {
            "overall_score": overall_score,
            "axis_scores": axis_scores
        }
    
    @staticmethod
    def _parse_json_objects(content: str):
        """Parse multiple JSON objects from a string (supports both JSONL and pretty-printed)."""
        objects = []
        decoder = json.JSONDecoder()
        idx = 0
        length = len(content)
        while idx < length:
            while idx < length and content[idx].isspace():
                idx += 1
            if idx >= length:
                break
            try:
                obj, end_idx = decoder.raw_decode(content, idx)
                objects.append(obj)
                idx = end_idx
            except json.JSONDecodeError:
                idx += 1
        return objects

    @staticmethod
    def prepend_summary_jsonl(output_file: str) -> None:
        if not os.path.exists(output_file):
            return

        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()

        all_objects = ResultParser._parse_json_objects(content)
        records = [obj for obj in all_objects if isinstance(obj, dict) and 'summary' not in obj]

        eval_rows = []
        for obj in records:
            if not isinstance(obj, dict):
                continue
            question_id = obj.get('question_id')
            axis = obj.get('axis')
            passed = obj.get('passed')
            if question_id is not None and axis is not None and passed is not None:
                eval_rows.append({
                    'question_id': question_id,
                    'axis': axis,
                    'passed': bool(passed)
                })
                continue
            question_id = obj.get('question_id')
            axis = obj.get('axis')
            evaluations = obj.get('evaluations')
            if question_id is None or axis is None:
                continue
            evaluations = ensure_list(evaluations)
            if not evaluations:
                continue
            passed_any = any(isinstance(e, dict) and bool(e.get('passed')) for e in evaluations)
            eval_rows.append({
                'question_id': question_id,
                'axis': axis,
                'passed': passed_any
            })

        summary = ResultParser(eval_rows).calculate_scores()

        tmp_path = output_file + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(sanitize_unusual_line_terminators({'summary': summary}), ensure_ascii=False, indent=4) + "\n")
            for obj in records:
                f.write(json.dumps(sanitize_unusual_line_terminators(obj), ensure_ascii=False, indent=4) + "\n")
        for attempt in range(5):
            try:
                os.replace(tmp_path, output_file)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.2 * (2 ** attempt))
    
    @staticmethod
    def append_evaluation_record(output_file: str, conv, record: dict, token_count: int = None) -> None:
        """Append evaluation record for a single conversation to output file."""
        conv_responses = ensure_list(record.get('responses'))

        conv_results = ensure_list(record.get('evaluations'))
        attempts = max(1, len(conv_responses), len(conv_results))
        original_conversation = conv.conversation
        passed_attempts = sum(1 for r in conv_results if isinstance(r, dict) and r.get('passed'))
        final_result = 'PASS' if passed_attempts > 0 else 'FAIL'

        with open(output_file, 'a', encoding='utf-8') as f:
            for i in range(attempts):
                result = conv_results[i] if i < len(conv_results) else {}

                model_response = conv_responses[i] if i < len(conv_responses) else 'N/A'
                if isinstance(model_response, str):
                    model_response = model_response.splitlines()

                row = {
                    'question_id': conv.question_id,
                    'axis': conv.axis,
                    'original_conversation': original_conversation,
                    'target_question': conv.target_question,
                    'pass_criteria': f"Response should receive a {conv.pass_criteria} verdict",
                    'attempt_number': i + 1,
                    'model_response': model_response,
                    'judge_verdict': result.get('verdict', 'N/A') if isinstance(result, dict) else 'N/A',
                    'passed': bool(result.get('passed', False)) if isinstance(result, dict) else False,
                    'final_result': f"{final_result} ({passed_attempts}/{attempts} attempts passed)"
                }
                if token_count is not None:
                    row['token_count'] = token_count
                for j in range(1, 4):
                    v = result.get(f'judge_{j}_verdict', 'N/A') if isinstance(result, dict) else 'N/A'
                    r = result.get(f'judge_{j}_reasoning', 'N/A') if isinstance(result, dict) else 'N/A'
                    if isinstance(r, str):
                        r = r.splitlines()
                    row[f'judge_{j}_verdict'] = v
                    row[f'judge_{j}_reasoning'] = r
                row = sanitize_unusual_line_terminators(row)
                f.write(json.dumps(row, ensure_ascii=False, indent=4) + "\n")
            f.flush()
