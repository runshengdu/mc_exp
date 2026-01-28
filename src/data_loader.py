import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from tqdm import tqdm

from src.conversation import Conversation
from src.models.base import ModelProvider
from src.utils import submit_with_mapping

def ensure_list(value: Any) -> List:
    """Ensure value is a list. If string, wrap in list. If None, return empty list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return value


def sanitize_unusual_line_terminators(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\u2028", "\n").replace("\u2029", "\n")
    if isinstance(value, list):
        return [sanitize_unusual_line_terminators(v) for v in value]
    if isinstance(value, tuple):
        return tuple(sanitize_unusual_line_terminators(v) for v in value)
    if isinstance(value, dict):
        return {k: sanitize_unusual_line_terminators(v) for k, v in value.items()}
    return value


def _parse_json_objects(content: str, response_file: str):
    """Parse one or more JSON objects from the given string, raising on the first error."""
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
            # Char position is 1-indexed to match typical error messages
            raise ValueError(f"Invalid JSON starting at char {idx + 1} in {response_file}: {e}") from e
        objs.append(obj)
        idx = end_idx

    return objs


class DataLoader:
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.conversations: List[Conversation] = []
        self.responses: Dict[int, List[str]] = {}
        self.token_counts: Dict[int, int] = {}

    def load_data(self, num_tasks: int = None):
        """Loads input data and creates Conversation objects."""
        with open(self.input_file, 'r') as f:
            for i, line in enumerate(f):
                if num_tasks is not None and i >= num_tasks:
                    break
                data = json.loads(line)
                question_id = data.get('question_id')
                if question_id is None:
                    question_id = data.get('QUESTION_ID')
                axis = data.get('axis')
                if axis is None:
                    axis = data.get('AXIS')
                conversation_data = data.get('conversation')
                if conversation_data is None:
                    conversation_data = data.get('CONVERSATION')
                target_question = data.get('target_question')
                if target_question is None:
                    target_question = data.get('TARGET_QUESTION')
                pass_criteria = data.get('pass_criteria')
                if pass_criteria is None:
                    pass_criteria = data.get('PASS_CRITERIA')
                if question_id is None or axis is None or conversation_data is None or target_question is None or pass_criteria is None:
                    continue
                conversation = Conversation(
                    question_id=question_id,
                    axis=axis,
                    conversation=conversation_data,
                    target_question=target_question,
                    pass_criteria=pass_criteria
                )
                self.conversations.append(conversation)

    def load_responses(self, response_file):
        """Loads model responses from the provided file."""
        if response_file:
            with open(response_file, 'r', encoding='utf-8') as f:
                content = f.read()

            self.responses = {}
            self.token_counts = {}

            items = _parse_json_objects(content, response_file)
            for item in items:
                if not isinstance(item, dict):
                    continue
                if 'summary' in item or 'SUMMARY' in item:
                    continue
                qid = item.get('question_id')
                if qid is None:
                    qid = item.get('QUESTION_ID')
                resp = item.get('responses')
                if resp is None:
                    resp = item.get('RESPONSES')
                if resp is None:
                    resp = item.get('response')
                if resp is None:
                    resp = item.get('RESPONSE')
                if qid is None or resp is None:
                    continue
                self.responses[qid] = ensure_list(resp)
                if 'token_count' in item:
                    self.token_counts[qid] = item['token_count']
                elif 'TOKEN_COUNT' in item:
                    self.token_counts[qid] = item['TOKEN_COUNT']
        return self.responses

    def save_responses(self, output_file: str) -> None:
        """Saves model responses to the provided file in JSONL format."""
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for question_id, response in self.responses.items():
                record = {"question_id": question_id, "response": ensure_list(response)}
                if question_id in self.token_counts:
                    record["token_count"] = self.token_counts[question_id]
                record = sanitize_unusual_line_terminators(record)
                f.write(json.dumps(record, ensure_ascii=False, indent=4) + "\n")

    def append_response(self, output_file: str, question_id, response, token_count: int = None) -> None:
        """Appends a single question's responses to a JSONL file."""
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        record = {"question_id": question_id, "response": ensure_list(response)}
        if token_count is not None:
            record["token_count"] = token_count
        record = sanitize_unusual_line_terminators(record)
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, indent=4) + "\n")

    def append_result_record(
        self,
        output_file: str,
        conversation: Conversation,
        responses,
        token_count: int = None,
        evaluations=None,
        final_status: str = "PENDING",
    ) -> None:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        record = {
            "question_id": conversation.question_id,
            "axis": conversation.axis,
            "original_conversation": conversation.conversation,
            "target_question": conversation.target_question,
            "pass_criteria": conversation.pass_criteria,
            "responses": ensure_list(responses),
            "evaluations": ensure_list(evaluations),
            "final_status": final_status,
        }
        if token_count is not None:
            record["token_count"] = token_count
        record = sanitize_unusual_line_terminators(record)
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, indent=4) + "\n")

    def generate_responses_with_checkpoint(
        self,
        model_provider: ModelProvider,
        output_file: str,
        attempts: int = 1,
        max_workers: int = 1,
        skip_question_ids=None,
        record_writer=None,
    ) -> Dict[int, List[str]]:
        """Generate responses, skipping QUESTION_IDs already present, and append each completed item to output_file."""

        if skip_question_ids is None:
            skip_question_ids = set()
        else:
            skip_question_ids = set(skip_question_ids)

        write_lock = threading.Lock()

        def _prompt_skip(question_id, error: Exception) -> bool:
            user_input = input(
                f"Generation failed for QUESTION_ID {question_id} ({error}). Skip and continue? [y/N]: "
            ).strip().lower()
            return user_input.startswith('y')

        def generate_conversation_responses(conversation):
            responses = []
            total_tokens = 0
            for _ in range(attempts):
                result = model_provider.generate(conversation.conversation, return_usage=True)
                if isinstance(result, tuple):
                    response, tokens = result
                    total_tokens += tokens
                else:
                    response = result
                responses.append(response)
            return conversation, responses, total_tokens

        conversations_to_run = [
            conversation
            for conversation in self.conversations
            if conversation.question_id not in skip_question_ids
        ]

        if not conversations_to_run:
            return self.responses

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_conv = submit_with_mapping(executor, conversations_to_run, generate_conversation_responses)

            for future in tqdm(as_completed(future_to_conv), total=len(future_to_conv), desc="Generating responses"):
                question_id = getattr(future_to_conv.get(future), "question_id", None)
                try:
                    conversation, responses, token_count = future.result()
                    question_id = conversation.question_id
                except Exception as e:
                    if _prompt_skip(question_id, e):
                        print(f"Skipping QUESTION_ID {question_id} and continuing.")
                        continue
                    print(f"Aborting at QUESTION_ID {question_id}.")
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

                self.responses[question_id] = responses
                self.token_counts[question_id] = token_count
                with write_lock:
                    if record_writer is not None:
                        record_writer(output_file, conversation, responses, token_count)
                    else:
                        self.append_response(output_file, question_id, responses, token_count)

        return self.responses

    def get_conversations(self) -> List[Conversation]:
        """Returns the list of Conversation objects."""
        return self.conversations
    
    def get_responses(self) -> Dict[int, List[str]]:
        """Returns the dictionary of responses."""
        return self.responses

    def get_token_counts(self) -> Dict[int, int]:
        """Returns the dictionary of token counts."""
        return self.token_counts
