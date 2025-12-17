from openai import OpenAI
from typing import Any, Dict, Optional, Tuple, Union
from src.models.base import ModelProvider

class OpenAIModel(ModelProvider):
    """OpenAI model provider that uses GPT-4 for evaluation."""

    @staticmethod
    def build_api_params(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
        cfg = model_cfg or {}
        api_params = dict(cfg.get('api') or {})
        if 'extra_body' in cfg and 'extra_body' not in api_params:
            api_params['extra_body'] = cfg.get('extra_body')
        return api_params

    @classmethod
    def from_model_config(
        cls,
        model_id: str,
        model_cfg: Dict[str, Any],
        **overrides: Any
    ):
        api_params = cls.build_api_params(model_cfg)
        api_params.update(overrides)
        return cls(
            model=model_id,
            temp=(model_cfg or {}).get('temperature', 0.0),
            api_key=(model_cfg or {}).get('api_key'),
            base_url=(model_cfg or {}).get('base_url'),
            **api_params,
        )

    def __init__(
        self,
        model: str,
        temp: float,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **default_params: Any
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.model = model
        self.temp = float(temp)
        self.default_params: Dict[str, Any] = dict(default_params)

    def generate(self, prompt: Any, return_usage: bool = False) -> Union[str, Tuple[str, int]]:
        """
        Args:
            prompt: The input prompt
            return_usage: If True, return (content, token_count) tuple
        
        Returns:
            str or (str, int): Response content, or tuple of (content, token_count)
        """
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) and 'role' in item and item['role'] in ['user', 'assistant'] for item in prompt):
            pass 
        else:
            raise ValueError("Prompt must be a string or a list of dictionaries with 'role' keys as 'user' or 'assistant'.")

        extra = dict(self.default_params)
        extra.pop('model', None)
        extra.pop('messages', None)
        extra.pop('temperature', None)

        stream = bool(extra.pop('stream', True))

        if stream:
            stream_resp = self.client.chat.completions.create(
                stream=True,
                model=self.model,
                messages=prompt,
                temperature=self.temp,
                stream_options={"include_usage": True} if return_usage else None,
                **extra
            )
            chunks = []
            token_count = 0
            for chunk in stream_resp:
                if return_usage and getattr(chunk, 'usage', None):
                    usage = chunk.usage
                    token_count = (getattr(usage, 'prompt_tokens', 0) or 0) + (getattr(usage, 'completion_tokens', 0) or 0)
                if not getattr(chunk, 'choices', None):
                    continue
                delta = getattr(chunk.choices[0], 'delta', None)
                if delta is None:
                    continue
                content = getattr(delta, 'content', None)
                if content:
                    chunks.append(content)
            content = ''.join(chunks)
            return (content, token_count) if return_usage else content

        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            temperature=self.temp,
            **extra
        )
        content = response.choices[0].message.content
        if return_usage:
            usage = getattr(response, 'usage', None)
            token_count = 0
            if usage:
                token_count = (getattr(usage, 'prompt_tokens', 0) or 0) + (getattr(usage, 'completion_tokens', 0) or 0)
            return (content, token_count)
        return content