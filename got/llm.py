"""LLM adapter: chat client that works against either LM Studio (local,
OpenAI-compatible) or the real OpenAI API, selected by provider name/env var.
"""
import os
import time

from openai import OpenAI

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_LMSTUDIO_BASE_URL = "http://localhost:1234/v1"


def _resolve_provider(provider):
    provider = provider or os.environ.get("LLM_PROVIDER")
    if provider:
        return provider.lower()
    # infer: if an OpenAI key is set and LM Studio wasn't explicitly requested, prefer OpenAI
    return "openai" if os.environ.get("OPENAI_API_KEY") else "lmstudio"


class LLM:
    """Provider-agnostic chat client.

    provider: "lmstudio" (default, local) or "openai" (hosted, needs an API key).
    Can also be selected via the LLM_PROVIDER env var.
    """

    def __init__(self, provider=None, base_url=None, api_key=None, model=None,
                 temperature=0.7, max_tokens=2048):
        self.provider = _resolve_provider(provider)

        if self.provider == "openai":
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "OpenAI provider selected but no API key found. "
                    "Pass api_key=... or set OPENAI_API_KEY."
                )
            self.client = OpenAI(base_url=base_url, api_key=key)
            self.model = model or os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)

        elif self.provider == "lmstudio":
            self.client = OpenAI(
                base_url=base_url or os.environ.get("LMSTUDIO_BASE_URL", DEFAULT_LMSTUDIO_BASE_URL),
                api_key=api_key or "lm-studio",  # LM Studio ignores the key but the SDK requires one
            )
            # Default to whichever model is currently loaded in LM Studio
            self.model = model or os.environ.get("LMSTUDIO_MODEL") or self.client.models.list().data[0].id

        else:
            raise ValueError(f"Unknown provider {self.provider!r}; use 'lmstudio' or 'openai'")

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def chat(self, prompt, system=None, temperature=None, retries=3):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        last_err = None
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature if temperature is None else temperature,
                    max_tokens=self.max_tokens,
                )
                self.calls += 1
                if resp.usage:
                    self.prompt_tokens += resp.usage.prompt_tokens or 0
                    self.completion_tokens += resp.usage.completion_tokens or 0
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2**attempt)
        raise RuntimeError(f"LLM call failed after {retries} retries: {last_err}")

    def stats(self):
        return {
            "provider": self.provider,
            "model": self.model,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
