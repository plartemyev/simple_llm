"""Thin OpenAI-compatible client for llama-server (chat completions API).

The server exposes the standard ``/v1/chat/completions`` endpoint (streaming
and non-streaming), so any OpenAI SDK pointed at ``base_url`` works. This
wrapper exists to codify the two call profiles used by latency-sensitive
text-transformation workloads:

* **phrase** — a complete phrase is transformed in one non-streaming call.
  Optimized for throughput and simplicity; latency is dominated by the full
  generation.
* **splits** — a running dictation stream is processed as a sequence of
  small fragments, one call per fragment. Optimized for time-to-first-token
  (streaming on) and short outputs.

See benchmarks/config.example.toml for the parameter sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import openai

DEFAULT_TEMPERATURE = 0.1
DEFAULT_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class CallParams:
    """Generation parameters for a single call."""

    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = 256
    seed: int | None = None


def build_messages(prompt: str, system: str | None = None) -> list[dict[str, str]]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


class LLMClient:
    """Small sync client over the OpenAI chat completions schema."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "not-needed",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = 1,
    ):
        self.model = model
        self._client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_s,
            max_retries=max_retries,
        )

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        params: CallParams | None = None,
    ) -> str:
        """Non-streaming completion (the *phrase* profile)."""
        p = params or CallParams()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=build_messages(prompt, system),
            temperature=p.temperature,
            max_tokens=p.max_tokens,
            seed=p.seed,
        )
        return resp.choices[0].message.content or ""

    def complete_messages(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = 256,
        seed: int | None = None,
    ) -> str:
        """Non-streaming completion over an explicit message list."""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        return resp.choices[0].message.content or ""

    def complete_stream(
        self,
        prompt: str,
        system: str | None = None,
        params: CallParams | None = None,
    ) -> Iterator[str]:
        """Streaming completion (the *splits* profile); yields text deltas."""
        p = params or CallParams()
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=build_messages(prompt, system),
            temperature=p.temperature,
            max_tokens=p.max_tokens,
            seed=p.seed,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
