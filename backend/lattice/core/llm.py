"""Provider-agnostic LLM client abstraction.

Everything that calls a model (extraction, RAG agent, router, vision, synthesis)
goes through :class:`LLMClient`. The production implementation uses LiteLLM so any
provider works and retries/cost-tracking are uniform; tests inject a fake. JSON
mode returns parsed output plus token usage for cost accounting.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

#: A fenced code block anywhere in the text (weak models wrap JSON in prose).
_FENCED_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def json(self) -> Any:
        """Parse the response text as JSON, salvaging common weak-model wrappings.

        Tried in order: the raw text; a ```json fenced block anywhere (models often
        add prose around it, e.g. "Here is the JSON: ..."); the outermost
        ``{...}`` span. Raises :class:`json.JSONDecodeError` if none parses, so the
        caller's repair loop gets a real signal instead of a silent None.
        """
        text = self.text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        fenced = _FENCED_BLOCK.search(text)
        if fenced:
            try:
                return json.loads(fenced.group(1).strip())
            except json.JSONDecodeError:
                pass
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            return json.loads(text[start : end + 1])
        raise json.JSONDecodeError("no JSON object found in response", text, 0)


class LLMClient(Protocol):
    async def complete(
        self,
        model: str,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse: ...


class LiteLLMClient:
    """LiteLLM-backed client. Provider keys come from the environment."""

    def __init__(self, timeout: float = 120.0, num_retries: int = 3):
        self._timeout = timeout
        self._retries = num_retries

    async def complete(  # pragma: no cover - exercised via integration, not unit
        self,
        model: str,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        import litellm

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": self._timeout,
            "num_retries": self._retries,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = tools

        resp = await litellm.acompletion(**kwargs)
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=choice.message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            model=model,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
        )
