"""
Unified LLM client: supports Anthropic, OpenAI, local models, and CellType models.

Provides a consistent interface regardless of backend. CellType's own models
(GlueLM, C2S, etc.) are imported directly as Python modules when available,
falling back to API calls if served remotely.
"""

from dataclasses import dataclass, field
from typing import Optional, Generator
import logging
import os
import time

logger = logging.getLogger("ct.llm")


@dataclass
class LLMResponse:
    """Standardized response from any LLM backend."""
    content: str
    model: str
    usage: dict = None
    raw: object = None
    content_blocks: list = None  # Raw content blocks from API (for tool use)


# Pricing per million tokens (USD) — updated Feb 2026
MODEL_PRICING = {
    # Anthropic
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


@dataclass
class UsageTracker:
    """Tracks cumulative token usage and cost across LLM calls."""
    calls: list = field(default_factory=list)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.get("input", 0) for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.get("output", 0) for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost(self) -> float:
        return sum(c.get("cost", 0.0) for c in self.calls)

    def record(self, model: str, usage: dict):
        """Record a single LLM call's usage."""
        if not usage:
            return
        cost = self._estimate_cost(model, usage)
        self.calls.append({
            "model": model,
            "input": usage.get("input", 0),
            "output": usage.get("output", 0),
            "cost": cost,
        })

    def _estimate_cost(self, model: str, usage: dict) -> float:
        pricing = MODEL_PRICING.get(model)
        if not pricing:
            return 0.0
        input_cost = (usage.get("input", 0) / 1_000_000) * pricing["input"]
        output_cost = (usage.get("output", 0) / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def summary(self) -> str:
        """Human-readable usage summary."""
        if not self.calls:
            return "No LLM calls made."
        models_used = set(c["model"] for c in self.calls)
        return (
            f"{len(self.calls)} LLM calls | "
            f"{self.total_input_tokens:,} in + {self.total_output_tokens:,} out tokens | "
            f"${self.total_cost:.2f} | "
            f"models: {', '.join(models_used)}"
        )

    def reset(self):
        self.calls.clear()


class LLMClient:
    """Unified LLM client supporting multiple providers."""

    # Default models per provider
    DEFAULT_MODELS = {
        "anthropic": "claude-opus-4-6",
        "openai": "gpt-4o",
        "local": None,  # User must specify
        "gluelm": None,  # CellType's own model
    }

    def __init__(self, provider: str = "anthropic", model: str = None,
                 api_key: str = None):
        self.provider = provider
        self.model = model or self.DEFAULT_MODELS.get(provider)
        self.api_key = api_key
        self._client = None
        self.usage = UsageTracker()

    def _get_client(self):
        """Lazily initialize the appropriate client."""
        if self._client is not None:
            return self._client

        if self.provider == "anthropic":
            import anthropic
            if os.environ.get("ANTHROPIC_FOUNDRY_API_KEY") or os.environ.get("ANTHROPIC_FOUNDRY_RESOURCE"):
                self._client = anthropic.AnthropicFoundry()
            else:
                self._client = anthropic.Anthropic(
                    api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY")
                )

        elif self.provider == "openai":
            import openai
            self._client = openai.OpenAI(
                api_key=self.api_key or os.environ.get("OPENAI_API_KEY")
            )

        elif self.provider == "local":
            # Local model via vLLM, ollama, or direct transformers
            self._client = self._init_local()

        elif self.provider == "gluelm":
            # CellType's own model — direct Python import
            self._client = self._init_gluelm()

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        return self._client

    def chat(self, system: str, messages: list[dict], temperature: float = 0.1,
             max_tokens: int = 4096, tools: list[dict] | None = None) -> LLMResponse:
        """Send a chat completion request.

        Args:
            tools: Optional list of tool definitions (Anthropic tool_use format).
                   When provided, the response may contain tool_use content blocks
                   accessible via ``response.content_blocks``.
        """
        client = self._get_client()

        if self.provider == "anthropic":
            resp = self._chat_anthropic(client, system, messages, temperature, max_tokens, tools=tools)
        elif self.provider == "openai":
            resp = self._chat_openai(client, system, messages, temperature, max_tokens)
        elif self.provider == "local":
            resp = self._chat_local(client, system, messages, temperature, max_tokens)
        elif self.provider == "gluelm":
            resp = self._chat_gluelm(client, system, messages, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        # Track usage
        if resp.usage:
            self.usage.record(resp.model, resp.usage)

        return resp

    def stream(self, system: str, messages: list[dict], temperature: float = 0.1,
               max_tokens: int = 4096) -> Generator[str, None, LLMResponse]:
        """Stream a chat completion, yielding text chunks.

        Yields individual text deltas. After the generator is exhausted,
        send() returns the final LLMResponse with full content and usage.

        Usage:
            gen = llm.stream(system, messages)
            chunks = []
            for chunk in gen:
                print(chunk, end="", flush=True)
                chunks.append(chunk)
            # Full response available after iteration
        """
        client = self._get_client()

        if self.provider == "anthropic":
            yield from self._stream_anthropic(client, system, messages, temperature, max_tokens)
        elif self.provider == "openai":
            yield from self._stream_openai(client, system, messages, temperature, max_tokens)
        else:
            # Fallback: non-streaming providers just yield the full response
            resp = self.chat(system, messages, temperature, max_tokens)
            yield resp.content

    def _chat_anthropic(self, client, system, messages, temperature, max_tokens, tools=None):
        return self._retry(
            lambda: self._call_anthropic(client, system, messages, temperature, max_tokens, tools=tools)
        )

    def _call_anthropic(self, client, system, messages, temperature, max_tokens, tools=None):
        kwargs = dict(
            model=self.model,
            system=system,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
        response = client.messages.create(**kwargs)
        # Guard against empty content array (e.g., content filtering)
        if not response.content:
            content_text = ""
        else:
            # Extract text parts only (skip tool_use blocks)
            text_parts = [b.text for b in response.content if hasattr(b, "text")]
            content_text = "\n".join(text_parts) if text_parts else ""
        return LLMResponse(
            content=content_text,
            model=self.model,
            usage={"input": response.usage.input_tokens, "output": response.usage.output_tokens},
            raw=response,
            content_blocks=list(response.content) if response.content else [],
        )

    def _retry(self, fn, max_retries: int = 3, base_delay: float = 2.0):
        """Retry a function with exponential backoff on transient errors."""
        for attempt in range(1, max_retries + 1):
            try:
                return fn()
            except Exception as e:
                err_str = str(e).lower()
                is_transient = any(w in err_str for w in (
                    "rate_limit", "rate limit", "429", "overloaded",
                    "529", "500", "502", "503", "connection", "timeout",
                ))
                if is_transient and attempt < max_retries:
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning("LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                                   attempt, max_retries, e, delay)
                    time.sleep(delay)
                else:
                    raise

    def _stream_anthropic(self, client, system, messages, temperature, max_tokens):
        """Stream from Anthropic API, yielding text deltas."""
        with client.messages.stream(
            model=self.model,
            system=system,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ) as stream:
            try:
                for text in stream.text_stream:
                    yield text
            finally:
                # Record usage even if stream is interrupted (Ctrl+C)
                try:
                    response = stream.get_final_message()
                    usage = {"input": response.usage.input_tokens, "output": response.usage.output_tokens}
                    self.usage.record(self.model, usage)
                except Exception:
                    logger.debug("Could not record usage after stream interrupt")

    def _stream_openai(self, client, system, messages, temperature, max_tokens):
        """Stream from OpenAI API, yielding text deltas."""
        all_messages = [{"role": "system", "content": system}] + messages
        stream = client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        usage = None
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
            if chunk.usage:
                usage = {"input": chunk.usage.prompt_tokens, "output": chunk.usage.completion_tokens}

        if usage:
            self.usage.record(self.model, usage)

    def _chat_openai(self, client, system, messages, temperature, max_tokens):
        return self._retry(
            lambda: self._call_openai(client, system, messages, temperature, max_tokens)
        )

    def _call_openai(self, client, system, messages, temperature, max_tokens):
        all_messages = [{"role": "system", "content": system}] + messages
        response = client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            usage={"input": response.usage.prompt_tokens, "output": response.usage.completion_tokens},
            raw=response,
        )

    def _init_local(self):
        """Initialize local model (vLLM or transformers)."""
        # Try vLLM first (fastest for local inference)
        try:
            from vllm import LLM
            return LLM(model=self.model)
        except ImportError:
            pass

        # Fall back to transformers
        try:
            from transformers import pipeline
            return pipeline("text-generation", model=self.model, device_map="auto")
        except ImportError:
            raise ImportError("Install vllm or transformers for local model support")

    def _chat_local(self, client, system, messages, temperature, max_tokens):
        """Chat with local model."""
        # Format for local model
        prompt = f"System: {system}\n\n"
        for msg in messages:
            role = msg["role"].capitalize()
            prompt += f"{role}: {msg['content']}\n\n"
        prompt += "Assistant: "

        if hasattr(client, 'generate'):
            # vLLM
            from vllm import SamplingParams
            params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
            outputs = client.generate([prompt], params)
            text = outputs[0].outputs[0].text
        else:
            # transformers pipeline
            outputs = client(prompt, max_new_tokens=max_tokens, temperature=temperature)
            text = outputs[0]["generated_text"][len(prompt):]

        return LLMResponse(content=text, model=self.model or "local")

    def _init_gluelm(self):
        """Initialize CellType's GlueLM model."""
        try:
            from gluelm import GlueLMModel
            return GlueLMModel.from_pretrained(self.model)
        except ImportError:
            raise ImportError(
                "GlueLM not installed. Install from CellType/GlueLM or "
                "set llm.provider to 'anthropic' for cloud inference."
            )

    def _chat_gluelm(self, client, system, messages, temperature, max_tokens):
        """Chat with GlueLM — specialized for degradation queries."""
        # GlueLM is a domain-specific model, not a general chat model
        # Route degradation-specific queries to it, general queries to fallback
        query = messages[-1]["content"] if messages else ""
        result = client.predict(query)
        return LLMResponse(
            content=str(result),
            model="gluelm",
        )
