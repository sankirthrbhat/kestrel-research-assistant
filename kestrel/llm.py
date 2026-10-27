"""LLM access layer.

Supports any OpenAI-compatible ``/chat/completions`` endpoint, including
OpenAI, Groq, Gemini (via its OpenAI-compatible surface), DeepSeek, and local
servers.

Provider auto-detection
-----------------------
The client reads credentials from the environment only — never from source.
Whichever key is present first in the priority list below wins:

1. ``KESTREL_LLM_API_KEY``  — explicit override; also respects
   ``KESTREL_LLM_BASE_URL`` and ``KESTREL_LLM_MODEL``.
2. ``GROQ_API_KEY``          — Groq endpoint, llama-3.3-70b-versatile.
3. ``GEMINI_API_KEY``        — Gemini OpenAI-compat endpoint, gemini-2.0-flash.
4. ``OPENAI_API_KEY``        — OpenAI, gpt-4o-mini.
5. ``DEEPSEEK_API_KEY``      — DeepSeek, deepseek-chat.

Retry / backoff
---------------
HTTP 429 (rate-limit) and 503 (service unavailable) are retried up to
``MAX_RETRIES`` times with exponential backoff (``RETRY_BASE_DELAY`` seconds,
doubling each attempt).  Any other HTTP error is raised immediately so the
calling agent node can fall back to its deterministic offline path.

Offline fallback
----------------
When ``LLMConfig.configured`` is False no network call is made.  Every agent
node checks this flag and routes to its heuristic fallback.  Offline answers
are clearly marked in the draft text (the word "offline mode" appears in every
fallback response).

Token usage
-----------
``LLMClient.last_usage`` holds the raw ``usage`` dict from the last successful
API response (keys: ``prompt_tokens``, ``completion_tokens``, ``total_tokens``
for most providers; Gemini may use ``input_tokens`` / ``output_tokens``).  It
is ``None`` before the first call or after a fallback.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_TOKENS = 1024

# Provider-specific defaults (used when auto-detected from API key)
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
_GEMINI_DEFAULT_MODEL = "gemini-3.5-flash"

_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
_DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"

# Retry configuration: at most MAX_RETRIES additional attempts after the first.
# Retried status codes: 429 (rate limit), 503 (service unavailable).
MAX_RETRIES: int = 4
RETRY_BASE_DELAY: float = 8.0  # seconds; doubles each attempt
_RETRYABLE_CODES: frozenset[int] = frozenset({429, 503})


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class LLMNotConfigured(RuntimeError):
    """Raised when an LLM call is attempted without credentials."""


class LLMQuotaExhausted(RuntimeError):
    """Raised when the LLM provider rate limit (TPD/TPM) is exhausted."""


@dataclass(frozen=True)
class LLMConfig:
    """Resolved LLM settings."""

    api_key: str | None
    base_url: str
    model: str
    timeout: float
    provider: str = "unknown"  # informational; used in tracing metadata

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Resolve configuration from environment variables.

        Priority order (first matching key wins):
          KESTREL_LLM_API_KEY > GROQ_API_KEY > GEMINI_API_KEY >
          OPENAI_API_KEY > DEEPSEEK_API_KEY
        """
        if os.environ.get("KESTREL_OFFLINE", "").lower() in ("1", "true"):
            return cls(api_key=None, base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL,
                       timeout=DEFAULT_TIMEOUT, provider="offline")
        # ---- Explicit KESTREL override -----------------------------------------
        kestrel_key = os.environ.get("KESTREL_LLM_API_KEY")
        if kestrel_key:
            return cls(
                api_key=kestrel_key,
                base_url=os.environ.get("KESTREL_LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
                model=os.environ.get("KESTREL_LLM_MODEL", DEFAULT_MODEL),
                timeout=float(os.environ.get("KESTREL_LLM_TIMEOUT", DEFAULT_TIMEOUT)),
                provider="custom",
            )

        # ---- Groq ---------------------------------------------------------------
        groq_key = os.environ.get("GROQ_API_KEY")
        if groq_key:
            return cls(
                api_key=groq_key,
                base_url=os.environ.get("KESTREL_LLM_BASE_URL", _GROQ_BASE_URL).rstrip("/"),
                model=os.environ.get("KESTREL_LLM_MODEL", _GROQ_DEFAULT_MODEL),
                timeout=float(os.environ.get("KESTREL_LLM_TIMEOUT", DEFAULT_TIMEOUT)),
                provider="groq",
            )

        # ---- Gemini (OpenAI-compatible surface) --------------------------------
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            return cls(
                api_key=gemini_key,
                base_url=os.environ.get("KESTREL_LLM_BASE_URL", _GEMINI_BASE_URL).rstrip("/"),
                model=os.environ.get("KESTREL_LLM_MODEL", _GEMINI_DEFAULT_MODEL),
                timeout=float(os.environ.get("KESTREL_LLM_TIMEOUT", DEFAULT_TIMEOUT)),
                provider="gemini",
            )

        # ---- OpenAI -------------------------------------------------------------
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            return cls(
                api_key=openai_key,
                base_url=os.environ.get("KESTREL_LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
                model=os.environ.get("KESTREL_LLM_MODEL", DEFAULT_MODEL),
                timeout=float(os.environ.get("KESTREL_LLM_TIMEOUT", DEFAULT_TIMEOUT)),
                provider="openai",
            )

        # ---- DeepSeek -----------------------------------------------------------
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        if deepseek_key:
            return cls(
                api_key=deepseek_key,
                base_url=os.environ.get("KESTREL_LLM_BASE_URL", _DEEPSEEK_BASE_URL).rstrip("/"),
                model=os.environ.get("KESTREL_LLM_MODEL", _DEEPSEEK_DEFAULT_MODEL),
                timeout=float(os.environ.get("KESTREL_LLM_TIMEOUT", DEFAULT_TIMEOUT)),
                provider="deepseek",
            )

        # ---- No key found -------------------------------------------------------
        return cls(
            api_key=None,
            base_url=os.environ.get("KESTREL_LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            model=os.environ.get("KESTREL_LLM_MODEL", DEFAULT_MODEL),
            timeout=float(os.environ.get("KESTREL_LLM_TIMEOUT", DEFAULT_TIMEOUT)),
            provider="none",
        )


# ---------------------------------------------------------------------------
# .env loader (optional dependency)
# ---------------------------------------------------------------------------


def _load_dotenv_if_available() -> None:
    """Load a local .env when python-dotenv is installed (optional)."""
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
    except ImportError:
        return
    from .config import ROOT

    load_dotenv(ROOT / ".env", override=False)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LLMClient:
    """Minimal OpenAI-compatible chat client with retry/backoff.

    Attributes
    ----------
    last_usage:
        Raw ``usage`` dict from the most recent successful API response, e.g.
        ``{"prompt_tokens": 120, "completion_tokens": 80, "total_tokens": 200}``.
        ``None`` before any call or when the offline path is used.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        _load_dotenv_if_available()
        self.config = config or LLMConfig.from_env()
        self.last_usage: dict[str, Any] | None = None

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Return the assistant message text for ``prompt``.

        Parameters
        ----------
        prompt:
            The user-turn content.
        system:
            Optional system prompt.
        temperature:
            Sampling temperature (0 = deterministic).
        max_tokens:
            Maximum tokens in the completion.  Defaults to ``DEFAULT_MAX_TOKENS``
            to keep agent outputs controlled.  Pass ``None`` to use the provider
            default.

        Raises
        ------
        LLMNotConfigured
            When no API key is available.
        RuntimeError
            When all retry attempts fail (HTTP errors, network errors).
        """
        if not self.config.configured:
            raise LLMNotConfigured(
                "No LLM API key found.  Copy .env.example to .env and set "
                "one of: KESTREL_LLM_API_KEY, GROQ_API_KEY, GEMINI_API_KEY, "
                "OPENAI_API_KEY, or DEEPSEEK_API_KEY."
            )

        self.last_usage = None
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        url = f"{self.config.base_url}/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        }

        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            request = urllib.request.Request(
                url, data=data, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                # Success — capture usage and return.
                try:
                    content = body["choices"][0]["message"]["content"]
                    if not isinstance(content, str) or not content.strip():
                        raise TypeError("Expected nonempty text completion")
                    self.last_usage = body.get("usage") or None
                    return content
                except (KeyError, IndexError, TypeError) as exc:
                    raise RuntimeError("Unexpected LLM response shape") from exc

            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code in _RETRYABLE_CODES:
                    if attempt < MAX_RETRIES:
                        delay = RETRY_BASE_DELAY * (2 ** attempt)
                        try:
                            import re
                            match = re.search(r'"retryDelay":\s*"([0-9.]+)s"', detail)
                            if match:
                                delay = min(60.0, max(0.0, float(match.group(1)))) + 0.5
                        except Exception:
                            pass
                        time.sleep(delay)
                        last_exc = exc
                        continue  # retry
                    else:
                        if exc.code == 429:
                            raise LLMQuotaExhausted(f"LLM quota exhausted ({exc.code}): {detail}") from exc
                
                # Non-retryable error or exhausted retries for 503.
                raise RuntimeError(
                    f"LLM request failed ({exc.code}): {detail}"
                ) from exc

            except (TimeoutError, urllib.error.URLError) as exc:
                raise RuntimeError("LLM network request failed") from exc

        # Should be unreachable, but satisfies type checkers.
        raise RuntimeError("LLM request failed after all retry attempts.") from last_exc


__all__ = [
    "LLMClient",
    "LLMConfig",
    "LLMNotConfigured",
    "MAX_RETRIES",
    "RETRY_BASE_DELAY",
    "DEFAULT_MAX_TOKENS",
]
