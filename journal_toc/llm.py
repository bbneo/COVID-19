"""Talk to a local LLM over the Ollama or OpenAI-compatible HTTP API."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from journal_toc.errors import LlmError


@dataclass
class LlmClient:
    """HTTP client for one local chat model."""

    host: str
    model: str
    backend: str
    timeout: float = 300.0
    num_ctx: int = 8192

    def complete(self, system: str, user: str) -> str:
        if self.backend == "ollama":
            return self._ollama(system, user)
        if self.backend == "openai":
            return self._openai(system, user)
        raise LlmError(f"Unknown backend {self.backend!r}. Use ollama or openai.")

    def _ollama(self, system: str, user: str) -> str:
        url = _chat_url(self.host, "ollama")
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
        }
        body = _post_json(url, payload, self.timeout)
        try:
            return str(body["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise LlmError(
                "Ollama returned a response without message content. "
                f"Body keys: {list(body) if isinstance(body, dict) else type(body).__name__}"
            ) from exc

    def _openai(self, system: str, user: str) -> str:
        url = _chat_url(self.host, "openai")
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        body = _post_json(url, payload, self.timeout)
        try:
            return str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError(
                "The OpenAI-compatible server returned a response without choices. "
                f"Body keys: {list(body) if isinstance(body, dict) else type(body).__name__}"
            ) from exc


def _chat_url(host: str, backend: str) -> str:
    base = host.rstrip("/")
    if backend == "ollama":
        if base.endswith("/api/chat"):
            return base
        return base + "/api/chat"
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("LOCAL_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise LlmError(
            f"Local model request to {url} failed with HTTP {exc.code}. {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise LlmError(
            f"Could not reach a local model at {url}. "
            "Start Ollama (ollama serve) and pull a model (ollama pull llama3.2), "
            "or pass --backend openai and --host for another local server. "
            f"Details: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise LlmError(
            f"The local model at {url} did not respond within {timeout:.0f} seconds. "
            "Raise --timeout or use a smaller model."
        ) from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LlmError(f"Local model endpoint {url} did not return JSON.") from exc
    if not isinstance(parsed, dict):
        raise LlmError(f"Local model endpoint {url} returned {type(parsed).__name__}, not an object.")
    return parsed


def parse_json_content(content: str) -> dict:
    """Parse model output, including JSON wrapped in a markdown fence."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("Top-level JSON value must be an object", text, 0)
    return parsed
