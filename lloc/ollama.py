"""Ollama HTTP client (urllib only). Structured output via `format`. SPEC.md §6.

Transports are injectable for tests:
  transport(url, payload, timeout) -> dict
  stream_transport(url, payload, timeout) -> iterator of dicts (NDJSON lines)

Notes:
- Ollama does not always enforce the schema (ollama/ollama#15540), so the
  caller must validate the parsed response itself.
- The HTTP API does not auto-download models (unlike `ollama run`): a chat
  request for an absent model returns 404 "model '<name>' not found". The
  client detects this, pulls the model via /api/pull (streaming download
  progress to the console) and retries the request once.
"""
from __future__ import annotations

import json
import re
import sys
import threading
import urllib.error
import urllib.request

from .config import OllamaConfig

# One download at a time even with workers > 1; pulling an already-present
# model is a fast no-op, so late waiters pass straight through.
_pull_lock = threading.Lock()


class OllamaError(Exception):
    """Transport/protocol-level failure (unreachable, HTTP error, model missing)."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class OllamaBadResponse(OllamaError):
    """The model answered, but the content is empty or not valid JSON.
    Retryable: the caller should re-ask rather than abort."""

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw


def _request(url: str, payload: dict, timeout: int) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def _http_error_to_ollama(url: str, e: urllib.error.HTTPError) -> OllamaError:
    body = e.read().decode("utf-8", "replace")
    try:
        msg = json.loads(body).get("error", body)
    except json.JSONDecodeError:
        msg = body[:500]
    return OllamaError(f"Ollama HTTP {e.code}: {msg}", status=e.code)


def _http_transport(url: str, payload: dict, timeout: int) -> dict:
    try:
        with urllib.request.urlopen(_request(url, payload, timeout), timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise _http_error_to_ollama(url, e) from e
    except urllib.error.URLError as e:
        raise OllamaError(f"cannot reach Ollama at {url}: {e.reason}") from e


def _http_stream_transport(url: str, payload: dict, timeout: int):
    try:
        with urllib.request.urlopen(_request(url, payload, timeout), timeout=timeout) as resp:
            for line in resp:
                line = line.strip()
                if line:
                    yield json.loads(line.decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise _http_error_to_ollama(url, e) from e
    except urllib.error.URLError as e:
        raise OllamaError(f"cannot reach Ollama at {url}: {e.reason}") from e


_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```\s*$")


def extract_json(raw: str) -> str:
    """Tolerant extraction: some models (live case: gpt-oss:120b as a judge)
    wrap the JSON into markdown fences despite the schema. Strip ```json
    fences; failing that, take the first {...} block."""
    text = raw.strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text).strip()
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
    return text


def string_schema(keys: list[str]) -> dict:
    """JSON schema: object with a required string property per key."""
    return {
        "type": "object",
        "properties": {k: {"type": "string"} for k in keys},
        "required": list(keys),
        "additionalProperties": False,
    }


class OllamaClient:
    def __init__(self, cfg: OllamaConfig, transport=None, stream_transport=None):
        self.cfg = cfg
        self.transport = transport or _http_transport
        self.stream_transport = stream_transport or _http_stream_transport
        self._send_think = True  # disabled if the model rejects the `think` param

    # --- chat ------------------------------------------------------------

    def chat(self, messages: list[dict], schema: dict) -> tuple[dict, str]:
        """Send a chat request with structured output. Returns (parsed, raw_content).

        If the model is not present locally, pulls it and retries once."""
        try:
            return self._chat_once(messages, schema)
        except OllamaError as e:
            if not self._is_model_missing(e):
                raise
            self.pull_model()
            return self._chat_once(messages, schema)

    def _chat_once(self, messages: list[dict], schema: dict) -> tuple[dict, str]:
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "format": schema,
            "options": {"temperature": 0.1, "num_ctx": self.cfg.num_ctx},
        }
        if self._send_think:
            # Thinking models (qwen3.5, deepseek-r1, ...) may put everything into
            # `message.thinking` and leave `content` empty; disable thinking.
            payload["think"] = False
        try:
            resp = self.transport(f"{self.cfg.url}/api/chat", payload, self.cfg.timeout_sec)
        except OllamaError as e:
            if self._send_think and self._is_think_unsupported(e):
                self._send_think = False
                return self._chat_once(messages, schema)
            raise
        if "error" in resp:
            err = str(resp["error"])
            if self._send_think and "think" in err.lower():
                self._send_think = False
                return self._chat_once(messages, schema)
            raise OllamaError(f"Ollama error: {err}")

        message = resp.get("message", {})
        raw = message.get("content", "")
        if not raw.strip():
            hint = " (the reply went into the 'thinking' field)" if message.get("thinking") else ""
            raise OllamaBadResponse(f"model returned empty content{hint}", raw=raw)
        try:
            parsed = json.loads(extract_json(raw))
        except json.JSONDecodeError as e:
            raise OllamaBadResponse(f"model returned invalid JSON: {raw[:300]!r}", raw=raw) from e
        if not isinstance(parsed, dict):
            raise OllamaBadResponse(f"model returned non-object JSON: {raw[:300]!r}", raw=raw)
        return parsed, raw

    @staticmethod
    def _is_think_unsupported(e: OllamaError) -> bool:
        msg = str(e).lower()
        return "think" in msg and ("support" in msg or "unknown" in msg or "invalid" in msg)

    def _is_model_missing(self, e: OllamaError) -> bool:
        msg = str(e)
        return "not found" in msg and self.cfg.model in msg

    # --- pull ------------------------------------------------------------

    def pull_model(self) -> None:
        """Download the model via /api/pull, streaming progress to the console."""
        model = self.cfg.model
        with _pull_lock:
            print(f"model '{model}' not found locally, pulling "
                  f"(one-time download, may take a while)...", flush=True)
            try:
                self._pull_stream(model)
            except OllamaError as e:
                raise OllamaError(
                    f"failed to pull model '{model}': {e}. "
                    f"Check the model tag on https://ollama.com/library and your network, "
                    f"or run manually: ollama pull {model}"
                ) from e
            print(f"model '{model}' is ready", flush=True)

    def _pull_stream(self, model: str) -> None:
        last_status = ""
        progress_shown = False
        chunks = self.stream_transport(
            f"{self.cfg.url}/api/pull", {"model": model, "stream": True},
            self.cfg.timeout_sec,
        )
        for chunk in chunks:
            if "error" in chunk:
                raise OllamaError(str(chunk["error"]))
            status = chunk.get("status", "")
            total, completed = chunk.get("total"), chunk.get("completed")
            if total and completed is not None:
                pct = completed * 100 // total
                mb, mb_total = completed // 2**20, total // 2**20
                sys.stdout.write(f"\r  {status}: {pct}% ({mb}/{mb_total} MB)   ")
                sys.stdout.flush()
                progress_shown = True
            elif status and status != last_status:
                if progress_shown:
                    sys.stdout.write("\n")
                    progress_shown = False
                print(f"  {status}", flush=True)
            last_status = status
        if progress_shown:
            sys.stdout.write("\n")
            sys.stdout.flush()
