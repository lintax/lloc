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
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request

from .config import DEFAULT_NUM_CTX, OllamaConfig, _is_cloud_tag

# One download at a time even with workers > 1; pulling an already-present
# model is a fast no-op, so late waiters pass straight through.
_pull_lock = threading.Lock()


class OllamaError(Exception):
    """Transport/protocol-level failure (unreachable, HTTP error, model missing)."""

    def __init__(self, message: str, status: int | None = None,
                 retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class AdaptiveLimiter:
    """Global adaptive concurrency limiter for ALL LLM requests of the
    process (SPEC.md §6): HTTP 429 means "slow down", not "fail the
    language". On a throttle the limit shrinks (x0.7, floor 2); after
    RECOVERY_STREAK consecutive successes it grows by one back towards the
    ceiling. Waiting threads are governed by a condition variable, so
    shrinking takes effect immediately and growth is picked up promptly."""

    CEILING = 64
    MIN = 2
    RECOVERY_STREAK = 20

    def __init__(self):
        self._cond = threading.Condition()
        self._limit = self.CEILING
        self._active = 0
        self._streak = 0

    @property
    def limit(self) -> int:
        return self._limit

    def __enter__(self):
        with self._cond:
            while self._active >= self._limit:
                self._cond.wait(timeout=1.0)  # re-check: the limit may grow
            self._active += 1
        return self

    def __exit__(self, *_exc):
        with self._cond:
            self._active -= 1
            self._cond.notify()

    def on_throttle(self) -> None:
        with self._cond:
            new = max(self.MIN, int(self._limit * 0.7))
            if new != self._limit:
                self._limit = new
                print(f"server throttling (429): concurrency limit -> {new}",
                      flush=True)
            self._streak = 0

    def on_success(self) -> None:
        with self._cond:
            self._streak += 1
            if self._streak >= self.RECOVERY_STREAK and self._limit < self.CEILING:
                self._limit += 1
                self._streak = 0
                self._cond.notify_all()


_limiter = AdaptiveLimiter()

BACKOFF_BASE_SEC = 1.0
BACKOFF_CAP_SEC = 30.0
BACKOFF_TRIES = 6


def _is_throttle(e: OllamaError) -> bool:
    return e.status == 429 or "too many concurrent" in str(e).lower()


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
    retry_after = None
    try:
        header = e.headers.get("Retry-After") if e.headers else None
        if header:
            retry_after = float(header)
    except (TypeError, ValueError):
        pass
    return OllamaError(f"Ollama HTTP {e.code}: {msg}", status=e.code,
                       retry_after=retry_after)


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


# Models that do not honour `think: false`: they reason anyway and dump the
# reasoning into `content` ahead of the JSON (live case: glm-5.3-flash, 5-8x
# the output tokens). A reasoning *level* is respected, so ask for the lowest.
_THINK_LEVEL_ONLY = ("glm-",)


def _think_value(model: str) -> bool | str:
    return "low" if model.startswith(_THINK_LEVEL_ONLY) else False


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
        # num_ctx: explicit config value always wins; otherwise local models
        # get the safe default (Ollama's own 4096 truncates few-shot prompts),
        # while CLOUD models get no num_ctx at all — the server sizes the
        # context itself (256K-1M class), and a hard 8192 would clip large
        # batches for no reason.
        options: dict = {"temperature": 0.1}
        if self.cfg.num_ctx is not None:
            options["num_ctx"] = self.cfg.num_ctx
        elif not _is_cloud_tag(self.cfg.model):
            options["num_ctx"] = DEFAULT_NUM_CTX
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "format": schema,
            "options": options,
        }
        if self._send_think:
            # Thinking models (qwen3.5, deepseek-r1, ...) may put everything into
            # `message.thinking` and leave `content` empty; disable thinking.
            payload["think"] = _think_value(self.cfg.model)
        try:
            resp = self._transport_throttled(f"{self.cfg.url}/api/chat", payload)
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

    def _transport_throttled(self, url: str, payload: dict) -> dict:
        """One logical request under the global adaptive limiter. HTTP 429 is
        retried with exponential backoff + jitter (honouring Retry-After),
        NEVER surfaced as fatal until the retry budget is exhausted."""
        delay = BACKOFF_BASE_SEC
        for attempt in range(BACKOFF_TRIES):
            with _limiter:
                try:
                    resp = self.transport(url, payload, self.cfg.timeout_sec)
                    _limiter.on_success()
                    return resp
                except OllamaError as e:
                    if not _is_throttle(e) or attempt == BACKOFF_TRIES - 1:
                        raise
                    _limiter.on_throttle()
                    wait = e.retry_after if e.retry_after \
                        else delay * (1.0 + random.random() * 0.25)
            time.sleep(wait)  # sleep OUTSIDE the limiter slot
            delay = min(delay * 2, BACKOFF_CAP_SEC)
        raise AssertionError("unreachable")  # pragma: no cover

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
