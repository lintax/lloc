"""Translation engine: delta -> batches -> Ollama -> validation -> store. SPEC.md §5-7."""
from __future__ import annotations

import concurrent.futures
import contextlib
import threading
import time
from dataclasses import dataclass, field

from .config import Project
from .hashes import text_hash
from .ollama import OllamaBadResponse, OllamaClient, OllamaError, string_schema
from .prompts import build_retry_prompt, build_translate_prompt
from .store import classify, load_locale, save_locale
from .validate import check_translation

_print_lock = threading.Lock()
_model_note_shown = False

HEARTBEAT_SEC = 30.0


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def _note_model_load(model: str) -> None:
    """Print a one-time hint before the first request (model load is slow)."""
    global _model_note_shown
    with _print_lock:
        if _model_note_shown:
            return
        _model_note_shown = True
        print(f"loading model '{model}' into memory, first response may take a while...",
              flush=True)


@contextlib.contextmanager
def _heartbeat(label: str, interval: float = HEARTBEAT_SEC):
    """Periodically report that a long request is still in flight."""
    stop = threading.Event()

    def beat():
        waited = 0
        while not stop.wait(interval):
            waited += interval
            _log(f"{label}: still working, {int(waited)}s...")

    t = threading.Thread(target=beat, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=1)


@dataclass
class LangResult:
    lang: str
    translated: int = 0
    failed: dict[str, list[str]] = field(default_factory=dict)  # key -> errors
    orphans_removed: int = 0
    error: str | None = None  # fatal (e.g. Ollama unreachable)


def _batches(keys: list[str], size: int) -> list[list[str]]:
    return [keys[i:i + size] for i in range(0, len(keys), size)]


def translate_batch(
    project: Project,
    client: OllamaClient,
    lang: str,
    keys: list[str],
    locale: dict[str, dict],
    extra_note: str | None = None,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Translate one batch. Returns (accepted {key: text}, failed {key: errors}).

    Validation failures trigger a retry of the failed subset with the error text
    appended to the conversation, up to max_retries. `extra_note` is prepended
    to the examples block (used by --fix-terms to pin a terminology variant)."""
    entries = [project.strings[k] for k in keys]
    schema = string_schema(keys)
    prompt = build_translate_prompt(project, lang, entries, locale, extra_note)
    messages = [{"role": "user", "content": prompt}]

    accepted: dict[str, str] = {}
    pending = {e.key: e for e in entries}
    last_errors: dict[str, list[str]] = {}

    for attempt in range(project.ollama.max_retries + 1):
        try:
            parsed, raw = client.chat(messages, schema)
        except OllamaBadResponse as e:
            # empty/garbled reply is retryable, not fatal (SPEC.md §7)
            last_errors = {k: [str(e)] for k in pending}
            if attempt == project.ollama.max_retries:
                break
            retry_prompt = build_retry_prompt(
                project, list(pending.values()),
                [f"your previous reply was unusable: {e}"])
            messages = messages + [
                {"role": "assistant", "content": e.raw},
                {"role": "user", "content": retry_prompt},
            ]
            continue
        last_errors = {}
        for key, entry in list(pending.items()):
            value = parsed.get(key)
            errors = check_translation(entry.text, value, project.length_ratio) \
                if value is not None else ["missing from response"]
            if not errors:
                accepted[key] = value
                del pending[key]
            else:
                last_errors[key] = errors
        if not pending or attempt == project.ollama.max_retries:
            break
        flat = [f"{k}: {'; '.join(v)}" for k, v in last_errors.items()]
        retry_prompt = build_retry_prompt(project, list(pending.values()), flat)
        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": retry_prompt},
        ]
        schema = string_schema(list(pending))
    return accepted, last_errors


def translate_language(
    project: Project,
    lang: str,
    keys: list[str],
    transport=None,
) -> LangResult:
    """Translate `keys` for one language, saving the locale after every batch."""
    result = LangResult(lang=lang)
    client = OllamaClient(project.ollama, transport=transport)
    locale = load_locale(project.root, lang)
    order = list(project.strings)

    orphans = [k for k in locale if k not in project.strings]
    result.orphans_removed = len(orphans)

    tag = f"{project.module}:{lang}" if project.module else lang
    batches = _batches(keys, project.ollama.batch_size)
    for i, batch in enumerate(batches, 1):
        label = f"[{tag}] batch {i}/{len(batches)}"
        _log(f"{label}: translating {len(batch)} strings...")
        _note_model_load(project.ollama.model)
        started = time.monotonic()
        try:
            with _heartbeat(label):
                accepted, failed = translate_batch(project, client, lang, batch, locale)
        except OllamaError as e:
            result.error = str(e)
            _log(f"{label}: FATAL {e}")
            break
        elapsed = time.monotonic() - started
        for key, text in accepted.items():
            locale[key] = {"text": text, "src": text_hash(project.strings[key].text)}
        result.translated += len(accepted)
        result.failed.update(failed)
        save_locale(project.root, lang, locale, order)  # progress survives interruption
        _log(f"{label}: +{len(accepted)} in {elapsed:.1f}s"
             + (f", failed {len(failed)}" if failed else ""))

    if result.error is None and (orphans or not batches):
        save_locale(project.root, lang, locale, order)  # drop orphans even with empty delta
    return result


def plan(project: Project, langs: list[str], force: bool, limit: int | None) -> dict[str, list[str]]:
    """{lang: keys to translate} respecting --force and --limit."""
    out: dict[str, list[str]] = {}
    for lang in langs:
        st = classify(project, lang)
        keys = list(project.strings) if force else st.pending
        if limit is not None:
            keys = keys[:limit]
        out[lang] = keys
    return out


def run_translate_jobs(
    jobs: list[tuple[Project, str, list[str]]],
    workers: int,
    transport=None,
) -> list[tuple[Project, LangResult]]:
    """Run (project, lang, keys) jobs over ONE shared worker pool — in
    workspace mode (SPEC.md §14) the pool spans all modules of the run.
    Results come back in job order."""
    global _model_note_shown
    _model_note_shown = False  # fresh note per run
    results: list[tuple[Project, LangResult] | None] = [None] * len(jobs)
    if workers <= 1:
        for i, (project, lang, keys) in enumerate(jobs):
            results[i] = (project, translate_language(project, lang, keys, transport))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(translate_language, project, lang, keys, transport): i
                for i, (project, lang, keys) in enumerate(jobs)
            }
            for fut in concurrent.futures.as_completed(futures):
                i = futures[fut]
                results[i] = (jobs[i][0], fut.result())
    return results  # type: ignore[return-value]


def run_translate(
    project: Project,
    langs: list[str],
    force: bool = False,
    limit: int | None = None,
    transport=None,
) -> list[LangResult]:
    work = plan(project, langs, force, limit)
    jobs = [(project, lang, keys) for lang, keys in work.items()]
    return [r for _, r in run_translate_jobs(jobs, project.ollama.workers, transport)]
