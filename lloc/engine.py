"""Translation engine: delta -> batches -> Ollama -> validation -> store."""
from __future__ import annotations

import concurrent.futures
import contextlib
import threading
import time
from dataclasses import dataclass, field

from fnmatch import fnmatchcase

from .config import Project
from .fsutil import utc_now_iso
from .hashes import text_hash
from .ollama import OllamaBadResponse, OllamaClient, OllamaError, string_schema
from .prompts import build_retry_prompt, build_translate_prompt
from .store import classify, load_locale, save_locale
from .validate import check_translation

_print_lock = threading.Lock()
_model_note_shown = False
_cancel = threading.Event()  # set on Ctrl+C: workers stop at the next batch boundary
# Requests in flight across the whole run, set by run_translate_jobs: a module
# whose batches are independent runs them in parallel, and all modules share
# these `workers` slots.
_batch_slots: threading.BoundedSemaphore | None = None

HEARTBEAT_SEC = 30.0
# the shortest wait for a part of a split batch (see translate_batch)
SPLIT_TIMEOUT_MIN_SEC = 30.0

# Requests timed out in a row across the run, reset by any reply: a string the
# model loops on times out alone, while a dead server times them all out.
_timeouts_lock = threading.Lock()
_timeouts_in_row = 0


def _note_timeout(e: OllamaError, workers: int) -> None:
    global _timeouts_in_row
    with _timeouts_lock:
        _timeouts_in_row += 1
        if _timeouts_in_row >= max(4, 2 * workers):
            raise e


def _note_reply() -> None:
    global _timeouts_in_row
    with _timeouts_lock:
        _timeouts_in_row = 0


class TranslateInterrupted(Exception):
    """Ctrl+C during translate. Batches completed so far are already saved
    (locale files are written after every batch) — a rerun continues."""


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
    arbitrated: int = 0       # semantic disputes auto-resolved by forced choice
    weak: bool = False        # arbitration ratio above WEAK_ARBITRATION_RATIO


WEAK_ARBITRATION_RATIO = 0.25  # >= 25% arbitrated -> the model looks weak here
# Only on a meaningful sample of FRESH pairs (this run's phase-A output).
# Backlog re-judging does not count: an arbitrated entry from an earlier run
# being replaced by a new B candidate says nothing about phase-A weakness
# (live misfire: 6 fresh strings + backlog pairs -> bogus warnings on ru/es/sv).
WEAK_MIN_SAMPLE = 15


def _effective_model(project: Project, lang: str) -> str:
    """[models] per-language override, else the main [ollama] model."""
    return project.models.get(lang, project.ollama.model)


def _maybe_warn_weak(project: Project, lang: str, result: LangResult, tag: str) -> None:
    """Loud per-language escalation: too many strings needed arbitration —
    the model likely does not know this language. No auto-swap (VRAM thrash);
    a mandatory warning with a concrete command instead."""
    if result.translated < WEAK_MIN_SAMPLE:
        return
    ratio = result.arbitrated / result.translated
    if ratio < WEAK_ARBITRATION_RATIO:
        return
    result.weak = True
    used = _effective_model(project, lang)
    alt = "qwen3.5:9b" if used != "qwen3.5:9b" else "gemma4:12b-it-qat"
    _log(f"[{tag}] WARNING: model looks weak in {lang} "
         f"({round(100 * ratio)}% arbitrated); retranslate with an alternate "
         f"model: translate --force --lang {lang} --model {alt} "
         f"(or pin it: [models] \"{lang}\" = \"{alt}\")")


def _batches(keys: list[str], size: int) -> list[list[str]]:
    return [keys[i:i + size] for i in range(0, len(keys), size)]


def translate_batch(
    project: Project,
    client: OllamaClient,
    lang: str,
    keys: list[str],
    locale: dict[str, dict],
    extra_note: str | None = None,
    semantic: bool = False,
    prompt: str | None = None,
    timeout: float | None = None,
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, tuple[str, str, int]]]:
    """Translate one batch. Returns (accepted {key: text}, failed {key: errors},
    arbitrated {key: (text, reason, n_candidates)}).

    Validation failures trigger a retry of the failed subset with the error text
    appended to the conversation, up to max_retries. `extra_note` is prepended
    to the examples block (used by --fix-terms to pin a terminology variant).

    With `semantic=True` every validation-accepted string additionally passes
    the back-translation QA (semantic.check_texts, same model): suspects are
    bounced back into the retry loop with the judge's feedback. When retries
    are exhausted on a PURELY SEMANTIC dispute, the judge arbitrates a forced
    choice among all collected candidates — the winner lands in `accepted`
    AND in `arbitrated`; `failed` remains only for mechanics
    (transport errors, broken placeholders — those would break the app and
    are not a matter of taste). Costs two extra LLM calls per batch
    (~2.5-3x slower) — that is the point: safety over speed."""
    entries = [project.strings[k] for k in keys]
    schema = string_schema(keys)
    if prompt is None:  # the ensemble builds it once for both translators
        prompt = build_translate_prompt(project, lang, entries, locale, extra_note)
    messages = [{"role": "user", "content": prompt}]

    accepted: dict[str, str] = {}
    pending = {e.key: e for e in entries}
    last_errors: dict[str, list[str]] = {}
    candidates: dict[str, list[str]] = {}  # semantically bounced texts per key

    timed_out = False  # the last request got no reply in time
    for attempt in range(project.ollama.max_retries + 1):
        try:
            # a timeout is not sent again as is: the batch is split below
            parsed, raw = client.chat(messages, schema, timeout_tries=1, timeout=timeout)
        except OllamaBadResponse as e:
            # empty/garbled reply is retryable, not fatal
            _note_reply()
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
        except OllamaError as e:
            if not e.timed_out:
                raise
            # A model that loops on one string (a line of "mmmm...", one
            # sentence repeated) never finishes the reply. Too many timeouts
            # in a row mean the server itself is down: that one is fatal.
            _note_timeout(e, project.ollama.workers)
            timed_out = True
            last_errors = {k: [str(e)] for k in pending}
            break
        _note_reply()
        last_errors = {}
        newly: dict[str, str] = {}
        for key, entry in list(pending.items()):
            value = parsed.get(key)
            errors = check_translation(entry.text, value, project.length_ratio,
                                       project.placeholder_style) \
                if value is not None else ["missing from response"]
            if not errors and project.script_check:
                # wrong-script candidates are rejected inline (live case: both
                # cloud models write Serbian in Latin unless pushed back)
                from .langnames import expected_scripts
                from .validate import script_error
                allow = frozenset(
                    {t.casefold() for t in project.latin_allowlist}
                    | {w.casefold() for term in project.glossary for w in term.split()})
                err = script_error(value, expected_scripts(lang),
                                   project.latin_token_max, allow,
                                   project.placeholder_style)
                if err:
                    errors = [err]
            if not errors:
                newly[key] = value
                del pending[key]
            else:
                last_errors[key] = errors

        if semantic and newly:
            from .semantic import check_texts
            try:
                qa_verdicts = check_texts(project, lang, newly, client)
            except OllamaBadResponse:
                # unusable judge output after retries: bounce the batch into
                # the normal retry loop rather than shipping unchecked text
                for key in list(newly):
                    pending[key] = project.strings[key]
                    last_errors[key] = ["semantic check unavailable "
                                        "(judge reply unusable); retrying"]
                    del newly[key]
                qa_verdicts = {}
            for key, (ok, reason, back) in qa_verdicts.items():
                if ok:
                    continue
                pending[key] = project.strings[key]
                candidates.setdefault(key, []).append(newly[key])
                # both sides of the conflict stay visible in the retry prompt
                last_errors[key] = [
                    f"meaning drift: {reason or 'the meaning was not preserved'}"
                    + f' (translation: "{newly[key]}"'
                    + (f', literally back-translates as "{back}")' if back else ")")
                    + "; translate the MEANING of the source text"]
                del newly[key]
        accepted.update(newly)

        if not pending or attempt == project.ollama.max_retries:
            break
        flat = [f"{k}: {'; '.join(v)}" for k, v in last_errors.items()]
        retry_prompt = build_retry_prompt(project, list(pending.values()), flat)
        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": retry_prompt},
        ]
        schema = string_schema(list(pending))

    # arbitration: retries exhausted on a purely semantic dispute -> the judge
    # makes a forced choice among the collected candidates (they all passed
    # placeholder validation, so any of them is safe to ship)
    arbitrated: dict[str, tuple[str, str, int]] = {}
    if timed_out and len(pending) > 1:
        # Split what the model could not answer, so one string it loops on
        # takes no other strings down with it; alone, it stays failed.
        rest = list(pending)
        mid = len(rest) // 2
        _log(f"splitting {len(rest)} strings in two after: {last_errors[rest[0]][0]}")
        whole = timeout or project.ollama.timeout_sec
        for half in (rest[:mid], rest[mid:]):
            # a smaller request answers sooner: wait for it in proportion
            part = min(whole, max(SPLIT_TIMEOUT_MIN_SEC, whole * len(half) / len(rest)))
            acc, fail, arb = translate_batch(project, client, lang, half, locale,
                                             extra_note, semantic, timeout=part)
            accepted.update(acc)
            arbitrated.update(arb)
            for key in half:
                pending.pop(key, None)
                last_errors.pop(key, None)
            last_errors.update(fail)
        return accepted, last_errors, arbitrated

    if semantic:
        from .semantic import arbitrate
        for key in list(pending):
            cands = candidates.get(key, [])
            errs = last_errors.get(key, [])
            if cands and errs and all(e.startswith("meaning drift") for e in errs):
                winner, reason = arbitrate(project, lang, key, cands, client)
                arbitrated[key] = (winner, reason, len(dict.fromkeys(cands)))
                accepted[key] = winner
                del pending[key]
                last_errors.pop(key, None)
    return accepted, last_errors, arbitrated


def translate_language(
    project: Project,
    lang: str,
    keys: list[str],
    transport=None,
) -> LangResult:
    """Translate `keys` for one language, saving the locale after every batch."""
    if project.semantic_inline_judge:
        return _judged_translate_language(project, lang, keys, transport)

    from dataclasses import replace as dc_replace

    result = LangResult(lang=lang)
    model_name = _effective_model(project, lang)  # [models] per-language override
    cfg = project.ollama if model_name == project.ollama.model \
        else dc_replace(project.ollama, model=model_name)
    client = OllamaClient(cfg, transport=transport)
    locale = load_locale(project.root, lang)
    order = list(project.strings)

    orphans = [k for k in locale if k not in project.strings]
    result.orphans_removed = len(orphans)

    # inline semantic QA: default mode, same model. A separate judge model
    # either runs concurrently (semantic_mode = "inline", small judge fits
    # in VRAM next to the main model) or switches the run to phased mode
    # (CLI level) — swapping big models per batch would thrash a 12 GB GPU.
    inline_qa = project.semantic_mode == "batch" and not project.semantic_model

    tag = f"{project.module}:{lang}" if project.module else lang
    batches = _batches(keys, project.ollama.batch_size)
    lock = threading.Lock()  # guards locale, result and the locale file
    stop = threading.Event()  # a batch failed fatally: start no more
    # Without inline QA a batch's results are unverified and never become
    # few-shot examples (select_examples is trusted-only), so the batches do
    # not depend on each other and may run in parallel.
    parallel = not inline_qa and _batch_slots is not None and len(batches) > 1

    def run_batch(i: int, batch: list[str]) -> None:
        if _cancel.is_set() or stop.is_set():
            return  # interrupted: completed batches are on disk already
        label = f"[{tag}] batch {i}/{len(batches)}"
        slot = _batch_slots or contextlib.nullcontext()
        with slot:
            if _cancel.is_set() or stop.is_set():
                return
            _log(f"{label}: translating {len(batch)} strings...")
            _note_model_load(model_name)
            started = time.monotonic()
            with lock:
                snapshot = dict(locale) if parallel else locale
            try:
                with _heartbeat(label):
                    accepted, failed, arbitrated = translate_batch(
                        project, client, lang, batch, snapshot, semantic=inline_qa)
            except OllamaError as e:
                with lock:
                    if result.error is None:
                        result.error = str(e)
                stop.set()
                _log(f"{label}: FATAL {e}")
                return
        elapsed = time.monotonic() - started
        with lock:
            result.arbitrated += len(arbitrated)
            for key, text in accepted.items():
                locale[key] = {"text": text, "src": text_hash(project.strings[key].text),
                               "model": model_name, "translated_at": utc_now_iso()}
            result.translated += len(accepted)
            result.failed.update(failed)
            save_locale(project.root, lang, locale, order)  # progress survives interruption
        if inline_qa and accepted:
            from .semantic import mark_arbitrated, mark_ok
            verified = {k: t for k, t in accepted.items() if k not in arbitrated}
            mark_ok(project, lang, verified)  # verified -> cache: reruns skip them
            mark_arbitrated(project, lang,
                            {k: (t, r) for k, (t, r, _n) in arbitrated.items()})
        for key, (text, reason, n) in arbitrated.items():
            _log(f'[{tag}] ARBITRATED {key}: chose "{text}" over {n} candidate(s)'
                 + (f" ({reason})" if reason else ""))
        _log(f"{label}: +{len(accepted)} in {elapsed:.1f}s"
             + (f", failed {len(failed)}" if failed else ""))

    if parallel:
        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(project.ollama.workers, len(batches)))
        try:
            for fut in [pool.submit(run_batch, i, b) for i, b in enumerate(batches, 1)]:
                fut.result()
        finally:
            # on Ctrl+C the queued batches see _cancel and return at once
            pool.shutdown(wait=False, cancel_futures=True)
    else:
        for i, batch in enumerate(batches, 1):
            run_batch(i, batch)
            if _cancel.is_set() or stop.is_set():
                break
    if _cancel.is_set():
        return result

    if result.error is None and (orphans or not batches):
        save_locale(project.root, lang, locale, order)  # drop orphans even with empty delta
    if result.error is None:
        _maybe_warn_weak(project, lang, result, tag)
    return result


def _judged_translate_language(
    project: Project,
    lang: str,
    keys: list[str],
    transport=None,
) -> LangResult:
    """Concurrent inline judge ([validation] semantic_mode = "inline"): the
    main model translates at full speed while a separate (small) judge model
    checks the PREVIOUS batch in parallel — batch N translates while N-1 is
    being judged. A locale entry is written only after an "ok" verdict;
    suspects are re-translated by the main model with the judge's feedback
    and re-judged. On interruption unverified batches are simply not written
    — they stay pending and the next run redoes them."""
    from dataclasses import replace as dc_replace

    from .semantic import check_texts, mark_ok

    result = LangResult(lang=lang)
    model_name = _effective_model(project, lang)  # [models] per-language override
    cfg = project.ollama if model_name == project.ollama.model \
        else dc_replace(project.ollama, model=model_name)
    client = OllamaClient(cfg, transport=transport)
    judge = OllamaClient(dc_replace(project.ollama, model=project.semantic_model),
                         transport=transport)
    locale = load_locale(project.root, lang)
    order = list(project.strings)
    orphans = [k for k in locale if k not in project.strings]
    result.orphans_removed = len(orphans)

    tag = f"{project.module}:{lang}" if project.module else lang
    batches = _batches(keys, project.ollama.batch_size)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    inflight: tuple[str, dict[str, str], concurrent.futures.Future] | None = None
    # key -> (judge reason, back-translation, flagged candidate text)
    to_retry: dict[str, tuple[str, str, str]] = {}

    def commit(texts: dict[str, str], verified: bool = True) -> None:
        if not texts:
            return
        for key, text in texts.items():
            locale[key] = {"text": text, "src": text_hash(project.strings[key].text),
                           "model": model_name, "translated_at": utc_now_iso()}
        result.translated += len(texts)
        save_locale(project.root, lang, locale, order)
        if verified:
            mark_ok(project, lang, texts)

    def resolve(entry) -> None:
        label, accepted, fut = entry
        verdicts = fut.result()
        good = {}
        for key, text in accepted.items():
            ok, reason, back = verdicts.get(key, (True, "", ""))
            if ok:
                good[key] = text
            else:
                to_retry[key] = (reason, back, text)
        commit(good)
        if len(good) < len(accepted):
            _log(f"{label}: judge flagged {len(accepted) - len(good)} suspect(s)")

    try:
        for i, batch in enumerate(batches, 1):
            if _cancel.is_set():  # unverified stays pending, verified is on disk
                return result
            label = f"[{tag}] batch {i}/{len(batches)}"
            _log(f"{label}: translating {len(batch)} strings...")
            _note_model_load(model_name)
            started = time.monotonic()
            try:
                with _heartbeat(label):
                    accepted, failed, _arb = translate_batch(project, client, lang, batch,
                                                             locale, semantic=False)
            except OllamaError as e:
                result.error = str(e)
                _log(f"{label}: FATAL {e}")
                return result
            result.failed.update(failed)
            if inflight is not None:  # judge of batch i-1 ran while we translated
                resolve(inflight)
                inflight = None
            if accepted:
                inflight = (label, accepted,
                            pool.submit(check_texts, project, lang, dict(accepted), judge))
            _log(f"{label}: +{len(accepted)} translated in "
                 f"{time.monotonic() - started:.1f}s, judging in background"
                 + (f", failed {len(failed)}" if failed else ""))
        if inflight is not None and not _cancel.is_set():
            resolve(inflight)
            inflight = None

        # retry cycle: main model re-translates suspects with the judge's
        # feedback, the judge re-checks synchronously; a still-standing
        # dispute goes to arbitration — a forced choice among the candidates:
        # auto-resolve, log, never block
        if to_retry and not _cancel.is_set():
            from .semantic import arbitrate, mark_arbitrated

            retry_keys = [k for k in order if k in to_retry]
            note = ("A separate QA judge flagged your previous translations "
                    "as meaning drift:\n" + "\n".join(
                        f'- {k}: {reason or "meaning not preserved"} '
                        f'(literal back-translation: "{back}")'
                        for k, (reason, back, _t) in to_retry.items())
                    + "\nTranslate the MEANING of the source text correctly.")
            for rbatch in _batches(retry_keys, project.ollama.batch_size):
                if _cancel.is_set():
                    return result
                accepted, failed, _arb = translate_batch(project, client, lang, rbatch,
                                                         locale, extra_note=note,
                                                         semantic=False)
                result.failed.update(failed)  # mechanics only (placeholders etc.)
                verdicts = check_texts(project, lang, accepted, judge)
                good = {}
                arbitrated: dict[str, tuple[str, str]] = {}
                for key, text in accepted.items():
                    ok, reason, back = verdicts.get(key, (True, "", ""))
                    if ok:
                        good[key] = text
                        continue
                    cands = [to_retry[key][2], text] if key in to_retry else [text]
                    winner, why = arbitrate(project, lang, key, cands, judge)
                    arbitrated[key] = (winner, why)
                    _log(f'[{tag}] ARBITRATED {key}: chose "{winner}" over '
                         f"{len(dict.fromkeys(cands))} candidate(s)"
                         + (f" ({why})" if why else ""))
                commit(good)
                commit({k: t for k, (t, _r) in arbitrated.items()}, verified=False)
                mark_arbitrated(project, lang, arbitrated)
                result.arbitrated += len(arbitrated)
    except OllamaError as e:  # judge unreachable: verified work is on disk
        result.error = str(e)
        _log(f"[{tag}] FATAL (judge): {e}")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    if result.error is None and (orphans or not batches):
        save_locale(project.root, lang, locale, order)  # drop orphans even with empty delta
    if result.error is None and not _cancel.is_set():
        _maybe_warn_weak(project, lang, result, tag)
    return result


ANCHOR_COLD_RATIO = 0.10  # cold start: fewer than 10% of strings accepted


def _anchor_order(project: Project, keys: list[str], ok_count: int) -> list[str]:
    """Cold-start ordering: when a language is (almost) empty,
    translate ANCHORS first — one representative string per frequent term,
    picked for the richest context and the shortest text. They pass the usual
    semantic QA, so the rest of the volume sees verified anchors in few-shot
    instead of learning terms from unchecked neighbours. Incremental runs
    (>= 10% accepted) keep the plain strings.toml order."""
    total = len(project.strings) or 1
    if not keys or ok_count >= ANCHOR_COLD_RATIO * total:
        return keys
    anchors = _anchors(project, keys)
    seen = set(anchors)
    return anchors + [k for k in keys if k not in seen]


def _anchors(project: Project, keys: list[str]) -> list[str]:
    """One representative of `keys` per frequent term (see _anchor_order)."""
    from .terms import get_terms

    keyset = set(keys)
    anchors: list[str] = []
    seen: set[str] = set()
    for _term, term_keys in get_terms(project).items():
        cands = [k for k in term_keys if k in keyset]
        if not cands:
            continue
        best = min(cands, key=lambda k: (-len(project.strings[k].context),
                                         len(project.strings[k].text), k))
        if best not in seen:
            seen.add(best)
            anchors.append(best)
    return anchors


def plan(project: Project, langs: list[str], force: bool, limit: int | None,
         retranslate_model: str | None = None) -> dict[str, list[str]]:
    """{lang: keys to translate} respecting --force and --limit.

    --force means force: everything is retranslated, manual fixes included.
    --retranslate-model is an honest fnmatch over the stored `model` field,
    no special cases: "gemma*" naturally does not match "manual", while an
    explicit --retranslate-model manual does — that is a deliberate request.
    A manual entry whose BASE text changed is stale like any other and is
    retranslated (the text it fixed is gone); the CLI reports those keys."""
    out: dict[str, list[str]] = {}
    for lang in langs:
        locale = load_locale(project.root, lang)
        st = classify(project, lang, locale)
        keys = list(project.strings) if force else st.pending
        if retranslate_model and not force:
            chosen = set(keys)
            extra = {k for k in st.ok
                     if fnmatchcase(locale[k].get("model", ""), retranslate_model)}
            keys = [k for k in project.strings if k in chosen or k in extra]
        keys = _anchor_order(project, keys, len(st.ok))
        if limit is not None:
            keys = keys[:limit]
        out[lang] = keys
    return out


def manual_stale(project: Project, langs: list[str]) -> dict[str, list[str]]:
    """{lang: manual keys that are stale} — the base text changed under a
    manual fix; they go back to the LLM and the CLI reports them."""
    out: dict[str, list[str]] = {}
    for lang in langs:
        locale = load_locale(project.root, lang)
        st = classify(project, lang, locale)
        ms = [k for k in st.stale if locale[k].get("model") == "manual"]
        if ms:
            out[lang] = ms
    return out


def _run_phase(indices, fn, workers: int) -> None:
    """Run one ensemble phase over language jobs, `workers` in parallel.
    With cloud models there is no VRAM constraint — parallelism is limited
    only by the provider; on Ctrl+C never wait for in-flight requests."""
    if workers <= 1:
        for i in indices:
            if _cancel.is_set():
                return
            fn(i)
        return
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    futures = [pool.submit(fn, i) for i in indices]
    try:
        for fut in concurrent.futures.as_completed(futures):
            fut.result()
    except KeyboardInterrupt:
        _cancel.set()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)


def _ensemble_batch_language(project: Project, lang: str, keys: list[str],
                             transport=None,
                             include_backlog: bool = False) -> LangResult:
    """Concurrent per-batch ensemble ([validation].ensemble_concurrency =
    "batch", or "auto" with an all-cloud model trio): the phases were a
    VRAM-swap artifact — with cloud models A and B translate THE SAME batch
    in parallel (two concurrent requests) and the judge follows immediately.
    Nothing is written before the verdict: an interrupted batch leaves no
    partial pairs, the whole batch stays pending for the next run."""
    from dataclasses import replace as dc_replace

    from .semantic import (ensemble_compare, load_semantic, mark_arbitrated,
                           mark_ok, _pair_hash)

    result = LangResult(lang=lang)
    model_a = _effective_model(project, lang)
    client_a = OllamaClient(
        project.ollama if model_a == project.ollama.model
        else dc_replace(project.ollama, model=model_a), transport=transport)
    client_b = OllamaClient(dc_replace(project.ollama, model=project.ensemble_model),
                            transport=transport)
    judge = OllamaClient(dc_replace(project.ollama, model=project.ensemble_judge),
                         transport=transport)
    locale = load_locale(project.root, lang)
    order = list(project.strings)
    orphans = [k for k in locale if k not in project.strings]
    result.orphans_removed = len(orphans)
    tag = f"{project.module}:{lang}" if project.module else lang
    judged_total = 0
    b_wins = 0
    processed: set[str] = set()
    # guards locale, result, the counters and the locale/semantic files while
    # batches run in parallel; network calls are made outside it
    commit_lock = threading.Lock()

    # strong third-party judge (produced neither candidate) compares the
    # pair against the original DIRECTLY — no back-translations, one call
    # per batch (see semantic.ensemble_compare)
    direct = project.ensemble_judge not in (model_a, project.ensemble_model)

    def judge_and_commit(pairs: dict[str, tuple[str, str]],
                         a_new: dict[str, str], label: str = "") -> None:
        """C-judge the pairs, then write everything in ONE save: A-only keys
        as unverified-pending, agreed as ok, disputes as arbitrated."""
        started = time.monotonic()
        try:
            verdicts = ensemble_compare(project, lang, pairs, judge,
                                        direct=direct) if pairs else {}
        except OllamaBadResponse as e:
            # judge unusable after retries: ship nothing from this batch's
            # pairs (they stay pending), keep A-only keys unverified
            _log(f"{label or f'[{tag}] C'}: warning — skipping "
                 f"{len(pairs)} pair(s), judge reply unusable: {e}")
            pairs = {}
            verdicts = {}
        with commit_lock:
            commit(pairs, verdicts, a_new, label, started)

    def commit(pairs, verdicts, a_new, label, started) -> None:
        nonlocal judged_total, b_wins
        for key, text in a_new.items():
            locale[key] = {"text": text, "src": text_hash(project.strings[key].text),
                           "model": model_a, "translated_at": utc_now_iso()}
        result.translated += len(a_new)
        agreed: dict[str, str] = {}
        arbitrated: dict[str, tuple[str, str]] = {}
        for k, (a_text, b_text) in pairs.items():
            v = verdicts.get(k, "c1")
            if k in a_new:  # weak stats count FRESH phase-A pairs only
                judged_total += 1
            processed.add(k)
            if v == "agree":
                agreed[k] = locale[k]["text"] if k not in a_new else a_text
                continue
            winner = a_text if v == "c1" else b_text
            if v == "c2":
                if k in a_new:
                    b_wins += 1
                locale[k] = {"text": winner,
                             "src": text_hash(project.strings[k].text),
                             "model": project.ensemble_model,
                             "translated_at": utc_now_iso()}
            arbitrated[k] = (winner, f"ensemble judge picked {v}")
            _log(f'[{tag}] ARBITRATED {k}: chose "{winner}" over 2 candidate(s) '
                 f"(ensemble {v})")
        save_locale(project.root, lang, locale, order)
        mark_ok(project, lang, agreed)
        mark_arbitrated(project, lang, arbitrated)
        result.arbitrated += len(arbitrated)
        if pairs:
            n_c2 = sum(1 for k, (t, r) in arbitrated.items() if r.endswith("c2"))
            _log(f"{label or f'[{tag}] C'}: agree {len(agreed)}, "
                 f"c1 {len(arbitrated) - n_c2}, c2 {n_c2} "
                 f"in {time.monotonic() - started:.1f}s")

    batches = _batches(keys, project.ollama.batch_size)
    # Batches run in a window of up to `workers`, each building its few-shot
    # pool from the verdicts committed when it starts. On a cold start the
    # anchor batches go first, one by one, so the rest learns their terms.
    # (at least one: a language with nothing to translate is here only to drop
    # the strings deleted since, and the pools below are made either way)
    window = max(1, min(project.ollama.workers, len(batches))) if _batch_slots else 1
    serial = len(batches)
    if window > 1:
        ok_count = len(classify(project, lang, locale).ok)
        cold = ok_count < ANCHOR_COLD_RATIO * (len(project.strings) or 1)
        serial = -(-len(_anchors(project, keys)) // project.ollama.batch_size) \
            if cold else 0
    stop = threading.Event()  # a batch failed fatally: start no more

    def ab_batch(n: int, batch: list[str]) -> None:
        if _cancel.is_set() or stop.is_set():
            return
        label = f"[{tag}] A+B {n}/{len(batches)}"
        with _batch_slots or contextlib.nullcontext():
            if _cancel.is_set() or stop.is_set():
                return
            _log(f"{label}: translating {len(batch)} strings (2 models)...")
            with commit_lock:
                snapshot = dict(locale)
            # one prompt for both: the same strings and the same examples
            prompt = build_translate_prompt(
                project, lang, [project.strings[k] for k in batch], snapshot, None)
            try:
                fut_b = ab_pool.submit(translate_batch, project, client_b, lang,
                                       batch, snapshot, None, False, prompt)
                acc_a, failed_a, _ = translate_batch(project, client_a, lang, batch,
                                                     snapshot, None, False, prompt)
                acc_b, _failed_b, _ = fut_b.result()
                with commit_lock:
                    result.failed.update(failed_a)
                pairs = {k: (acc_a[k], acc_b[k]) for k in acc_a if k in acc_b}
                judge_and_commit(pairs, acc_a, label=f"[{tag}] C {n}/{len(batches)}")
            except OllamaError as e:
                with commit_lock:
                    if result.error is None:
                        result.error = str(e)
                stop.set()
                _log(f"[{tag}] FATAL {e}")
                return
        # survivor rule: A's candidate failed validation (e.g. wrong
        # script), B's passed -> B ships as arbitrated, not a failure
        b_only = {k: t for k, t in acc_b.items() if k not in acc_a}
        with commit_lock:
            if b_only:
                for k, t in b_only.items():
                    locale[k] = {"text": t, "src": text_hash(project.strings[k].text),
                                 "model": project.ensemble_model,
                                 "translated_at": utc_now_iso()}
                    result.failed.pop(k, None)
                    _log(f'[{tag}] ARBITRATED {k}: chose "{t}" — the only valid '
                         "candidate (main model candidate failed validation)")
                save_locale(project.root, lang, locale, order)
                from .semantic import mark_arbitrated as _mark_arb
                _mark_arb(project, lang,
                          {k: (t, "only valid candidate") for k, t in b_only.items()})
                result.translated += len(b_only)
                result.arbitrated += len(b_only)
            failed_now = len(result.failed)
        _log(f"{label}: +{len(acc_a) + len(b_only)} judged"
             + (f", failed {failed_now}" if failed_now else ""))

    # B runs beside A in this pool, one per batch in flight
    ab_pool = concurrent.futures.ThreadPoolExecutor(max_workers=window)
    batch_pool = concurrent.futures.ThreadPoolExecutor(max_workers=window) \
        if window > 1 else None
    try:
        # --- fresh delta: A and B race on every batch --------------------
        for n, batch in enumerate(batches[:serial], 1):
            ab_batch(n, batch)
        if batch_pool is not None:
            futures = [batch_pool.submit(ab_batch, n, batch)
                       for n, batch in enumerate(batches, 1) if n > serial]
            for fut in futures:
                fut.result()
        if _cancel.is_set():
            return result
        if stop.is_set():
            raise OllamaError(result.error or "a batch failed")

        # --- backlog (unverified/suspect/arbitrated from earlier runs) ---
        # Re-resolved ONLY on explicit request (translate --fix-semantic /
        # the audit cycle): a daily incremental translate touches nothing but
        # its delta — the backlog must not be a tax on every run.
        from .semantic import is_resolved
        due = []
        if include_backlog:
            entries = load_semantic(project, lang)["entries"]
            for key, se in project.strings.items():
                e = locale.get(key)
                if not (e and e.get("text")) or key in processed \
                        or e.get("model") in ("manual", "imported"):
                    continue
                cached = entries.get(key, {})
                if cached.get("src") == _pair_hash(se.text, e["text"]) \
                        and (cached.get("verdict") == "ok" or is_resolved(cached)):
                    continue  # verified, or a stable disagreement — leave alone
                due.append(key)
        for batch in _batches(due, project.ollama.batch_size):
            if _cancel.is_set():
                return result
            acc_b, _failed_b, _ = translate_batch(project, client_b, lang, batch,
                                                  locale, semantic=False)
            pairs = {k: (locale[k]["text"], acc_b[k]) for k in acc_b}
            judge_and_commit(pairs, {})
    except OllamaError as e:
        if not stop.is_set():  # a windowed batch has reported it already
            result.error = str(e)
            _log(f"[{tag}] FATAL {e}")
    finally:
        ab_pool.shutdown(wait=False, cancel_futures=True)
        if batch_pool is not None:
            batch_pool.shutdown(wait=False, cancel_futures=True)

    if result.error is None and (orphans or not keys) and not _cancel.is_set():
        save_locale(project.root, lang, locale, order)  # orphan cleanup
    if result.error is None and not _cancel.is_set() \
            and judged_total >= WEAK_MIN_SAMPLE \
            and b_wins / judged_total >= WEAK_ARBITRATION_RATIO:
        result.weak = True
        _log(f"[{tag}] WARNING: phase-A model looks weak in {lang} "
             f"({round(100 * b_wins / judged_total)}% of arbitrations lost to "
             f"{project.ensemble_model}); consider pinning: "
             f"[models] \"{lang}\" = \"{project.ensemble_model}\"")
    return result


def _ensemble_run(jobs: list[tuple[Project, str, list[str]]],
                  workers: int = 1,
                  transport=None,
                  include_backlog: bool = False) -> list[tuple[Project, LangResult]]:
    """Ensemble mode: every string is translated by TWO models
    with meaning-only arbitration, strictly phased to minimize model swaps
    (two models do NOT fit into 12 GB together):

      A: the main model (respecting [models] pins) translates the plan delta
         with format validation; results land in locale UNVERIFIED (no cache
         verdict -> excluded from the few-shot pool until phase C).
      B: one swap — the ensemble model independently translates every
         unverified entry (same trusted few-shot pool).
      C: the judge (semantic_model if set, else the ensemble model — zero
         extra swaps by default) compares the candidates per key by MEANING:
         agree -> verdict ok (trusted pool), otherwise the winner is written
         with verdict arbitrated (re-checked by the next semantic pass).

    Interruption is safe at any point: phase A is saved per batch as
    unverified-pending; phases B/C redo whatever is still unverified."""
    from dataclasses import replace as dc_replace

    from .semantic import (ensemble_compare, load_semantic, mark_arbitrated,
                           mark_ok, _pair_hash)

    jobs = order_jobs(jobs)
    results: dict[int, LangResult] = {i: LangResult(lang=jobs[i][1])
                                      for i in range(len(jobs))}

    # concurrent per-batch mode (cloud trio / explicit "batch"): no phases —
    # A+B race per batch, the judge follows inline; languages parallelized
    batch_idx = [i for i in range(len(jobs)) if jobs[i][0].ensemble_batch]
    phased_idx = [i for i in range(len(jobs)) if not jobs[i][0].ensemble_batch]
    if batch_idx:
        _log("=== ensemble (concurrent): A+B in parallel per batch, judge inline ===")

        def _batch_job(i: int) -> None:
            p, lang, keys = jobs[i]
            results[i] = _ensemble_batch_language(p, lang, keys, transport,
                                                  include_backlog=include_backlog)

        _run_phase(batch_idx, _batch_job, workers)
    if not phased_idx:
        return [(jobs[i][0], results[i]) for i in range(len(jobs))]

    # --- phase A: main model over the plan delta -------------------------
    _log("=== ensemble phase A: main model translates the delta ===")

    def _phase_a(i: int) -> None:
        project, lang, keys = jobs[i]
        tag = f"{project.module}:{lang}" if project.module else lang
        result = results[i]
        model_a = _effective_model(project, lang)
        cfg = project.ollama if model_a == project.ollama.model \
            else dc_replace(project.ollama, model=model_a)
        client = OllamaClient(cfg, transport=transport)
        locale = load_locale(project.root, lang)
        order = list(project.strings)
        orphans = [k for k in locale if k not in project.strings]
        result.orphans_removed = len(orphans)
        batches = _batches(keys, project.ollama.batch_size)
        for n, batch in enumerate(batches, 1):
            if _cancel.is_set():
                return
            label = f"[{tag}] A {n}/{len(batches)}"
            _note_model_load(model_a)
            try:
                with _heartbeat(label):
                    accepted, failed, _arb = translate_batch(
                        project, client, lang, batch, locale, semantic=False)
            except OllamaError as e:
                result.error = str(e)
                _log(f"{label}: FATAL {e}")
                break
            result.failed.update(failed)
            result.translated += len(accepted)
            for key, text in accepted.items():
                locale[key] = {"text": text, "src": text_hash(project.strings[key].text),
                               "model": model_a, "translated_at": utc_now_iso()}
            save_locale(project.root, lang, locale, order)  # unverified-pending
            _log(f"{label}: +{len(accepted)}"
                 + (f", failed {len(failed)}" if failed else ""))
        if result.error is None and (orphans or not batches):
            save_locale(project.root, lang, locale, order)

    _run_phase(phased_idx, _phase_a, workers)

    # --- phase B: the ensemble model translates independently ------------
    # Default (incremental) scope: ONLY this run's fresh delta — phase A's
    # output. The backlog (unverified/suspect/arbitrated of earlier runs) is
    # swept only on explicit request (--fix-semantic / audit cycle).
    from .semantic import is_resolved

    b_texts: dict[int, dict[str, str]] = {}
    unverified: dict[int, list[str]] = {}
    for i in phased_idx:
        project, lang, fresh_keys = jobs[i]
        if results[i].error is not None or _cancel.is_set():
            continue
        locale = load_locale(project.root, lang)
        entries = load_semantic(project, lang)["entries"]
        scope = project.strings if include_backlog else \
            {k: project.strings[k] for k in fresh_keys if k in project.strings}
        due = []
        for key, se in scope.items():
            e = locale.get(key)
            # machine translations only: manual fixes are human truth,
            # imported entries are live production strings — both are audited
            # by verify --semantic, never rewritten by the ensemble
            if not (e and e.get("text")) or e.get("model") in ("manual", "imported"):
                continue
            cached = entries.get(key, {})
            if cached.get("src") == _pair_hash(se.text, e["text"]) \
                    and (cached.get("verdict") == "ok" or is_resolved(cached)):
                continue  # verified or a stable (resolved) disagreement
            due.append(key)
        unverified[i] = due
    if any(unverified.values()) and not _cancel.is_set():
        _log("=== ensemble phase B: independent second translation ===")

    def _phase_b(i: int) -> None:
        project, lang, _keys = jobs[i]
        due = unverified.get(i) or []
        if not due:
            return
        tag = f"{project.module}:{lang}" if project.module else lang
        client_b = OllamaClient(dc_replace(project.ollama, model=project.ensemble_model),
                                transport=transport)
        locale = load_locale(project.root, lang)
        texts: dict[str, str] = {}
        for n, batch in enumerate(_batches(due, project.ollama.batch_size), 1):
            if _cancel.is_set():
                break
            label = f"[{tag}] B {n}"
            _note_model_load(project.ensemble_model)
            try:
                with _heartbeat(label):
                    accepted, failed, _arb = translate_batch(
                        project, client_b, lang, batch, locale, semantic=False)
            except OllamaError as e:
                results[i].error = str(e)
                _log(f"{label}: FATAL {e}")
                break
            texts.update(accepted)  # B's validation rejects stay A-only
        b_texts[i] = texts

    _run_phase(phased_idx, _phase_b, workers)

    # --- phase C: meaning-only arbitration -------------------------------
    if any(b_texts.values()) and not _cancel.is_set():
        _log("=== ensemble phase C: meaning arbitration ===")

    def _phase_c(i: int) -> None:
        project, lang, fresh_keys = jobs[i]
        fresh = set(fresh_keys)  # weak stats count fresh phase-A pairs only
        texts = b_texts.get(i) or {}
        if not texts or results[i].error is not None:
            return
        tag = f"{project.module}:{lang}" if project.module else lang
        judge = OllamaClient(dc_replace(project.ollama, model=project.ensemble_judge),
                             transport=transport)
        locale = load_locale(project.root, lang)
        order = list(project.strings)
        result = results[i]
        judged_total = 0
        b_wins = 0
        # third-party judge (produced neither candidate) -> direct comparison,
        # no back-translations (semantic.ensemble_compare)
        direct = project.ensemble_judge not in (_effective_model(project, lang),
                                                project.ensemble_model)
        keys_c = [k for k in order if k in texts and locale.get(k, {}).get("text")]
        c_batches = _batches(keys_c, project.ollama.batch_size)
        for n, batch in enumerate(c_batches, 1):
            if _cancel.is_set():
                break
            started = time.monotonic()
            pairs = {k: (locale[k]["text"], texts[k]) for k in batch}
            try:
                verdicts = ensemble_compare(project, lang, pairs, judge,
                                            direct=direct)
            except OllamaBadResponse as e:
                _log(f"[{tag}] C warning: skipping batch of {len(batch)} "
                     f"(judge reply unusable: {e})")
                continue  # stays unverified, re-judged next run
            except OllamaError as e:
                result.error = str(e)
                _log(f"[{tag}] C FATAL {e}")
                break
            agreed: dict[str, str] = {}
            arbitrated: dict[str, tuple[str, str]] = {}
            batch_c2 = 0
            for k in batch:
                v = verdicts.get(k, "c1")
                if k in fresh:
                    judged_total += 1
                if v == "agree":
                    agreed[k] = locale[k]["text"]
                    continue
                winner = pairs[k][0] if v == "c1" else pairs[k][1]
                if v == "c2":
                    if k in fresh:
                        b_wins += 1
                    batch_c2 += 1
                    locale[k] = {"text": winner,
                                 "src": text_hash(project.strings[k].text),
                                 "model": project.ensemble_model,
                                 "translated_at": utc_now_iso()}
                arbitrated[k] = (winner, f"ensemble judge picked {v}")
                _log(f'[{tag}] ARBITRATED {k}: chose "{winner}" over 2 candidate(s) '
                     f"(ensemble {v})")
            save_locale(project.root, lang, locale, order)
            mark_ok(project, lang, agreed)
            mark_arbitrated(project, lang, arbitrated)
            result.arbitrated += len(arbitrated)
            _log(f"[{tag}] C {n}/{len(c_batches)}: agree {len(agreed)}, "
                 f"c1 {len(arbitrated) - batch_c2}, c2 {batch_c2} "
                 f"in {time.monotonic() - started:.1f}s")
        # Weak heuristic, ensemble edition: a high DISAGREEMENT share is
        # normal for two model families — weakness is the phase-A model
        # systematically LOSING the arbitration to the ensemble model.
        if result.error is None and not _cancel.is_set() \
                and judged_total >= WEAK_MIN_SAMPLE \
                and b_wins / judged_total >= WEAK_ARBITRATION_RATIO:
            result.weak = True
            _log(f"[{tag}] WARNING: phase-A model looks weak in {lang} "
                 f"({round(100 * b_wins / judged_total)}% of arbitrations lost to "
                 f"{project.ensemble_model}); consider pinning: "
                 f"[models] \"{lang}\" = \"{project.ensemble_model}\"")

    _run_phase(phased_idx, _phase_c, workers)

    return [(jobs[i][0], results[i]) for i in range(len(jobs))]


def order_jobs(jobs: list[tuple[Project, str, list[str]]]
               ) -> list[tuple[Project, str, list[str]]]:
    """Group jobs by their effective model ([models] overrides), models in
    first-seen order — each model is loaded at most once per run instead of
    swapping per language (swaps are expensive on 12 GB)."""
    first_seen: dict[str, int] = {}
    for p, lang, _keys in jobs:
        m = _effective_model(p, lang)
        if m not in first_seen:
            first_seen[m] = len(first_seen)
    return sorted(jobs, key=lambda j: first_seen[_effective_model(j[0], j[1])])


def run_translate_jobs(
    jobs: list[tuple[Project, str, list[str]]],
    workers: int,
    transport=None,
    include_backlog: bool = False,
) -> list[tuple[Project, LangResult]]:
    """Run (project, lang, keys) jobs over ONE shared worker pool — in
    workspace mode the pool spans all modules of the run.
    Jobs are grouped by effective model (order_jobs) to minimize model swaps.
    Results come back in the (reordered) job order."""
    global _model_note_shown, _batch_slots
    _model_note_shown = False  # fresh note per run
    _cancel.clear()
    _note_reply()
    _batch_slots = threading.BoundedSemaphore(workers) if workers > 1 else None
    if any(p.ensemble for p, _l, _k in jobs):
        try:
            return _ensemble_run(jobs, workers, transport,
                                 include_backlog=include_backlog)
        except KeyboardInterrupt:
            _cancel.set()
            raise TranslateInterrupted from None
    jobs = order_jobs(jobs)
    results: list[tuple[Project, LangResult] | None] = [None] * len(jobs)
    if workers <= 1:
        try:
            for i, (project, lang, keys) in enumerate(jobs):
                results[i] = (project, translate_language(project, lang, keys, transport))
        except KeyboardInterrupt:
            _cancel.set()
            raise TranslateInterrupted from None
        return results  # type: ignore[return-value]

    # Ctrl+C lands in the main thread (blocked in as_completed) while worker
    # threads may sit in a long HTTP request. Do NOT wait for them: signal
    # cancellation, drop queued futures, shut down without joining and let
    # the CLI exit — in-flight requests die with the process, everything
    # completed is already saved (locale written after every batch).
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    futures = {
        pool.submit(translate_language, project, lang, keys, transport): i
        for i, (project, lang, keys) in enumerate(jobs)
    }
    try:
        for fut in concurrent.futures.as_completed(futures):
            i = futures[fut]
            results[i] = (jobs[i][0], fut.result())
    except KeyboardInterrupt:
        _cancel.set()                                  # workers stop at batch boundary
        pool.shutdown(wait=False, cancel_futures=True)  # never block on live requests
        raise TranslateInterrupted from None
    pool.shutdown(wait=True)
    return results  # type: ignore[return-value]


def run_translate(
    project: Project,
    langs: list[str],
    force: bool = False,
    limit: int | None = None,
    transport=None,
    retranslate_model: str | None = None,
    include_backlog: bool = False,
) -> list[LangResult]:
    work = plan(project, langs, force, limit, retranslate_model)
    jobs = [(project, lang, keys) for lang, keys in work.items()]
    return [r for _, r in run_translate_jobs(jobs, project.ollama.workers, transport,
                                             include_backlog=include_backlog)]
