"""Semantic QA via back-translation (verify --semantic / translate --fix-semantic).

Two-step check for meaning drift the format validators cannot see
(e.g. lv "Rezerves avīze" = "backup NEWSPAPER" for "Backup alarm"):

1. **Back-translation**: the model translates each stored translation back
   into the base language, literally, WITHOUT the key's context — so a wrong
   meaning cannot be "repaired" from surrounding knowledge.
2. **Judge**: the model compares the back-translation against the original
   base text (this time WITH the key's context) and answers per key: "ok",
   or a short reason describing the drift -> the entry becomes a suspect.

Results are cached in `semantic/<lang>.json` (written by the tool, committed)
keyed by a hash of base+translation — reruns only re-check what changed,
like the terms cache. This is an expensive, two-LLM-passes operation: run it
selectively (--lang lv,hy) and/or in the background, not on every commit.

`translate --fix-semantic` re-translates the cached suspects, telling the
model exactly what was wrong (judge's reason + the literal back-translation),
then clears those cache entries so the next --semantic run re-checks them.
"""
from __future__ import annotations

import functools
import json
import threading

from .config import Project
from .fsutil import atomic_write_text
from .hashes import text_hash
from .ollama import OllamaBadResponse, OllamaClient, string_schema


def _chat_json(client: OllamaClient, prompt: str, schema: dict,
               retries: int = 2) -> dict:
    """chat() with retries on unusable replies (empty/garbled JSON), matching
    the translate path. Raises OllamaBadResponse only after the budget is
    exhausted — callers skip the batch with a warning instead of crashing."""
    last: OllamaBadResponse | None = None
    for _attempt in range(retries + 1):
        try:
            parsed, _raw = client.chat([{"role": "user", "content": prompt}], schema)
            return parsed
        except OllamaBadResponse as e:
            last = e
    raise last  # type: ignore[misc]
from .prompts import load_template, render
from .langnames import language_name

SEMANTIC_DIR = "semantic"


def verified_pairs(root, lang: str) -> dict[str, str]:
    """{key: pair-hash} of entries the judge marked ok — the trust source for
    the few-shot pool (prompts.select_examples). Takes a plain root path so
    sibling modules' caches can be read too."""
    from pathlib import Path

    p = Path(root) / SEMANTIC_DIR / f"{lang}.json"
    try:
        st = p.stat()
    except FileNotFoundError:
        return {}
    # asked for on every batch: parse the file again only when it changed
    stamp = (st.st_mtime_ns, st.st_size, st.st_ino)
    with _verified_lock:
        hit = _verified_cache.get(p)
        if hit is not None and hit[0] == stamp:
            return hit[1]
    data = json.loads(p.read_text(encoding="utf-8"))
    pairs = {k: e.get("src", "") for k, e in data.get("entries", {}).items()
             if e.get("verdict") == "ok"}
    with _verified_lock:
        _verified_cache[p] = (stamp, pairs)
    return pairs


_verified_cache: dict = {}
_verified_lock = threading.Lock()


def semantic_path(project: Project, lang: str):
    return project.root / SEMANTIC_DIR / f"{lang}.json"


def load_semantic(project: Project, lang: str) -> dict:
    p = semantic_path(project, lang)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"version": 1, "language": lang, "entries": {}}


def save_semantic(project: Project, lang: str, data: dict) -> None:
    p = semantic_path(project, lang)
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(p, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


@functools.lru_cache(maxsize=1 << 18)
def _pair_hash(base: str, translated: str) -> str:
    return text_hash(base + "\x00" + translated)


def check_texts(project: Project, lang: str, texts: dict[str, str],
                client: OllamaClient) -> dict[str, tuple[bool, str, str]]:
    """Back-translate + judge `texts` ({key: translation}) in one pass.
    Returns {key: (ok, reason, back_translation)}. No cache interaction —
    callers decide what to record."""
    keys = list(texts)
    if not keys:
        return {}
    # step 1: literal back-translation, WITHOUT the key's context — a wrong
    # meaning must not be "repaired" from surrounding knowledge
    back_tpl = load_template(project, "backtranslate.txt")
    strings_block = "\n".join(f"key: {k}\ntext: {texts[k]}\n" for k in keys)
    prompt = render(back_tpl,
                    base_language=language_name(project.base_language),
                    target_language=language_name(lang),
                    strings_block=strings_block)
    back = _chat_json(client, prompt, string_schema(keys),
                      retries=project.ollama.max_retries)

    # cheap shortcut: a back-translation that matches the original verbatim
    # cannot have drifted — trivially ok, no judge call needed
    out: dict[str, tuple[bool, str, str]] = {}
    to_judge = []
    for k in keys:
        b = str(back.get(k, ""))
        if _same_text(b, project.strings[k].text):
            out[k] = (True, "", b)
        else:
            to_judge.append(k)
    if not to_judge:
        return out

    # step 2: judge against the original, WITH the context
    judge_tpl = load_template(project, "semantic_judge.txt")
    items_block = "\n".join(
        f"key: {k}\n"
        f"original: {project.strings[k].text}\n"
        + (f"context: {project.strings[k].context}\n" if project.strings[k].context else "")
        + f"back-translation: {back.get(k, '')}\n"
        for k in to_judge)
    prompt = render(judge_tpl,
                    app_context=project.full_context or "(none)",
                    base_language=language_name(project.base_language),
                    target_language=language_name(lang),
                    items_block=items_block)
    verdicts = _chat_json(client, prompt, string_schema(to_judge),
                          retries=project.ollama.max_retries)
    for k in to_judge:
        v = str(verdicts.get(k, "")).strip()
        ok = v.lower().rstrip(".") == "ok"
        out[k] = (ok, "" if ok else v, str(back.get(k, "")))
    return out


def _same_text(a: str, b: str) -> str:
    """Case/whitespace-insensitive equality for the no-judge shortcut."""
    norm = lambda s: " ".join(s.split()).casefold()  # noqa: E731
    return norm(a) == norm(b)


def arbitrate(project: Project, lang: str, key: str, candidates: list[str],
              client: OllamaClient) -> tuple[str, str]:
    """Forced choice among imperfect candidates: semantic
    conflicts must resolve automatically — a background run has no human, and
    a hole (or a Google-Translate patch) is worse than the least bad
    candidate. Returns (winner, judge's reason)."""
    cands = list(dict.fromkeys(candidates))  # dedupe, keep order
    if not cands:
        raise ValueError("arbitrate: no candidates")
    if len(cands) == 1:
        return cands[0], "the only candidate"
    se = project.strings[key]
    ids = [f"c{i}" for i in range(1, len(cands) + 1)]
    block = "\n".join(f'{i}: "{t}"' for i, t in zip(ids, cands))
    prompt = render(load_template(project, "arbitrate.txt"),
                    app_context=project.full_context or "(none)",
                    base_language=language_name(project.base_language),
                    target_language=language_name(lang),
                    original=se.text,
                    context=se.context or "(none)",
                    candidates_block=block)
    schema = {"type": "object",
              "properties": {"choice": {"type": "string", "enum": ids},
                             "reason": {"type": "string"}},
              "required": ["choice"], "additionalProperties": False}
    parsed = _chat_json(client, prompt, schema,
                        retries=project.ollama.max_retries)
    choice = str(parsed.get("choice", ""))
    idx = ids.index(choice) if choice in ids else 0  # must pick; fallback: first
    return cands[idx], str(parsed.get("reason", "")).strip()


def ensemble_compare(project: Project, lang: str, pairs: dict[str, tuple[str, str]],
                     client: OllamaClient, direct: bool = False) -> dict[str, str]:
    """Ensemble phase C: compare two independent candidate
    translations per key by MEANING only. Returns {key: "agree"|"c1"|"c2"}.
    Verbatim-equal pairs are "agree" without an LLM call; an unparseable
    judge answer falls back to "c1" (keep the main model's text) — the caller
    records it as arbitrated, so it gets re-checked later anyway.

    `direct=True` — the fast path for a STRONG THIRD-PARTY judge (a model
    that produced neither candidate): it compares the candidates against the
    original in ONE call per batch, no back-translations — several times
    fewer output tokens. The back-translation path stays essential for
    self-checks: a model judging its own translation must not be allowed to
    "remember" what it meant."""
    out: dict[str, str] = {}
    todo = []
    for k, (a, b) in pairs.items():
        if _same_text(a, b):
            out[k] = "agree"
        else:
            todo.append(k)
    if not todo:
        return out

    if direct:
        items_block = "\n".join(
            f"key: {k}\n"
            f"original: {project.strings[k].text}\n"
            + (f"context: {project.strings[k].context}\n"
               if project.strings[k].context else "")
            + f'c1: "{pairs[k][0]}"\nc2: "{pairs[k][1]}"\n'
            for k in todo)
        prompt = render(load_template(project, "ensemble_direct.txt"),
                        app_context=project.full_context or "(none)",
                        base_language=language_name(project.base_language),
                        target_language=language_name(lang),
                        items_block=items_block)
        schema = {"type": "object",
                  "properties": {k: {"type": "string",
                                     "enum": ["agree", "c1", "c2"]} for k in todo},
                  "required": todo, "additionalProperties": False}
        parsed = _chat_json(client, prompt, schema,
                            retries=project.ollama.max_retries)
        for k in todo:
            v = str(parsed.get(k, "")).strip().lower()
            out[k] = v if v in ("agree", "c1", "c2") else "c1"
        return out

    back_tpl = load_template(project, "backtranslate.txt")

    def back(texts: dict[str, str]) -> dict[str, str]:
        block = "\n".join(f"key: {k}\ntext: {texts[k]}\n" for k in todo)
        prompt = render(back_tpl,
                        base_language=language_name(project.base_language),
                        target_language=language_name(lang),
                        strings_block=block)
        parsed = _chat_json(client, prompt, string_schema(todo),
                            retries=project.ollama.max_retries)
        return {k: str(parsed.get(k, "")) for k in todo}

    back_a = back({k: pairs[k][0] for k in todo})
    back_b = back({k: pairs[k][1] for k in todo})
    items_block = "\n".join(
        f"key: {k}\n"
        f"original: {project.strings[k].text}\n"
        + (f"context: {project.strings[k].context}\n" if project.strings[k].context else "")
        + f'c1: "{pairs[k][0]}"\nc1 back-translation: {back_a[k]}\n'
        + f'c2: "{pairs[k][1]}"\nc2 back-translation: {back_b[k]}\n'
        for k in todo)
    prompt = render(load_template(project, "ensemble_judge.txt"),
                    app_context=project.full_context or "(none)",
                    base_language=language_name(project.base_language),
                    target_language=language_name(lang),
                    items_block=items_block)
    schema = {"type": "object",
              "properties": {k: {"type": "string", "enum": ["agree", "c1", "c2"]}
                             for k in todo},
              "required": todo, "additionalProperties": False}
    parsed = _chat_json(client, prompt, schema,
                        retries=project.ollama.max_retries)
    for k in todo:
        v = str(parsed.get(k, "")).strip().lower()
        out[k] = v if v in ("agree", "c1", "c2") else "c1"
    return out


def is_resolved(entry: dict) -> bool:
    """Arbitrated twice -> resolved: a stable disagreement between the models
    is accepted (visible in report, overridable via `lloc set`) instead of
    being an eternal tax re-judged on every run."""
    return entry.get("verdict") == "arbitrated" and entry.get("attempts", 1) >= 2


def mark_arbitrated(project: Project, lang: str,
                    items: dict[str, tuple[str, str]]) -> None:
    """Record arbitration winners: verdict "arbitrated" — NOT "ok", so they
    never enter the trusted few-shot pool. Each re-arbitration of the same
    key bumps `attempts`; after 2 the entry counts as RESOLVED (is_resolved)
    and stops being re-judged until its text changes."""
    if not items:
        return
    data = load_semantic(project, lang)
    for key, (text, reason) in items.items():
        base = project.strings.get(key)
        if base is None:
            continue
        old = data["entries"].get(key, {})
        attempts = old.get("attempts", 0) + 1 if old.get("verdict") == "arbitrated" \
            else 1
        data["entries"][key] = {"src": _pair_hash(base.text, text),
                                "verdict": "arbitrated", "reason": reason,
                                "back": "", "attempts": attempts}
    save_semantic(project, lang, data)


def mark_ok(project: Project, lang: str, texts: dict[str, str]) -> None:
    """Record inline-QA-passed translations as verified in the cache, so a
    later `verify --semantic` / phased run only re-checks what changed."""
    if not texts:
        return
    data = load_semantic(project, lang)
    for key, text in texts.items():
        base = project.strings.get(key)
        if base is None:
            continue
        data["entries"][key] = {"src": _pair_hash(base.text, text),
                                "verdict": "ok", "back": ""}
    save_semantic(project, lang, data)


def build_semantic(project: Project, lang: str, locale: dict[str, dict],
                   client: OllamaClient) -> dict:
    """Update the semantic cache for `lang`: back-translate + judge the delta
    (pairs whose base+translation hash is not in the cache). Returns the data."""
    data = load_semantic(project, lang)
    entries = data["entries"]

    todo: list[tuple[str, str]] = []  # (key, pair_hash)
    for key, se in project.strings.items():
        e = locale.get(key)
        if not (e and e.get("text")):
            continue
        h = _pair_hash(se.text, e["text"])
        cached = entries.get(key, {})
        # arbitrated entries are due for a re-check — but only until they are
        # RESOLVED (two arbitrations): stable disagreement is accepted, not
        # re-litigated forever
        if cached.get("src") != h \
                or (cached.get("verdict") == "arbitrated" and not is_resolved(cached)):
            todo.append((key, h))

    bs = project.ollama.batch_size
    for i in range(0, len(todo), bs):
        batch = todo[i:i + bs]
        try:
            results = check_texts(project, lang,
                                  {k: locale[k]["text"] for k, _ in batch}, client)
        except OllamaBadResponse as e:
            # unusable judge output even after retries: skip THIS batch with a
            # warning (it stays unverified and is re-checked next run) — never
            # crash the whole audit
            print(f"[{lang}] warning: skipping a semantic batch of {len(batch)} "
                  f"({e})", flush=True)
            continue
        for key, h in batch:
            ok, reason, back = results.get(key, (True, "", ""))
            entries[key] = {"src": h, "verdict": "ok" if ok else "suspect",
                            "back": back}
            if not ok:
                entries[key]["reason"] = reason
        save_semantic(project, lang, data)  # progress survives interruption
    return data


def suspects(data: dict) -> list[tuple[str, str, str]]:
    """[(key, reason, back-translation)] for suspect entries, key order."""
    return [(k, e.get("reason", ""), e.get("back", ""))
            for k, e in data["entries"].items() if e.get("verdict") == "suspect"]


def clear_keys(project: Project, lang: str, keys: list[str]) -> None:
    """Drop cache entries (after --fix-semantic re-translation: the new text
    gets re-checked by the next --semantic run)."""
    data = load_semantic(project, lang)
    changed = False
    for k in keys:
        if k in data["entries"]:
            del data["entries"][k]
            changed = True
    if changed:
        save_semantic(project, lang, data)


def prune_semantic_caches(project: Project) -> None:
    """Offline cleanup on translate: drop entries for deleted keys."""
    for lang in project.languages:
        p = semantic_path(project, lang)
        if not p.is_file():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        entries = data.get("entries", {})
        orphans = [k for k in entries if k not in project.strings]
        for k in orphans:
            del entries[k]
        if orphans:
            save_semantic(project, lang, data)
