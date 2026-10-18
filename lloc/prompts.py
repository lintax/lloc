"""Prompt templates and few-shot example selection."""
from __future__ import annotations

import functools
import math
import re
import threading
from pathlib import Path

from .config import Project, StringEntry
from .langnames import language_name
from .validate import strip_placeholders

DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_WORD_RE = re.compile(r"[A-Za-zÀ-ɏЀ-ӿ]{3,}")


class _SafeDict(dict):
    def __missing__(self, key):  # leave unknown placeholders intact
        return "{" + key + "}"


def prompts_dir(project: Project) -> Path:
    if project.ollama.prompts_dir:
        return (project.root / project.ollama.prompts_dir).resolve()
    return DEFAULT_PROMPTS_DIR


def load_template(project: Project, name: str) -> str:
    """A custom prompts_dir may hold only the templates it overrides; the
    rest come from the built-in set."""
    path = prompts_dir(project) / name
    if not path.is_file():
        path = DEFAULT_PROMPTS_DIR / name
    return path.read_text(encoding="utf-8")


def render(template: str, **kw: str) -> str:
    return template.format_map(_SafeDict(kw))


# --- blocks -------------------------------------------------------------

def glossary_block(project: Project, lang: str) -> str:
    lines = []
    for term, translations in project.glossary.items():
        val = translations.get(lang, translations.get("*"))
        if val is not None:
            lines.append(f'- "{term}" -> "{val}"')
    return "\n".join(lines) or "(none)"


def strings_block(entries: list[StringEntry]) -> str:
    lines = []
    for e in entries:
        lines.append(f"key: {e.key}")
        lines.append(f"text: {e.text}")
        if e.context:
            lines.append(f"context: {e.context}")
        if e.type:
            lines.append(f"type: {e.type}")
        lines.append("")
    return "\n".join(lines).rstrip()


@functools.lru_cache(maxsize=1 << 18)
def _words(text: str) -> frozenset[str]:
    """Cached: few-shot selection scores the same pool on every batch."""
    return frozenset(w.lower() for w in _WORD_RE.findall(text))


_TRUST_MANUAL = 2000    # human fixes: golden examples
_TRUST_VERIFIED = 1000  # passed the semantic judge for exactly this base+translation
_TRUST_IMPORTED = 0     # production translations: trusted, lowest tier


def select_examples(
    project: Project, batch: list[StringEntry], locale: dict[str, dict],
    limit: int = 8, lang: str | None = None,
) -> list[tuple[str, str]]:
    """Pick already-translated strings for few-shot:
    strings sharing extracted *terms* with the batch score far above mere
    word overlap. In workspace mode accepted translations of
    same-group sibling modules are candidates too (pass `lang`); the group is
    the boundary — other groups never contribute. Returns [(base, translated)].

    The pool is TRUSTED-ONLY: manual fixes, judge-verified
    entries (semantic/<lang>.json verdict ok for exactly this base+translation
    pair) and imported production translations, ranked in that order.
    Unverified entries (e.g. translated with semantic="off") never become
    examples — an early wrong term cannot multiply through few-shot."""
    from . import terms as terms_mod  # local import: terms.py imports prompts helpers
    from .semantic import _pair_hash, verified_pairs

    batch_keys = {e.key for e in batch}
    batch_words = set().union(*(_words(e.text) for e in batch)) if batch else set()
    batch_terms = set().union(*(terms_mod.text_terms(project, e.text) for e in batch)) \
        if batch else set()

    def trust(entry: dict, key: str, strings, verified: dict[str, str]) -> int | None:
        if entry.get("model") == "manual":
            return _TRUST_MANUAL
        se = strings.get(key)
        if se is not None and verified.get(key) == _pair_hash(se.text, entry.get("text", "")):
            return _TRUST_VERIFIED
        if entry.get("model") == "imported":
            return _TRUST_IMPORTED
        return None  # unverified: never a few-shot example

    def score_pool(pool: dict[str, dict], strings, skip_keys, verified):
        for key, entry in pool.items():
            if key in skip_keys or key not in strings or not entry.get("text"):
                continue
            bonus = trust(entry, key, strings, verified)
            if bonus is None:
                continue
            base_text = strings[key].text
            shared_terms = len(terms_mod.text_terms(project, base_text) & batch_terms)
            score = shared_terms * 10 + len(_words(base_text) & batch_words)
            if score > 0:
                scored.append((score + bonus, base_text, entry["text"]))

    scored: list[tuple[int, str, str]] = []
    own_verified = verified_pairs(project.root, lang) if lang else {}
    score_pool(locale, project.strings, batch_keys, own_verified)
    if lang is not None:
        from .store import load_locale
        for sib in project.siblings:
            score_pool(load_locale(sib.root, lang), sib.strings, batch_keys,
                       verified_pairs(sib.root, lang))
    scored.sort(key=lambda t: (-t[0], t[1]))
    picked = [(b, t) for _, b, t in scored[:limit]]
    if lang is not None and project.references:
        seen = {b for b, _ in picked}
        style = project.placeholder_style
        texts = [strip_placeholders(e.text, style) for e in batch]
        for ref in project.references:
            picked += [p for p in _reference_index(ref, lang, style).search(texts)
                       if p[0] not in seen]
    return picked


# --- reference corpora --------------------------------------------------

REFERENCE_EXAMPLES = 24      # per batch, on top of the regular few-shot limit
REFERENCE_MAX_CHARS = 160    # longer strings are prose: poor term carriers
REFERENCE_POSTINGS = 40      # candidates considered per batch word
REFERENCE_PHRASE_WORDS = 4   # longest word sequence looked up as a whole string

_reference_cache: dict[tuple[str, str], "_ReferenceIndex"] = {}
_reference_lock = threading.Lock()


def _stem(word: str) -> str:
    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"
    return word[:-1] if len(word) > 4 and word.endswith("s") else word


def _sequence(text: str) -> tuple[str, ...]:
    return tuple(_stem(w.lower()) for w in _WORD_RE.findall(text))


class _ReferenceIndex:
    """Inverted word index over the short approved translations of a
    reference module. A corpus of this kind (a whole base game) is far too
    large to score string by string for every batch, and whole-pool word
    overlap would favour its longest paragraphs. Retrieval is therefore
    term-oriented: every word of the batch pulls the shortest strings that
    contain it, and a greedy cover picks the ones that explain the most
    still-uncovered words, rarer words first.

    Before that, word sequences of the batch that ARE a reference string
    ("Stagnant Ascendancy", "Minor Artifacts") are looked up directly: an
    established name must not depend on how the cover happens to fall."""

    def __init__(self, ref, lang: str, style: str):
        from .store import load_locale
        from .terms import _EN_STOPWORDS

        self.pairs: list[tuple[str, str]] = []
        self.words: list[frozenset[str]] = []
        self.phrases: dict[tuple[str, ...], int] = {}
        postings: dict[str, list[int]] = {}
        seen: set[str] = set()
        for key, entry in load_locale(ref.root, lang).items():
            base = ref.strings.get(key)
            text = entry.get("text")
            if base is None or not text or entry.get("model") not in ("imported", "manual") \
                    or len(base.text) > REFERENCE_MAX_CHARS or base.text in seen:
                continue
            sequence = _sequence(strip_placeholders(base.text, style))
            words = frozenset(sequence) - _EN_STOPWORDS
            if not words:
                continue
            seen.add(base.text)
            if 2 <= len(sequence) <= REFERENCE_PHRASE_WORDS:
                self.phrases.setdefault(sequence, len(self.pairs))
            for w in words:
                postings.setdefault(w, []).append(len(self.pairs))
            self.pairs.append((base.text, text))
            self.words.append(words)
        total = len(self.pairs) or 1
        self.idf = {w: math.log(total / len(ids)) + 1.0 for w, ids in postings.items()}
        brevity = lambda i: (len(self.words[i]), len(self.pairs[i][0]), self.pairs[i][0])
        self.postings = {w: sorted(ids, key=brevity)[:REFERENCE_POSTINGS]
                         for w, ids in postings.items()}

    def search(self, texts: list[str], limit: int = REFERENCE_EXAMPLES
               ) -> list[tuple[str, str]]:
        sequences = [_sequence(t) for t in texts]
        hits: dict[int, int] = {}  # reference string -> words it names in the batch
        for seq in sequences:
            for n in range(REFERENCE_PHRASE_WORDS, 1, -1):
                for start in range(len(seq) - n + 1):
                    i = self.phrases.get(seq[start:start + n])
                    if i is not None:
                        hits[i] = hits.get(i, 0) + n
        exact = sorted(hits, key=lambda i: (-hits[i], i))[:limit // 2]
        picked = [self.pairs[i] for i in exact]
        remaining = {w for seq in sequences for w in seq} & self.postings.keys()
        for i in exact:
            remaining -= self.words[i]
        candidates = sorted({i for w in remaining for i in self.postings[w]} - set(exact))
        while remaining and candidates and len(picked) < limit:
            def gain(i: int) -> float:
                shared = self.words[i] & remaining
                # precision: a string made of the wanted words beats a
                # sentence that merely mentions one of them
                return sum(self.idf[w] for w in shared) * len(shared) / len(self.words[i])
            best = max(candidates, key=lambda i: (gain(i), -i))
            if gain(best) <= 0:
                break
            picked.append(self.pairs[best])
            remaining -= self.words[best]
            candidates.remove(best)
        return picked


def _reference_index(ref, lang: str, style: str) -> _ReferenceIndex:
    key = (str(ref.root), lang)
    with _reference_lock:
        if key not in _reference_cache:
            _reference_cache[key] = _ReferenceIndex(ref, lang, style)
        return _reference_cache[key]


def examples_block(examples: list[tuple[str, str]]) -> str:
    if not examples:
        return "(none yet)"
    return "\n".join(f'- "{base}" -> "{tr}"' for base, tr in examples)


# --- full prompts -------------------------------------------------------

def build_translate_prompt(
    project: Project, lang: str, batch: list[StringEntry], locale: dict[str, dict],
    extra_note: str | None = None,
) -> str:
    template = load_template(project, "translate.txt")
    examples = examples_block(select_examples(project, batch, locale, lang=lang))
    if extra_note:
        examples = extra_note + "\n\n" + examples
    return render(
        template,
        # layered context: group context -> module description
        app_context=project.full_context,
        tone=project.tone or "neutral",
        base_language=language_name(project.base_language),
        target_language=language_name(lang),
        glossary_block=glossary_block(project, lang),
        examples_block=examples,
        strings_block=strings_block(batch),
    )


def build_retry_prompt(project: Project, batch: list[StringEntry], errors: list[str]) -> str:
    template = load_template(project, "retry.txt")
    return render(
        template,
        error="\n".join(f"- {e}" for e in errors),
        strings_block=strings_block(batch),
    )
