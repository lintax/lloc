"""Prompt templates and few-shot example selection. SPEC.md §6."""
from __future__ import annotations

import re
from pathlib import Path

from .config import Project, StringEntry

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
    return (prompts_dir(project) / name).read_text(encoding="utf-8")


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


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


def select_examples(
    project: Project, batch: list[StringEntry], locale: dict[str, dict],
    limit: int = 8, lang: str | None = None,
) -> list[tuple[str, str]]:
    """Pick already-translated strings for few-shot (SPEC.md §6, §13.2):
    strings sharing extracted *terms* with the batch score far above mere
    word overlap. In workspace mode (SPEC.md §14) accepted translations of
    same-group sibling modules are candidates too (pass `lang`); the group is
    the boundary — other groups never contribute. Returns [(base, translated)]."""
    from . import terms as terms_mod  # local import: terms.py imports prompts helpers

    batch_keys = {e.key for e in batch}
    batch_words = set().union(*(_words(e.text) for e in batch)) if batch else set()
    batch_terms = set().union(*(terms_mod.text_terms(project, e.text) for e in batch)) \
        if batch else set()

    def score_pool(pool: dict[str, dict], strings, skip_keys):
        for key, entry in pool.items():
            if key in skip_keys or key not in strings or not entry.get("text"):
                continue
            base_text = strings[key].text
            shared_terms = len(terms_mod.text_terms(project, base_text) & batch_terms)
            score = shared_terms * 10 + len(_words(base_text) & batch_words)
            if score > 0:
                scored.append((score, base_text, entry["text"]))

    scored: list[tuple[int, str, str]] = []
    score_pool(locale, project.strings, batch_keys)
    if lang is not None:
        from .store import load_locale
        for sib in project.siblings:
            score_pool(load_locale(sib.root, lang), sib.strings, batch_keys)
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [(b, t) for _, b, t in scored[:limit]]


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
        # layered context (SPEC.md §14): group context -> module description
        app_context=project.full_context,
        tone=project.tone or "neutral",
        base_language=project.base_language,
        target_language=lang,
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
