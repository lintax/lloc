"""Automatic terminology consistency.

- extract_terms: frequency-based term candidates from base strings (no LLM).
- text_terms: which terms a text contains (used for few-shot prioritization).
- heuristic_divergences: cheap warning-level detector (no LLM).
- build_term_map / divergences: LLM-built per-language term map, cached in
  terms/<lang>.json with src hashes (incremental).
- fix_language: re-translate the minority side of a divergence with the
  majority variant pinned in the prompt; persistent divergence -> accepted.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .config import Project
from .fsutil import atomic_write_text
from .hashes import text_hash
from .langnames import language_name
from .ollama import OllamaClient, string_schema
from .store import base_hashes, load_locale, save_locale

_WORD_RE = re.compile(r"[^\W\d_]{2,}")
PREFIX = 4  # casefold + common prefix >= 4 chars folds light morphology
STOPWORD_MIN = 4  # DF alone misfires on small string sets; require absolute count too
NON_SEGMENTED_PRIMARY = {"ja", "zh", "th", "ko", "my", "km", "lo"}
TERMS_DIR = "terms"


def _primary(lang: str) -> str:
    return lang.split("-")[0]


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


_AR_WEAK_RE = re.compile("[اويىة]")


def _ar_skeleton(w: str) -> str:
    """Rough consonant skeleton: prefix matching fails for Arabic (definite
    article at the start, broken plurals change the middle), so compare
    skeletons instead: strip ال, weak letters and a trailing ت.
    المشروع -> مشرع, مشاريع -> مشرع."""
    if w.startswith("ال") and len(w) > 4:
        w = w[2:]
    if not w:
        return w
    return w[0] + _AR_WEAK_RE.sub("", w[1:]).rstrip("ت")


def _words_match(a: str, b: str, lang: str = "") -> bool:
    """Same word modulo morphology. Inflection is mostly suffixal, so a common
    prefix of >= max(3, len-2) counts as a match: этот/этого and Projekt/Projekte
    merge, but проект/подпроект and Projekte/Vorhaben stay apart."""
    if a == b:
        return True
    if _primary(lang) == "ar":
        return _ar_skeleton(a) == _ar_skeleton(b)
    return _common_prefix_len(a, b) >= max(3, min(len(a), len(b)) - 2)


def _variants_match(a: str, b: str, lang: str = "") -> bool:
    """Variant comparison, morphology-tolerant on the target side.

    - non-segmented scripts (ja/zh/...): substring containment;
    - equal word counts: pairwise match («этого проекта» ~ «этот проект»);
    - different word counts: every word of the shorter variant must match some
      word of the longer one («проекта» ~ «этот проект», but not vorhaben ~ projekte)."""
    if _primary(lang) in NON_SEGMENTED_PRIMARY:
        return a in b or b in a
    aw, bw = a.split(), b.split()
    if len(aw) == len(bw):
        return all(_words_match(x, y, lang) for x, y in zip(aw, bw))
    short, long_ = (aw, bw) if len(aw) < len(bw) else (bw, aw)
    return all(any(_words_match(s, l, lang) for l in long_) for s in short)


def _tokens(text: str) -> list[str]:
    return [w.casefold() for w in _WORD_RE.findall(text)]


def _glossary_words(project: Project) -> frozenset[str]:
    """Casefolded single-word [glossary] keys. Derivation vs inflection has no
    reliable lexical criterion (edit + editor got prefix-merged and
    --fix-terms lost the "editor" concept), so the glossary is the user's last
    word: such a word is always its own term and is never merged into another
    one, while inflections may still merge INTO it (editors -> editor)."""
    return frozenset(t.casefold() for t in project.glossary if " " not in t.strip())


def _match(token: str, rep: str) -> bool:
    return token == rep or (len(rep) >= PREFIX and token.startswith(rep))


# --- extraction (group-scoped) ---------------------------------------------

def _group_docs(project: Project):
    """(doc_id, entry, own) triples over the whole group: own module strings
    plus same-group siblings. The group is the terminology boundary:
    frequency/stopword statistics are group-wide, other groups contribute
    nothing. In single-project mode siblings are empty — behavior unchanged."""
    for key, e in project.strings.items():
        yield key, e, True
    for sib in project.siblings:
        for key, e in sib.strings.items():
            yield f"{sib.module}::{key}", e, False


# Minimal English function words: articles, prepositions, conjunctions,
# pronouns, auxiliaries and a few adverbs that carry no terminology. Words
# under 3 letters are filtered earlier and are not listed.
_EN_STOPWORDS = frozenset("""
    the and for with from your you our they them their this that these those
    are was were been being will would can could should may might must have
    has had does did not all any some each every more most other another
    into over under again while where when what which who whom whose then
    than also just only very much many such off out about after before
    during between through against
""".split())


def extract_terms(project: Project) -> dict[str, list[str]]:
    """{term: [keys of *own* base strings containing it]}, deterministic order.
    Occurrence counting spans the whole group; the key lists stay module-local
    (the term map and --fix-terms operate on the module's locale)."""
    cfg = project.terms
    occ: dict[str, set[str]] = {}       # word -> group doc ids
    own_occ: dict[str, set[str]] = {}   # word -> own keys
    total = 0
    for doc, e, own in _group_docs(project):
        total += 1
        for w in set(_tokens(e.text)):
            if len(w) >= 3:  # 1-2 letter words are function words in most languages
                occ.setdefault(w, set()).add(doc)
                if own:
                    own_occ.setdefault(w, set()).add(doc)
    total = total or 1
    gwords = _glossary_words(project)
    stop = {w for w, ks in occ.items()
            if len(ks) > max(total * cfg.stopword_df, STOPWORD_MIN)} - gwords
    if project.base_language.split("-")[0] == "en":
        # DF alone misses function words in small/medium corpora (live case:
        # hundreds of "possibly divergent" warnings for the/and/you/with/off);
        # a minimal built-in stop list for an English base plugs that hole
        stop |= _EN_STOPWORDS - gwords

    merged: dict[str, set[str]] = {}
    merged_own: dict[str, set[str]] = {}
    rep = None
    for w in sorted(w for w in occ if w not in stop):
        if rep is not None and len(rep) >= PREFIX and w.startswith(rep) \
                and w not in gwords:  # glossary word: own term, never swallowed
            merged[rep] |= occ[w]
            merged_own[rep] |= own_occ.get(w, set())
        else:
            rep = w
            merged[w] = set(occ[w])
            merged_own[w] = set(own_occ.get(w, set()))
    terms = {t: merged_own[t] for t, ks in merged.items()
             if (len(ks) >= cfg.min_occurrences or t in gwords) and merged_own[t]}

    bigrams: dict[str, set[str]] = {}
    bigrams_own: dict[str, set[str]] = {}
    for doc, e, own in _group_docs(project):
        toks = [t for t in _tokens(e.text) if t not in stop]
        for a, b in zip(toks, toks[1:]):
            bg = f"{a} {b}"
            bigrams.setdefault(bg, set()).add(doc)
            if own:
                bigrams_own.setdefault(bg, set()).add(doc)
    for t, ks in bigrams.items():
        if len(ks) >= cfg.min_bigram_occurrences and bigrams_own.get(t):
            terms.setdefault(t, bigrams_own[t])

    key_order = {k: i for i, k in enumerate(project.strings)}
    return {
        t: sorted(terms[t], key=key_order.__getitem__)
        for t in sorted(terms, key=lambda t: (-len(terms[t]), t))
    }


def get_terms(project: Project) -> dict[str, list[str]]:
    cache = getattr(project, "_terms_cache", None)
    if cache is None:
        cache = extract_terms(project)
        project._terms_cache = cache
    return cache


def text_terms(project: Project, text: str) -> set[str]:
    gwords = _glossary_words(project)
    toks = _tokens(text)

    def match(tok: str, rep: str) -> bool:
        # a glossary word only matches itself, not a shorter term it extends
        return _match(tok, rep) and (tok not in gwords or tok == rep)

    found = set()
    for term in get_terms(project):
        parts = term.split(" ")
        if len(parts) == 1:
            if any(match(t, parts[0]) for t in toks):
                found.add(term)
        else:
            for a, b in zip(toks, toks[1:]):
                if match(a, parts[0]) and match(b, parts[1]):
                    found.add(term)
                    break
    return found


# --- cheap heuristic detector (default verify level) ---------------------

def heuristic_divergences(project: Project, lang: str, locale: dict[str, dict]) -> list[str]:
    """Warning strings; empty for non-space-segmented target languages."""
    if lang.split("-")[0] in NON_SEGMENTED_PRIMARY:
        return []
    current = base_hashes(project)
    warnings = []
    for term, keys in get_terms(project).items():
        ok_keys = [k for k in keys
                   if locale.get(k, {}).get("text") and locale[k].get("src") == current.get(k)]
        if len(ok_keys) < 2:
            continue
        buckets: dict[str, set[str]] = {}
        for k in ok_keys:
            for tok in set(t for t in _tokens(locale[k]["text"]) if len(t) >= 3):
                buckets.setdefault(tok[:PREFIX], set()).add(k)
        best = max(buckets.values(), key=len, default=set())
        if len(best) < len(ok_keys):
            minority = [k for k in ok_keys if k not in best]
            warnings.append(
                f'term "{term}": possibly divergent translation in: {", ".join(minority)}')
    return warnings


# --- LLM term map, cached in terms/<lang>.json ------------

def term_map_path(project: Project, lang: str):
    return project.root / TERMS_DIR / f"{lang}.json"


def load_term_map(project: Project, lang: str) -> dict:
    p = term_map_path(project, lang)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"version": 1, "language": lang, "terms": {}, "accepted": {}}


def save_term_map(project: Project, lang: str, data: dict) -> None:
    p = term_map_path(project, lang)
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(p, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def prune_term_maps(project: Project) -> None:
    """Offline orphan cleanup of terms/<lang>.json (no LLM): drop terms that
    are no longer extracted (incl. their accepted variants) and keys that left
    a term's key list. Called from translate so that deleting a string from
    strings.toml cleans the caches even without --terms/--fix-terms runs."""
    terms_dir = project.root / TERMS_DIR
    if not terms_dir.is_dir():
        return
    current = get_terms(project)
    for p in sorted(terms_dir.glob("*.json")):
        lang = p.stem
        data = load_term_map(project, lang)
        changed = False
        for term in list(data.get("terms", {})):
            if term not in current:
                del data["terms"][term]
                data.get("accepted", {}).pop(term, None)
                changed = True
                continue
            tmap = data["terms"][term]
            for k in list(tmap):
                if k not in current[term]:
                    del tmap[k]
                    changed = True
        for term in list(data.get("accepted", {})):
            if term not in current:
                del data["accepted"][term]
                changed = True
        if changed:
            save_term_map(project, lang, data)


def build_term_map(project: Project, lang: str, locale: dict[str, dict],
                   client: OllamaClient) -> dict:
    """Incrementally (by src hash) ask the model which word in each accepted
    translation expresses each term."""
    from .prompts import load_template, render

    data = load_term_map(project, lang)
    current = base_hashes(project)
    template = load_template(project, "terms.txt")
    for term, keys in get_terms(project).items():
        tmap = data["terms"].setdefault(term, {})
        for k in list(tmap):
            if k not in keys:  # removed string, or no longer part of this term
                del tmap[k]
        todo = []
        for k in keys:
            e = locale.get(k)
            if not e or not e.get("text") or e.get("src") != current.get(k):
                continue
            src = text_hash(project.strings[k].text + "\x00" + e["text"])
            if tmap.get(k, {}).get("src") != src:
                todo.append((k, src))
        bs = project.ollama.batch_size
        for i in range(0, len(todo), bs):
            batch = todo[i:i + bs]
            entries_block = "\n".join(
                f"key: {k}\nbase: {project.strings[k].text}\ntranslation: {locale[k]['text']}\n"
                for k, _ in batch)
            prompt = render(template, term=term,
                            base_language=language_name(project.base_language),
                            target_language=language_name(lang),
                            entries_block=entries_block)
            parsed, _ = client.chat([{"role": "user", "content": prompt}],
                                    string_schema([k for k, _ in batch]))
            for k, src in batch:
                tmap[k] = {"src": src, "tr": str(parsed.get(k, "-")).strip()}
    save_term_map(project, lang, data)
    return data


# --- divergences --------------------------------------------------------

@dataclass
class Divergence:
    term: str
    variants: dict[str, list[str]]  # variant representative -> keys
    majority: str
    fraction: float


def _group_variants(values: dict[str, str], lang: str = "") -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for k, tr in sorted(values.items(), key=lambda kv: (kv[1].casefold(), kv[0])):
        t = tr.casefold()
        for rep in groups:
            if _variants_match(t, rep, lang):
                groups[rep].append(k)
                break
        else:
            groups[t] = [k]
    return groups


def divergences(data: dict) -> list[Divergence]:
    lang = data.get("language", "")
    out = []
    for term in sorted(data.get("terms", {})):
        tmap = data["terms"][term]
        values = {k: v["tr"] for k, v in tmap.items() if v.get("tr") and v["tr"] != "-"}
        groups = _group_variants(values, lang)
        if len(groups) < 2:
            continue
        if set(groups) <= set(data.get("accepted", {}).get(term, [])):
            continue  # legitimate polysemy, accepted earlier
        majority = max(groups, key=lambda g: (len(groups[g]), g))
        total = sum(len(v) for v in groups.values())
        out.append(Divergence(term, groups, majority, len(groups[majority]) / total))
    return out


# --- auto-fix ----------------------------------------------

WEAK_MAJORITY = 0.6


def _choose_variant(project: Project, lang: str, d: Divergence,
                    client: OllamaClient) -> str:
    from .prompts import load_template, render

    template = load_template(project, "choose_term.txt")
    variants_block = "\n".join(
        f'- "{v}" used in: ' + "; ".join(
            f'{k}: "{project.strings[k].text}" ({project.strings[k].context})'
            for k in ks)
        for v, ks in d.variants.items())
    prompt = render(template, term=d.term, target_language=language_name(lang),
                    app_context=project.full_context, variants_block=variants_block)
    schema = {"type": "object",
              "properties": {"variant": {"type": "string", "enum": list(d.variants)}},
              "required": ["variant"], "additionalProperties": False}
    parsed, _ = client.chat([{"role": "user", "content": prompt}], schema)
    choice = parsed.get("variant")
    return choice if choice in d.variants else d.majority


def fix_language(project: Project, lang: str, client: OllamaClient) -> dict:
    """Returns {"retranslated": n, "accepted": [terms], "failed": {key: errors}}."""
    from .engine import translate_batch

    locale = load_locale(project.root, lang)
    order = list(project.strings)
    data = build_term_map(project, lang, locale, client)
    divs = divergences(data)
    stats = {"retranslated": 0, "accepted": [], "failed": {}}
    tried: list[str] = []

    for d in divs:
        chosen = d.majority if d.fraction >= WEAK_MAJORITY \
            else _choose_variant(project, lang, d, client)
        minority = [k for v, ks in d.variants.items() if v != chosen for k in ks]
        if not minority:
            continue
        tried.append(d.term)
        examples = "; ".join(
            f'"{project.strings[k].text}" -> "{locale[k]["text"]}"'
            for k in d.variants.get(chosen, [])[:3])
        note = (f'TERMINOLOGY: in this app "{d.term}" is translated as "{chosen}" '
                f'(approved examples: {examples}). Use this translation unless '
                f"this string's context demands otherwise.")
        accepted, failed, _arb = translate_batch(project, client, lang, minority, locale,
                                           extra_note=note)
        for k, text in accepted.items():
            locale[k] = {"text": text, "src": text_hash(project.strings[k].text)}
        stats["retranslated"] += len(accepted)
        stats["failed"].update(failed)
        save_locale(project.root, lang, locale, order)

    if tried:
        # re-extract for the changed strings; a divergence that survives the fix
        # is meaningful polysemy -> remember and stop flagging (idempotence)
        data = build_term_map(project, lang, locale, client)
        still = {d.term: d for d in divergences(data) if d.term in tried}
        for term, d in still.items():
            acc = set(data["accepted"].get(term, [])) | set(d.variants)
            data["accepted"][term] = sorted(acc)
            stats["accepted"].append(term)
        if still:
            save_term_map(project, lang, data)
    return stats
