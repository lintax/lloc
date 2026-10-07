"""CLI: status, translate, build, verify, import."""
from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import ExitStack, nullcontext, suppress

from .adapters import get_adapter
from .config import ConfigError, Workspace, load
from .engine import TranslateInterrupted, manual_stale, plan, run_translate, run_translate_jobs
from .fsutil import parse_duration, parse_when, utc_now_iso
from .runlock import LockBusy, module_lock
from .store import base_hashes, classify, load_locale, load_lock, save_lock


# Flags are accepted in any position (before or after the subcommand): every
# group below is added both to the main parser and to the relevant subparsers.
# default=SUPPRESS everywhere, otherwise subparser defaults would overwrite
# values already parsed by the main parser; real defaults live in DEFAULTS.
_S = argparse.SUPPRESS

DEFAULTS = {
    "dir": ".", "lang": None, "verbose": False,
    "force": False, "limit": None, "dry_run": False, "no_build": False, "model": None,
    "assume_stale": False, "overwrite": False, "bootstrap_strings": False,
    "fix_terms": False, "terms": False, "retranslate_model": None,
    "last": False, "since": "24h", "force_set": False,
    "semantic": False, "fix_semantic": False,
}


def _global_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dir", default=_S, help="directory containing l10n.toml (default: .)")
    p.add_argument("--lang", default=_S, help="comma-separated language filter, e.g. ru,de")
    p.add_argument("--verbose", action="store_true", default=_S)


def _translate_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--force", action="store_true", default=_S,
                   help="retranslate everything for selected languages")
    p.add_argument("--limit", type=int, default=_S, help="max strings per language per run")
    p.add_argument("--dry-run", action="store_true", default=_S, help="show plan, no LLM calls")
    p.add_argument("--no-build", action="store_true", default=_S)
    p.add_argument("--model", default=_S,
                   help="override the Ollama model from l10n.toml (for A/B runs)")
    p.add_argument("--fix-terms", action="store_true", default=_S,
                   help="fix terminology divergences via re-translation")
    p.add_argument("--fix-semantic", action="store_true", default=_S,
                   help="re-translate suspects found by `verify --semantic`, telling "
                        "the model what was wrong")
    p.add_argument("--retranslate-model", default=_S, metavar="PATTERN",
                   help="re-translate strings whose stored `model` matches this "
                        "fnmatch pattern (e.g. 'gemma3*', 'imported', 'manual')")


def _import_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--assume-stale", action="store_true", default=_S)
    p.add_argument("--overwrite", action="store_true", default=_S)
    p.add_argument("--bootstrap-strings", action="store_true", default=_S,
                   help="generate strings.toml from the base platform file")


def _verify_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--terms", action="store_true", default=_S,
                   help="build the LLM term map and report divergences")
    p.add_argument("--semantic", action="store_true", default=_S,
                   help="back-translation QA: judge meaning drift per string "
                        "(expensive, two LLM passes; run selectively via --lang)")


def _report_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--last", action="store_true", default=_S,
                   help="show recently translated strings (the default and only mode)")
    p.add_argument("--since", default=_S, metavar="DURATION",
                   help="time window for --last: 90m / 24h / 7d (default 24h)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lloc", description="LLM localization tool")
    _global_flags(p)
    _translate_flags(p)   # tolerate command flags before the subcommand too
    _import_flags(p)
    _verify_flags(p)
    _report_flags(p)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("status", help="show ok/missing/stale/orphan per language")
    _global_flags(s)

    t = sub.add_parser("translate", help="translate delta via Ollama, then build")
    _global_flags(t)
    _translate_flags(t)

    b = sub.add_parser("build", help="generate platform files from locales/ via adapter")
    _global_flags(b)

    i = sub.add_parser("import", help="import existing platform translations into locales/")
    _global_flags(i)
    _import_flags(i)

    v = sub.add_parser("verify", help="validate existing translations (LLM only with --terms)")
    _global_flags(v)
    _verify_flags(v)
    v.add_argument("--model", default=_S,
                   help="override the Ollama model for --terms/--semantic "
                        "(audit with a different judge)")

    r = sub.add_parser("report", help="review recently translated strings")
    _global_flags(r)
    _report_flags(r)

    st = sub.add_parser("set", help='manual fix: lloc set <key> --lang de "text"')
    st.add_argument("key", help="string key from strings.toml")
    st.add_argument("text", help="the corrected translation")
    st.add_argument("--force-set", action="store_true", default=_S,
                    help="accept the text even if placeholder/format validation fails")
    _global_flags(st)
    return p


def _langs(project, args) -> list[str]:
    if not args.lang:
        return project.languages
    langs = [x.strip() for x in args.lang.split(",") if x.strip()]
    unknown = [x for x in langs if x not in project.languages]
    if unknown:
        raise ConfigError(f"--lang: not in [app].languages: {', '.join(unknown)}")
    return langs


# --- commands -----------------------------------------------------------

def cmd_status(project, args) -> int:
    langs = _langs(project, args)
    total = len(project.strings)
    print(f"{'lang':<10} {'ok':>5} {'missing':>8} {'stale':>6} {'orphan':>7}")
    dirty = False
    for lang in langs:
        st = classify(project, lang)
        if not st.clean:
            dirty = True
        print(f"{lang:<10} {len(st.ok):>5} {len(st.missing):>8} {len(st.stale):>6} {len(st.orphan):>7}")
    print(f"\n{total} base strings, {len(langs)} languages")
    return 1 if dirty else 0


def _print_plan(work: dict[str, list[str]], total: int, mod: str = "") -> None:
    """Dry-run plan, inverted: languages grouped by identical pending keysets
    (map<keyset, languages>), largest keyset first. A real translate run
    batches per language and may differ in order — the plan answers
    "what changes", not "in which order"."""
    groups: dict[tuple[str, ...], list[str]] = {}
    for lang, keys in work.items():
        if keys:
            groups.setdefault(tuple(keys), []).append(lang)
    if not groups:
        print(f"{mod}nothing to translate")
        return
    for keys, langs in sorted(groups.items(), key=lambda kv: (-len(kv[0]), kv[1])):
        if len(keys) == total:
            what = f"all {total} keys"
        else:
            shown = ", ".join(keys[:10]) + (", …" if len(keys) > 10 else "")
            what = f"{len(keys)} key{'s' if len(keys) != 1 else ''} ({shown})"
        if len(langs) == 1:
            who = langs[0]
        else:
            shown = ", ".join(langs[:8]) + (", …" if len(langs) > 8 else "")
            who = f"{len(langs)} languages ({shown})"
        print(f"{mod}{what} → {who}")


def cmd_translate(project, args) -> int:
    if args.model:  # explicit A/B override beats [models] per-language pins
        project.ollama.model = args.model
        project.models = {}
    langs = _langs(project, args)
    work = plan(project, langs, args.force, args.limit, args.retranslate_model)
    todo = {k: v for k, v in work.items() if v}
    if args.dry_run:
        _print_plan(work, total=len(project.strings))
        print(f"\nplan: {sum(len(v) for v in work.values())} translations, no requests made")
        return 0
    if not todo:
        print("nothing to translate")
    stale_manual = manual_stale(project, langs)  # snapshot before the run
    started = time.monotonic()
    results = run_translate(project, langs, force=args.force, limit=args.limit,
                            retranslate_model=args.retranslate_model)
    elapsed = time.monotonic() - started

    # lock: snapshot of current base hashes (only when no fatal errors);
    # terms/semantic caches: offline orphan cleanup (deleted strings leave everywhere)
    fatal = [r for r in results if r.error]
    if not fatal:
        save_lock(project.root, base_hashes(project))
        from .terms import prune_term_maps
        from .semantic import prune_semantic_caches
        prune_term_maps(project)
        prune_semantic_caches(project)

    translated = sum(r.translated for r in results)
    failed = {r.lang: r.failed for r in results if r.failed}
    print(f"\ntranslated {translated} strings across {len(langs)} languages "
          f"in {elapsed:.1f}s (model {project.ollama.model})")
    for r in results:
        if r.error:
            print(f"[{r.lang}] FATAL: {r.error}")
    for lang, fails in failed.items():
        for key, errors in fails.items():
            print(f"[{lang}] FAILED {key}: {'; '.join(errors)}")
    _report_stale_manual(stale_manual, results)
    _report_weak([("", r) for r in results])

    rc = 0
    if fatal or failed:
        rc = 1

    if args.fix_terms and not fatal:
        rc = max(rc, _run_fix_terms(project, langs))
    if args.fix_semantic and not fatal:
        rc = max(rc, _run_fix_semantic(project, langs))
    if _phased_semantic(project) and not fatal:
        # separate judge model: check phase (one model swap), then fix phase
        sus_langs = _phased_semantic_check(project, langs)
        if sus_langs:
            rc = max(rc, _run_fix_semantic(project, sus_langs))

    if not args.no_build and not fatal:
        rc2 = cmd_build(project, args)
        rc = rc or rc2
    return rc


def _report_weak(tagged_results) -> None:
    """End-of-run summary of weak languages (per-language hints were already
    printed by the engine): mandatory, loud, never blocking."""
    weak = [f"{mod}{r.lang}" if mod else r.lang
            for mod, r in tagged_results if r.weak]
    if weak:
        print(f"\nWARNING: model looks weak in: {', '.join(weak)} — "
              "see the per-language retranslation hints above")


def _report_stale_manual(stale_manual: dict, results, mod: str = "") -> None:
    """The base text changed under these manual fixes — they went back to the
    LLM. Flag them: the human curation is gone and may want a fresh look."""
    by_lang = {r.lang: r for r in results}
    for lang, keys in stale_manual.items():
        r = by_lang.get(lang)
        if r is None or r.error:
            continue
        done = [k for k in keys if k not in r.failed]
        if done:
            print(f"{mod}[{lang}] {len(done)} manual translation(s) were stale "
                  f"and retranslated: {', '.join(done)}")


def _phased_semantic(project) -> bool:
    return project.semantic_phased


def _phased_semantic_check(project, langs) -> list[str]:
    """Phase 2 of the separate-judge-model flow: after the whole run is
    translated by the main model, swap to [validation].semantic_model ONCE and
    QA everything fresh (cache delta — verified pairs are skipped). Returns
    languages that have suspects; phase 3 (_run_fix_semantic) fixes them with
    the main model — two model swaps per run total, never per batch."""
    from dataclasses import replace as dc_replace

    from .ollama import OllamaClient
    from .semantic import build_semantic, suspects

    judge = OllamaClient(dc_replace(project.ollama, model=project.semantic_model))
    mod = f"{project.module}:" if project.module else ""
    out = []
    for lang in langs:
        data = build_semantic(project, lang, load_locale(project.root, lang), judge)
        n = len([1 for k, _, _ in suspects(data) if k in project.strings])
        if n:
            print(f"[{mod}{lang}] semantic (judge {project.semantic_model}): "
                  f"{n} suspect(s)")
            out.append(lang)
    if not out:
        print(f"{mod or ''}semantic (judge {project.semantic_model}): no suspects")
    return out


def _run_fix_semantic(project, langs) -> int:
    """Re-translate suspects from the semantic cache (verify --semantic must
    have run first), telling the model what the judge found wrong. Fixed keys
    are dropped from the cache — the next --semantic run re-checks them."""
    from .engine import translate_batch
    from .fsutil import utc_now_iso
    from .ollama import OllamaClient
    from .semantic import clear_keys, load_semantic, suspects
    from .store import save_locale

    if project.ensemble:
        # systemic path: the whole backlog (unverified/suspect/arbitrated,
        # minus resolved) is re-resolved by the ensemble phases — this is
        # the EXPLICIT sweep; a plain translate touches only its delta
        print("ensemble mode: re-resolving the semantic backlog via the "
              "ensemble phases")
        results = run_translate(project, langs, include_backlog=True)
        return 1 if any(r.error or r.failed for r in results) else 0

    client = OllamaClient(project.ollama)
    mod = f"{project.module}:" if project.module else ""
    rc = 0
    for lang in langs:
        found = suspects(load_semantic(project, lang))
        found = [(k, r, b) for k, r, b in found if k in project.strings]
        if not found:
            print(f"[{mod}{lang}] fix-semantic: no suspects")
            continue
        locale = load_locale(project.root, lang)
        note = "Previous translations of these strings drifted in meaning:\n" + "\n".join(
            f'- {k}: {r or "meaning drift"} (previous text literally meant: "{b}")'
            for k, r, b in found) + "\nTranslate the MEANING of the source text correctly."
        keys = [k for k, _, _ in found]
        accepted, failed, _arb = translate_batch(project, client, lang, keys, locale,
                                                 extra_note=note)
        for key, text in accepted.items():
            locale[key] = {"text": text, "src": base_hashes(project)[key],
                           "model": project.ollama.model, "translated_at": utc_now_iso()}
        if accepted:
            save_locale(project.root, lang, locale, list(project.strings))
            clear_keys(project, lang, list(accepted))
        print(f"[{mod}{lang}] fix-semantic: retranslated {len(accepted)}"
              + (f", failed {len(failed)}" if failed else ""))
        for key, errors in failed.items():
            print(f"[{mod}{lang}] fix-semantic FAILED {key}: {'; '.join(errors)}")
            rc = 1
    return rc


def _run_fix_terms(project, langs) -> int:
    from .ollama import OllamaClient
    from .terms import fix_language

    client = OllamaClient(project.ollama)
    mod = f"{project.module}:" if project.module else ""
    rc = 0
    for lang in langs:
        stats = fix_language(project, lang, client)
        if stats["retranslated"] or stats["accepted"] or stats["failed"]:
            acc = f", accepted as polysemy: {', '.join(stats['accepted'])}" \
                if stats["accepted"] else ""
            print(f"[{mod}{lang}] fix-terms: retranslated {stats['retranslated']}{acc}")
            for key, errors in stats["failed"].items():
                print(f"[{mod}{lang}] fix-terms FAILED {key}: {'; '.join(errors)}")
                rc = 1
        else:
            print(f"[{mod}{lang}] fix-terms: no divergences")
    return rc


def cmd_build(project, args) -> int:
    adapter = get_adapter(project)
    locales = {lang: load_locale(project.root, lang) for lang in project.languages}
    written = adapter.export(locales)
    mod = f"[{project.module}] " if project.module else ""
    print(f"{mod}build: wrote {len(written)} files via {adapter.name}")
    if args.verbose:
        for w in written:
            print(f"  {w}")
    return 0


def cmd_verify(project, args) -> int:
    from .validate import check_translation

    if args.model:  # audit with a different judge model
        project.ollama.model = args.model
    adapter = get_adapter(project)
    langs = _langs(project, args)
    problems = 0

    lock = load_lock(project.root)
    current = base_hashes(project)
    if lock and lock != current:
        changed = [k for k in current if lock.get(k) != current[k]]
        removed = [k for k in lock if k not in current]
        print(f"lock drift: {len(changed)} changed/new, {len(removed)} removed "
              f"(run translate to refresh)")

    from .terms import build_term_map, divergences, heuristic_divergences

    term_client = None
    if args.terms:
        from .ollama import OllamaClient
        term_client = OllamaClient(project.ollama)

    # semantic audit: languages are independent jobs (per-language cache
    # files, per-job clients) -> run them over the configured worker pool
    # up front; the sequential loop below only prints the results
    sem_data: dict[str, dict] = {}
    if args.semantic:
        import concurrent.futures

        from .ollama import OllamaClient
        from .semantic import build_semantic

        def _sem_job(lang: str) -> None:
            client = OllamaClient(project.ollama)
            sem_data[lang] = build_semantic(project, lang,
                                            load_locale(project.root, lang), client)

        workers = max(1, project.ollama.workers)
        if workers == 1 or len(langs) == 1:
            for lang in langs:
                _sem_job(lang)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                for fut in [pool.submit(_sem_job, lang) for lang in langs]:
                    fut.result()

    warn_count = 0
    # duplicate keys across modules of one group: Android merges resources
    # silently, so this is a warning, not an error
    for sib in project.siblings:
        dup = sorted(set(project.strings) & set(sib.strings))
        if dup:
            warn_count += 1
            print(f'warning: duplicate keys with module "{sib.module}" '
                  f"(group {project.group}): {', '.join(dup)}")

    from .langnames import expected_scripts
    from .validate import script_mix

    # script-mix allowlist: [validation].latin_allowlist + glossary terms (brands)
    allow = frozenset({t.casefold() for t in project.latin_allowlist}
                      | {w.casefold() for term in project.glossary for w in term.split()})

    for lang in langs:
        st = classify(project, lang)
        locale = load_locale(project.root, lang)
        expected = expected_scripts(lang)
        for key in st.ok:
            base = project.strings[key].text
            text = locale[key]["text"]
            errs = check_translation(base, text, project.length_ratio) + adapter.check(text)
            for e in errs:
                problems += 1
                print(f"[{lang}] {key}: {e}")
            for w in script_mix(text, expected, project.latin_token_max, allow):
                warn_count += 1
                print(f"[{lang}] warning: {key}: {w}")
        if not st.clean:
            print(f"[{lang}] pending work: {len(st.missing)} missing, "
                  f"{len(st.stale)} stale, {len(st.orphan)} orphan")

        # terminology: warnings only, do not affect the exit code;
        # aggregated per language — full list with --verbose
        TERM_WARN_CAP = 8
        if term_client is not None:
            lines = []
            for d in divergences(data := build_term_map(project, lang, locale,
                                                        term_client)):
                variants = "; ".join(f'"{v}" ({", ".join(ks)})'
                                     for v, ks in d.variants.items())
                lines.append(f'[{lang}] term "{d.term}" diverges: {variants} '
                             f'-> majority "{d.majority}" ({d.fraction:.0%})')
        else:
            lines = [f"[{lang}] warning: {w}"
                     for w in heuristic_divergences(project, lang, locale)]
        warn_count += len(lines)
        shown = lines if args.verbose else lines[:TERM_WARN_CAP]
        for line in shown:
            print(line)
        if len(lines) > len(shown):
            print(f"[{lang}] ... and {len(lines) - len(shown)} more term "
                  f"warning(s) (rerun with --verbose for the full list)")

        # semantic QA: back-translation + judge, cached, warnings only
        # (the LLM work already happened in parallel above)
        if args.semantic:
            from .semantic import suspects
            for key, reason, back in suspects(sem_data[lang]):
                warn_count += 1
                print(f'[{lang}] SUSPECT {key}: {reason} | back: "{back}"')

    if warn_count:
        print(f"verify: {warn_count} warning(s)")
    print(f"verify: {problems} problem(s)" if problems else "verify: OK")
    return 1 if problems else 0


def cmd_import(project, args) -> int:
    if not args.bootstrap_strings:
        return _import_translations(project, args)
    from .bootstrap import write_strings_toml
    from .config import STRINGS_NAME

    dest = project.root / STRINGS_NAME
    if dest.exists() and not args.overwrite:
        print(f"error: {dest.name} already exists (use --overwrite to replace)", file=sys.stderr)
        return 2
    adapter = get_adapter(project)
    entries, skipped = adapter.bootstrap_source()
    if not entries:
        print("error: no translatable strings found in the base file", file=sys.stderr)
        return 2
    write_strings_toml(dest, entries)
    print(f"bootstrap: wrote {dest.name} with {len(entries)} strings "
          f"(skipped {skipped} special/non-translatable)")
    return 0


def _import_translations(project, args) -> int:
    """Import existing platform translations into locales/ + lock.

    Reads every per-language platform file via the adapter, stores translations
    for keys known to strings.toml with src = the current base hash (they are
    assumed up to date; --assume-stale marks them for retranslation instead).
    Existing locale entries win unless --overwrite. Idempotent: a rerun
    imports nothing new and rewrites nothing."""
    from .store import save_locale

    adapter = get_adapter(project)
    langs = _langs(project, args)
    imported = adapter.import_()
    current = base_hashes(project)
    order = list(project.strings)
    mod = f"[{project.module}] " if project.module else ""
    totals = {"imported": 0, "kept": 0, "skipped": 0}
    no_file = []
    touched = False
    for lang in langs:
        data = imported.get(lang)
        if not data:
            no_file.append(lang)
            continue
        locale = load_locale(project.root, lang)
        new = kept = skipped = 0
        for key, text in data.items():
            if key not in current or not text.strip():
                skipped += 1  # unknown to strings.toml, or empty
                continue
            if locale.get(key, {}).get("text") and not args.overwrite:
                kept += 1
                continue
            src = "imported-stale" if args.assume_stale else current[key]
            locale[key] = {"text": text, "src": src,
                           "model": "imported", "translated_at": utc_now_iso()}
            new += 1
        if new:
            save_locale(project.root, lang, locale, order)
            touched = True
        totals["imported"] += new
        totals["kept"] += kept
        totals["skipped"] += skipped
        if new or kept or skipped or args.verbose:
            print(f"{mod}[{lang}] imported {new}, kept {kept}, skipped {skipped}")
    if touched:
        save_lock(project.root, current)  # snapshot of the base, as after translate
    if no_file:
        print(f"{mod}no platform file for: {', '.join(no_file)}")
    print(f"{mod}import: {totals['imported']} imported, {totals['kept']} kept, "
          f"{totals['skipped']} skipped (unknown/empty)")
    return 0


def cmd_set(project, args) -> int:
    """Manual fix: write one translation into locales/ with model="manual"
    (protected from --force / --retranslate-model), then incrementally build
    that language. The canonical answer to "this translation is bad"."""
    from .store import save_locale
    from .validate import check_translation

    if args.key not in project.strings:
        raise ConfigError(f"set: unknown key {args.key!r} (not in strings.toml)")
    if not args.lang:
        raise ConfigError('set: --lang is required, e.g. set greeting --lang de "Hallo"')
    lang = args.lang.strip()
    if "," in lang:
        raise ConfigError("set: exactly one language per call")
    if lang not in project.languages:
        raise ConfigError(f"set: language {lang!r} not in [app].languages")

    base = project.strings[args.key].text
    adapter = get_adapter(project)
    errors = check_translation(base, args.text, project.length_ratio) \
        + adapter.check(args.text)
    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        if not args.force_set:
            print("use --force-set to accept the text anyway", file=sys.stderr)
            return 1
        print("accepted despite validation errors (--force-set)", file=sys.stderr)

    locale = load_locale(project.root, lang)
    locale[args.key] = {"text": args.text, "src": base_hashes(project)[args.key],
                        "model": "manual", "translated_at": utc_now_iso()}
    save_locale(project.root, lang, locale, list(project.strings))
    written = adapter.export({lang: locale}, langs=[lang])
    mod = f"[{project.module}] " if project.module else ""
    print(f'{mod}set [{lang}] {args.key} = "{args.text}" (manual), '
          f"rebuilt {len(written)} file(s)")
    return 0


def cmd_report(project, args) -> int:
    """Review pass: strings translated within the --since window (default 24h),
    based on the `translated_at` stamp of locale entries. No LLM, no writes."""
    from datetime import datetime, timezone

    try:
        window = parse_duration(args.since)
    except ValueError as e:
        raise ConfigError(str(e)) from e
    from .semantic import load_semantic

    cutoff = datetime.now(timezone.utc) - window
    mod = f"[{project.module}] " if project.module else ""
    total = arb_total = 0
    for lang in _langs(project, args):
        locale = load_locale(project.root, lang)
        rows = [(key, e) for key in project.strings
                if (e := locale.get(key)) and (when := parse_when(e.get("translated_at", "")))
                and when >= cutoff]
        if not rows:
            continue
        sem = load_semantic(project, lang)["entries"]
        arb = [(k, e) for k, e in rows if sem.get(k, {}).get("verdict") == "arbitrated"]
        plain = [(k, e) for k, e in rows if (k, e) not in arb]
        total += len(rows)
        print(f"{mod}[{lang}] {len(rows)} translated in the last {args.since}:")
        for key, e in plain:
            print(f'  {key}: "{e["text"]}" ({e.get("model", "?")}, {e["translated_at"]})')
        if arb:  # auto-resolved conflicts: reviewable, never blocking
            arb_total += len(arb)
            print(f"{mod}[{lang}] arbitrated (auto-resolved semantic disputes, "
                  f"review optional — override via `lloc set`):")
            for key, e in arb:
                reason = sem.get(key, {}).get("reason", "")
                print(f'  {key}: "{e["text"]}"' + (f" — {reason}" if reason else ""))
            from .engine import WEAK_ARBITRATION_RATIO, WEAK_MIN_SAMPLE
            if len(rows) >= WEAK_MIN_SAMPLE \
                    and len(arb) / len(rows) >= WEAK_ARBITRATION_RATIO:
                print(f"{mod}[{lang}] WARNING: model looks weak in {lang} "
                      f"({round(100 * len(arb) / len(rows))}% arbitrated); "
                      f"consider: translate --force --lang {lang} "
                      f"--model qwen3.5:9b")
    print(f"{mod}report: {total} strings in the last {args.since}"
          + (f", {arb_total} arbitrated" if arb_total else ""))
    return 0


# --- run lock (one mutating run per module) -------------------------------

def _needs_lock(args) -> bool:
    """Mutating commands hold the per-module run lock; `verify --terms` and
    `verify --semantic` write caches, so they count too."""
    return args.command in ("translate", "build", "import", "set") \
        or (args.command == "verify" and (args.terms or args.semantic))


def _lock_ctx(project, args):
    if not _needs_lock(args):
        return nullcontext()
    return module_lock(project.root, label=project.module or str(project.root))


# --- workspace mode ----------------------------------------

def _member_langs(member, args) -> list[str]:
    if not args.lang:
        return member.languages
    requested = [x.strip() for x in args.lang.split(",") if x.strip()]
    return [x for x in requested if x in member.languages]


def _check_workspace_langs(ws: Workspace, args) -> None:
    if not args.lang:
        return
    requested = [x.strip() for x in args.lang.split(",") if x.strip()]
    known = {lang for m in ws.members for lang in m.languages}
    unknown = [x for x in requested if x not in known]
    if unknown:
        raise ConfigError(f"--lang: not in any member's languages: {', '.join(unknown)}")


def cmd_translate_workspace(ws: Workspace, args) -> int:
    from .terms import prune_term_maps

    module_work = []
    jobs = []
    for m in ws.members:
        if args.model:  # explicit A/B override beats [models] per-language pins
            m.ollama.model = args.model
            m.models = {}
        work = plan(m, _member_langs(m, args), args.force, args.limit,
                    args.retranslate_model)
        module_work.append((m, work))
        jobs += [(m, lang, keys) for lang, keys in work.items() if keys]

    if args.dry_run:
        for m, work in module_work:
            _print_plan(work, total=len(m.strings), mod=f"[{m.module}] ")
        total = sum(len(k) for _, w in module_work for k in w.values())
        print(f"\nplan: {total} translations, no requests made")
        return 0

    if not jobs:
        print("nothing to translate")
    stale_by_module = {m.module: manual_stale(m, [lang for lang in work])
                       for m, work in module_work}  # snapshot before the run
    started = time.monotonic()
    results = run_translate_jobs(jobs, ws.workers)  # one shared pool for the run
    elapsed = time.monotonic() - started

    rc = 0
    translated = sum(r.translated for _, r in results)
    print(f"\ntranslated {translated} strings across {len(ws.members)} modules "
          f"in {elapsed:.1f}s")
    fatal_modules = set()
    for m, work in module_work:
        rs = [r for p, r in results if p is m]
        if any(r.error for r in rs):
            fatal_modules.add(m.module)
            rc = 1
        else:
            save_lock(m.root, base_hashes(m))  # lock is per module, as before
            prune_term_maps(m)
            from .semantic import prune_semantic_caches
            prune_semantic_caches(m)
        for r in rs:
            if r.error:
                print(f"[{m.module}:{r.lang}] FATAL: {r.error}")
            for key, errors in r.failed.items():
                print(f"[{m.module}:{r.lang}] FAILED {key}: {'; '.join(errors)}")
                rc = 1
        _report_stale_manual(stale_by_module.get(m.module, {}), rs, mod=f"[{m.module}] ")
    _report_weak([(f"{m.module}:", r) for m, r in results])

    if args.fix_terms:
        for m, work in module_work:
            if m.module not in fatal_modules:
                rc = max(rc, _run_fix_terms(m, [lang for lang in work]))
    if args.fix_semantic:
        for m, work in module_work:
            if m.module not in fatal_modules:
                rc = max(rc, _run_fix_semantic(m, [lang for lang in work]))

    # separate judge model: check ALL modules first (one model swap), then fix
    # suspects per module with the main model (one swap back)
    phased = [(m, [lang for lang in work]) for m, work in module_work
              if m.module not in fatal_modules and _phased_semantic(m)]
    to_fix = [(m, _phased_semantic_check(m, langs_)) for m, langs_ in phased]
    for m, sus_langs in to_fix:
        if sus_langs:
            rc = max(rc, _run_fix_semantic(m, sus_langs))

    if not args.no_build:
        for m, _ in module_work:
            if m.module not in fatal_modules:
                rc = max(rc, cmd_build(m, args))
    return rc


def run_workspace(ws: Workspace, args) -> int:
    if args.command in ("import", "set"):
        print(f"{args.command} runs per module in a workspace: use --dir <member>",
              file=sys.stderr)
        return 2
    _check_workspace_langs(ws, args)
    if args.command == "translate":
        with ExitStack() as stack:  # one run touches every member: lock them all
            for m in ws.members:
                stack.enter_context(_lock_ctx(m, args))
            return cmd_translate_workspace(ws, args)
    rc = 0
    for m in ws.members:
        sub = argparse.Namespace(**vars(args))
        langs = _member_langs(m, args)
        if not langs:
            continue  # --lang selected languages this member does not have
        sub.lang = ",".join(langs) if args.lang else None
        print(f"\n=== {m.module} (group: {m.group}) ===")
        with _lock_ctx(m, args):
            rc = max(rc, COMMANDS[args.command](m, sub))
    return rc


COMMANDS = {
    "status": cmd_status,
    "translate": cmd_translate,
    "build": cmd_build,
    "verify": cmd_verify,
    "import": cmd_import,
    "report": cmd_report,
    "set": cmd_set,
}


def main(argv: list[str] | None = None) -> int:
    if os.name == "nt":  # pragma: no cover - Windows console may not be UTF-8
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    for key, value in DEFAULTS.items():  # fill flags suppressed in both positions
        if not hasattr(args, key):
            setattr(args, key, value)
    try:
        # bootstrap creates strings.toml, so its absence must not be an error there
        require = not (args.command == "import" and args.bootstrap_strings)
        loaded = load(args.dir, require_strings=require)
        if isinstance(loaded, Workspace):
            return run_workspace(loaded, args)
        with _lock_ctx(loaded, args):
            return COMMANDS[args.command](loaded, args)
    except (TranslateInterrupted, KeyboardInterrupt):
        # the lock is already released: the exception unwound _lock_ctx, and
        # the OS drops the flock on process death anyway
        return _interrupted()
    except LockBusy as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _interrupted() -> int:
    """Ctrl+C: report, flush and leave NOW — worker threads may be stuck in
    long HTTP requests and must not keep the process alive (they are killed
    by os._exit; completed batches are already saved). A second Ctrl+C during
    this handler exits just as immediately via the signal override."""
    import signal
    with suppress(Exception):
        signal.signal(signal.SIGINT, lambda *_: os._exit(130))
    print("\ninterrupted — progress saved, rerun to continue", file=sys.stderr)
    with suppress(Exception):
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(130)
    return 130  # unreachable; keeps the mocked-os._exit test path sane


if __name__ == "__main__":
    sys.exit(main())
