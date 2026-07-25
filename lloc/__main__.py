"""CLI: status, translate, build, verify, import. SPEC.md §5, §14."""
from __future__ import annotations

import argparse
import sys
import time

from .adapters import get_adapter
from .config import ConfigError, Workspace, load
from .engine import plan, run_translate, run_translate_jobs
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
    "fix_terms": False, "terms": False,
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
                   help="fix terminology divergences via re-translation (SPEC.md §13.4)")


def _import_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--assume-stale", action="store_true", default=_S)
    p.add_argument("--overwrite", action="store_true", default=_S)
    p.add_argument("--bootstrap-strings", action="store_true", default=_S,
                   help="generate strings.toml from the base platform file")


def _verify_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--terms", action="store_true", default=_S,
                   help="build the LLM term map and report divergences (SPEC.md §13.3)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lloc", description="LLM localization tool (see SPEC.md)")
    _global_flags(p)
    _translate_flags(p)   # tolerate command flags before the subcommand too
    _import_flags(p)
    _verify_flags(p)
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


def cmd_translate(project, args) -> int:
    if args.model:
        project.ollama.model = args.model
    langs = _langs(project, args)
    work = plan(project, langs, args.force, args.limit)
    todo = {k: v for k, v in work.items() if v}
    if args.dry_run:
        for lang, keys in work.items():
            print(f"[{lang}] {len(keys)} to translate" + (f": {', '.join(keys[:10])}"
                  + ("…" if len(keys) > 10 else "") if keys else ""))
        print(f"\nplan: {sum(len(v) for v in work.values())} translations, no requests made")
        return 0
    if not todo:
        print("nothing to translate")
    started = time.monotonic()
    results = run_translate(project, langs, force=args.force, limit=args.limit)
    elapsed = time.monotonic() - started

    # lock: snapshot of current base hashes (only when no fatal errors);
    # terms caches: offline orphan cleanup (deleted strings leave everywhere)
    fatal = [r for r in results if r.error]
    if not fatal:
        save_lock(project.root, base_hashes(project))
        from .terms import prune_term_maps
        prune_term_maps(project)

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

    rc = 0
    if fatal or failed:
        rc = 1

    if args.fix_terms and not fatal:
        rc = max(rc, _run_fix_terms(project, langs))

    if not args.no_build and not fatal:
        rc2 = cmd_build(project, args)
        rc = rc or rc2
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

    warn_count = 0
    # duplicate keys across modules of one group: Android merges resources
    # silently, so this is a warning, not an error (SPEC.md §14)
    for sib in project.siblings:
        dup = sorted(set(project.strings) & set(sib.strings))
        if dup:
            warn_count += 1
            print(f'warning: duplicate keys with module "{sib.module}" '
                  f"(group {project.group}): {', '.join(dup)}")

    for lang in langs:
        st = classify(project, lang)
        locale = load_locale(project.root, lang)
        for key in st.ok:
            base = project.strings[key].text
            text = locale[key]["text"]
            errs = check_translation(base, text, project.length_ratio) + adapter.check(text)
            for e in errs:
                problems += 1
                print(f"[{lang}] {key}: {e}")
        if not st.clean:
            print(f"[{lang}] pending work: {len(st.missing)} missing, "
                  f"{len(st.stale)} stale, {len(st.orphan)} orphan")

        # terminology: warnings only, do not affect the exit code (SPEC.md §13.3)
        if term_client is not None:
            data = build_term_map(project, lang, locale, term_client)
            for d in divergences(data):
                warn_count += 1
                variants = "; ".join(f'"{v}" ({", ".join(ks)})' for v, ks in d.variants.items())
                print(f'[{lang}] term "{d.term}" diverges: {variants} '
                      f'-> majority "{d.majority}" ({d.fraction:.0%})')
        else:
            for w in heuristic_divergences(project, lang, locale):
                warn_count += 1
                print(f"[{lang}] warning: {w}")

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
    """Import existing platform translations into locales/ + lock. SPEC.md §5, §9.

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
            locale[key] = {"text": text, "src": src}
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


# --- workspace mode (SPEC.md §14) ----------------------------------------

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
        if args.model:
            m.ollama.model = args.model
        work = plan(m, _member_langs(m, args), args.force, args.limit)
        module_work.append((m, work))
        jobs += [(m, lang, keys) for lang, keys in work.items() if keys]

    if args.dry_run:
        for m, work in module_work:
            for lang, keys in work.items():
                print(f"[{m.module}:{lang}] {len(keys)} to translate"
                      + (f": {', '.join(keys[:10])}" + ("…" if len(keys) > 10 else "")
                         if keys else ""))
        total = sum(len(k) for _, w in module_work for k in w.values())
        print(f"\nplan: {total} translations, no requests made")
        return 0

    if not jobs:
        print("nothing to translate")
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
        for r in rs:
            if r.error:
                print(f"[{m.module}:{r.lang}] FATAL: {r.error}")
            for key, errors in r.failed.items():
                print(f"[{m.module}:{r.lang}] FAILED {key}: {'; '.join(errors)}")
                rc = 1

    if args.fix_terms:
        for m, work in module_work:
            if m.module not in fatal_modules:
                rc = max(rc, _run_fix_terms(m, [lang for lang in work]))

    if not args.no_build:
        for m, _ in module_work:
            if m.module not in fatal_modules:
                rc = max(rc, cmd_build(m, args))
    return rc


def run_workspace(ws: Workspace, args) -> int:
    if args.command == "import":
        print("import runs per module in a workspace: use --dir <member>", file=sys.stderr)
        return 2
    _check_workspace_langs(ws, args)
    if args.command == "translate":
        return cmd_translate_workspace(ws, args)
    rc = 0
    for m in ws.members:
        sub = argparse.Namespace(**vars(args))
        langs = _member_langs(m, args)
        if not langs:
            continue  # --lang selected languages this member does not have
        sub.lang = ",".join(langs) if args.lang else None
        print(f"\n=== {m.module} (group: {m.group}) ===")
        rc = max(rc, COMMANDS[args.command](m, sub))
    return rc


COMMANDS = {
    "status": cmd_status,
    "translate": cmd_translate,
    "build": cmd_build,
    "verify": cmd_verify,
    "import": cmd_import,
}


def main(argv: list[str] | None = None) -> int:
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
        return COMMANDS[args.command](loaded, args)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
