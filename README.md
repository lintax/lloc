# lloc — LLM Localization

A command-line tool that localizes app strings with LLMs served by
[Ollama](https://ollama.com). It reads base strings and their context from
TOML, translates only what changed, validates the result, keeps terminology
consistent and writes platform files: Android `strings.xml`, iOS `.strings`,
flat JSON for the web (i18next-compatible).

- **Zero install**: Python standard library only, no pip, no venv.
- **Delta only**: missing and stale strings are translated; deleted keys are
  purged everywhere.
- **Git-friendly**: all state is text files with deterministic key order,
  meant to be committed.
- **Monorepo-ready**: workspace mode with inherited settings, product groups
  and a shared worker pool.

## Requirements

- **Python 3.11+** (for `tomllib`).
- **Ollama**, only for commands that call a model: `translate`,
  `verify --terms`, `verify --semantic`. Everything else works offline.

The default profile, `cloud`, runs three Ollama Cloud models and needs an
Ollama Pro subscription. `[ollama] profile = "local"` switches to two local
models (about 12 GB of VRAM); missing models are pulled on first use. See
[Model profiles](#model-profiles).

## Quickstart: single project

The localization directory (usually `myapp/l10n/`) holds two hand-written
files:

```toml
# l10n.toml
[app]
description = "Project tracker for small teams: boards, tasks, deadlines"
tone = "neutral, concise UI language; formal address"
base_language = "en"
languages = ["ru", "de", "ja"]      # BCP-47

[output]
adapter = "android-xml"
[output.android-xml]
res_dir = "../app/src/main/res"     # paths are relative to l10n.toml
```

```toml
# strings.toml — the source of truth for every string
[menu_search]
text = "Search"
context = "Main menu item, opens project search"
type = "button"                     # button | title | message
```

```bash
python lloc.py --dir myapp/l10n status       # what needs translating (no LLM)
python lloc.py --dir myapp/l10n translate    # translate the delta, then build
```

If the strings already live in a platform project, generate `strings.toml`
from the base file, then import the existing translations:

```bash
python lloc.py --dir myapp/l10n import --bootstrap-strings
python lloc.py --dir myapp/l10n import
```

`python -m lloc` is equivalent to `python lloc.py`.

## Files

| File | Written by | Purpose |
|---|---|---|
| `l10n.toml` | you | Project config: app description, tone, languages, adapter. |
| `strings.toml` | you | Base strings: `text` (required), `context`, `type`. |
| `locales/<lang>.json` | lloc | Canonical translation store. |
| `l10n.lock.json` | lloc | Hashes of the base texts; its diff shows which strings changed. |
| `terms/<lang>.json`, `semantic/<lang>.json` | lloc | Terminology and QA caches. |
| platform files | lloc (`build`) | Generated output, never edited by hand. |

Commit everything lloc writes. Add `.lloc.lock` (the run lock) to
`.gitignore`.

Each locale entry stores the translation, `src` (hash of the base text it was
made from), `model` and `translated_at`. A key is **missing** when it has no
entry, **stale** when `src` differs from the current hash, **orphan** when it
is gone from `strings.toml`, and **ok** otherwise.

**Plurals are not supported, by design.** Write base texts whose form does
not depend on the number: `"Items: %d"`, not `"%d items"`. This removes a
whole class of translation errors across languages with up to six plural
forms. Platform plural constructs (`<plurals>`, `.stringsdict`) stay in the
platform files and are left untouched.

An optional glossary forces specific translations; `"*"` means "do not
translate in any language":

```toml
[glossary]
"PDF" = { "*" = "PDF" }
"board" = { ru = "доска" }
```

## Commands

Common options: `--dir PATH` (directory with `l10n.toml`, or a workspace root;
default `.`), `--lang xx[,yy]`, `--verbose`.

| Command | What it does | LLM |
|---|---|---|
| `status` | Table of ok / missing / stale / orphan per language. Exit code 1 if there is work. | no |
| `translate` | Translates the delta, purges deleted keys, updates `locales/` and the lock, then runs `build`. | yes |
| `build` | Generates platform files from `locales/`. | no |
| `verify` | Validates existing translations: placeholders, lock consistency, terminology heuristics, script mixing, duplicate keys within a group. | `--terms`, `--semantic` |
| `import` | Imports existing platform translations into `locales/`. Per module only in a workspace. | no |
| `report` | Lists recently translated strings for review. `--since 90m/24h/7d` (default 24h). | no |
| `set` | Manual fix: `set <key> --lang de "text"`. Per module only in a workspace. | no |

`translate` flags:

- `--dry-run` shows the plan without calling a model; `--limit N` caps strings
  per language; `--no-build` skips the build.
- `--force` retranslates everything, including manual entries.
- `--model NAME` overrides the translator for an A/B run.
- `--retranslate-model PATTERN` retranslates entries whose stored `model`
  matches an fnmatch pattern, e.g. `"gemma3*"` or `imported`.
- `--fix-terms` retranslates strings that diverge from the majority term.
- `--fix-semantic` re-resolves the backlog of suspect and arbitrated entries.

`import` flags: `--bootstrap-strings` generates `strings.toml` from the base
platform file; `--assume-stale` marks imported entries as stale; `--overwrite`
replaces entries already in `locales/`. Imported entries get
`model = "imported"`.

`verify` flags: `--terms` builds an exact LLM term map; `--semantic` audits
meaning by back-translation. Both only warn and do not change the exit code.

```bash
python lloc.py --dir l10n translate --dry-run          # estimate the work
python lloc.py --dir l10n translate --lang ru,de       # two languages only
python lloc.py --dir l10n translate --force --lang ja  # redo one language
python lloc.py --dir l10n verify --terms               # terminology audit
```

## Adapters

Selected with `[output] adapter`. All adapters regenerate managed keys (those
in `strings.toml`) on `build`, keep everything else found in existing files,
and write atomically.

```toml
[output]
adapter = "android-xml"             # values/ + values-b+xx/strings.xml
[output.android-xml]
res_dir = "../app/src/main/res"
legacy_codes = { id = "in", he = "iw" }
```

```toml
[output]
adapter = "ios"                     # <lang>.lproj/Localizable.strings (UTF-8)
[output.ios]
lproj_dir = "../App/Resources"
# filename = "Localizable.strings"  # default
```

```toml
[output]
adapter = "json"                    # one flat, key-sorted <lang>.json per language
[output.json]
dir = "../web/src/i18n"
```

- **android-xml**: strings marked `translatable="false"`, `<string-array>` and
  `<plurals>` live in the XML only and survive `build`.
- **ios**: directories are named by BCP-47 code as is (`es-419.lproj`); the
  base language goes to `<base>.lproj`, not `Base.lproj`. `context` is written
  as a `/* ... */` comment. `%@` and `%1$@` are validated like `%s`.
- **json**: no nesting; dots in keys are literal.

Android legacy codes (`in`, `iw`) are never used in iOS or JSON file names.

## Workspace mode (monorepo)

Everything workspace-level lives in one `l10n/` directory at the repository
root; module configs stay next to the modules.

```
monorepo/
├── l10n/
│   ├── config.toml        # members + shared defaults
│   └── tracker.toml       # group "tracker" (any *.toml except config.toml)
├── apps/web/l10n.toml
├── apps/mobile/l10n.toml
└── libs/core/l10n.toml
```

```toml
# l10n/config.toml
members = ["apps/*", "libs/*"]      # explicit globs, no auto-scan

[app]
tone = "neutral, concise"
base_language = "en"
languages = ["ru", "de"]

[ollama]
workers = 3                         # one pool for the whole run
```

```toml
# l10n/tracker.toml — product group context
context = "Trackly product family: project tracking apps and a shared core."
```

```toml
# apps/web/l10n.toml — inherits the defaults, adds its own
group = "tracker"

[app]
description = "Web variant; keyboard-first UI."

[output]
adapter = "json"
[output.json]
dir = "src/i18n"
```

```bash
python lloc.py --dir monorepo status             # all modules
python lloc.py --dir monorepo translate          # shared worker pool
python lloc.py --dir monorepo/apps/web status    # one module
```

Inheritance has two levels: workspace defaults, then the module, which wins
per key. A group is the consistency boundary: few-shot examples and
terminology are shared inside it and never cross it. `verify` warns about
duplicate keys within a group.

## How translation works

Every machine translation is produced by **two models and a judge**:

1. Model A translates the delta.
2. Model B translates the same strings independently.
3. Judge C compares the candidates on meaning alone. Agreement gives the
   verdict `ok`; picking one gives `arbitrated`, which is re-checked later.

One model's blind spot is covered by the other. The cost is roughly 3x a
single-model run, which is a deliberate trade of speed for reliability.

Each candidate is validated before it is judged; a failure triggers a retry
with the specific errors, up to `max_retries`:

- **Placeholders**: the multiset of `%s`, `%d`, `%1$s`, `%@`, `\n` must match
  the base text.
- **Length**: at most `length_ratio` (default 3.0) times the base, with a
  floor so short strings are not rejected.
- **Cleanliness**: no reasoning, wrapping quotes or empty values.
- **Script**: the dominant script must be the one expected for the language.
  Disable with `[validation] script_check = false`.

A translation identical to the base text is not an error.

**Nothing blocks on a disagreement.** If the models and the judge still
disagree after the retries, the judge must pick the least bad candidate. It is
stored as `arbitrated`, printed as `ARBITRATED key: chose … (reason)` and
listed in `report`. `failed` is reserved for mechanical problems: broken
placeholders or transport errors. An entry arbitrated twice is treated as
resolved until its base text changes.

**Few-shot examples come from trusted entries only**, in this priority: manual
fixes, judge-verified translations, imported translations. Unverified output
never becomes an example, so one early mistake cannot spread. On a cold start,
one representative string per frequent term is translated first, and the rest
of the language learns its terms from those.

**Terminology** is extracted without markup and without an LLM: frequent words
and bigrams of the base texts become terms, with inflections merged
(`projects` → `project`). `verify` flags a term translated inconsistently
within a language; `translate --fix-terms` retranslates the minority strings.
If the model insists on its variant, the divergence is recorded as accepted
and no longer flagged. A word listed in `[glossary]` is always its own term,
which separates derivations such as `edit` and `editor`.

`verify` also warns about **script mixing**: letters of a foreign script, or
long untranslated Latin tokens in a non-Latin language. Tune it with
`[validation] latin_token_max` (default 5) and `latin_allowlist`; glossary
terms are allowed automatically.

## Model profiles

A profile sets all three roles and the throughput settings, so projects need
no `[ollama]` section at all.

| | `cloud` (default) | `local` |
|---|---|---|
| A — `[ollama] model` | `deepseek-v4.1-flash:cloud` | `gemma4:12b-it-qat` |
| B — `[validation] ensemble_model` | `glm-5.3-flash:cloud` | `qwen3.5:9b` |
| C — `[validation] semantic_model` | `gemma4:31b-cloud` | none: B judges |
| `batch_size` / `timeout_sec` / `workers` | 20 / 600 / 10 | 10 / 120 / 1 |
| Ensemble | concurrent per batch | phased, two model swaps per run |

`cloud` uses three independent model families and runs A and B in parallel on
every batch. `local` gives the same ensemble on a 12 GB GPU, two to three
times slower.

```toml
[ollama]
profile = "local"
```

Any single key overrides the profile, in a module or in the workspace
defaults:

```toml
[ollama]
workers = 4
batch_size = 30
[validation]
semantic_model = "qwen3.5:4b"   # a different judge
ensemble_model = false          # one translator plus self-check
```

`false` removes a role the profile set. `semantic = "off"` disables semantic
QA and the ensemble altogether.

Further settings:

- **`[models]`** pins a language to a model: `"lv" = "qwen3.5:9b"`. Languages
  are grouped by model, so each model loads once per run. If 25% or more of a
  language's strings go to arbitration, lloc prints `model looks weak in lv`
  and suggests retranslating with another model.
- **`batch_size`**: larger batches mean fewer round trips but coarser retries.
  Use about 10 locally and 20–30 in the cloud.
- **`num_ctx`**: unset, local models get 8192 and cloud models get no limit.
  An explicit value is always respected.
- **`[validation] ensemble_concurrency`** (`auto` | `batch` | `phased`):
  `auto` runs concurrently only when all models are cloud models. Forcing
  `batch` on local models makes Ollama swap models on every batch.
- **`[validation] semantic_mode`** (`auto` | `inline` | `phased`), for a
  single translator with a separate judge: `phased` translates everything,
  swaps to the judge once, then swaps back to fix the suspects. `inline`
  checks the previous batch while the next one is translated; it needs both
  models in memory, so set `OLLAMA_MAX_LOADED_MODELS=2`.

When the judge belongs to neither translator's family, it compares the
candidates with the original directly, in one call per batch. Otherwise it
judges a literal back-translation made without context, so a model cannot
"remember" what it meant.

To audit existing translations with a stronger model and fix what it finds:

```bash
python lloc.py --dir myapp/l10n verify --semantic --lang am,my,lo --model qwen3.5:9b
python lloc.py --dir myapp/l10n translate --fix-semantic --lang am,my,lo
```

Limitation: a model judging its own family misses words that family does not
know, and small judges fail first on rare languages.

## Manual fixes

```bash
python lloc.py --dir myapp/l10n set menu_search --lang de "Suchen"
```

`set` writes the entry with `model = "manual"`, validates placeholders
(`--force-set` accepts it anyway) and rebuilds that language.

- Manual entries are the top-priority few-shot examples, so one fix pulls the
  style of neighbouring strings along.
- When the base text changes, a manual entry becomes stale and is
  retranslated; `translate` reports which ones.
- `--force` rewrites manual entries too. `--retranslate-model "gemma*"` does
  not match them; `--retranslate-model manual` does.
- Editing `locales/<lang>.json` by hand is valid if you set
  `"model": "manual"`, but `set` also validates and builds.
- Never edit generated platform files. If fixes were already made there, pull
  them in with `import --overwrite --lang xx`.

## Interrupting and rerunning

`translate` saves the locale after every batch, so it can be interrupted at
any point and the next run continues where it stopped. Ctrl+C stops the
workers, prints `interrupted — progress saved, rerun to continue` and exits
with code 130; a second Ctrl+C exits immediately.

All files are written atomically. Concurrent runs on one module are excluded
by an OS lock (`.lloc.lock`) that dies with the process, so no cleanup is
needed after a crash.

## Rules for coding agents

A block to paste into the `CLAUDE.md` or `AGENTS.md` of a repository that uses
lloc. Adjust the path in the first line.

```markdown
## Localization (lloc)

Run: `python <path>/lloc.py --dir <l10n directory> <command>`.

- **`strings.toml` is the source of truth** (plus `l10n.toml`). Never edit
  generated files by hand: `locales/*.json`, `l10n.lock.json`, platform files.
- To fix one translation: `set <key> --lang de "text"`. This is the only
  correct way to make a manual fix.
- When you add or change a string in `strings.toml`, fill in `context` and
  `type` (button / title / message): translation quality depends on them.
  Then run `status` to see the delta.
- Development does not wait for translations: run `build` right after editing
  `strings.toml` (locales fall back to the base text), build and test, and run
  `translate` in the background. Check `status` before committing.
- `translate` is incremental and takes seconds to minutes after small edits.
  Initial translation, `--force` and new languages are slow: run them in the
  background. Reruns are safe and continue where the last run stopped.
- Estimate a long run first: `translate --dry-run`.
- Before a release: `verify` (fast, no LLM); `verify --terms` for a deeper
  terminology audit.
- To delete a string, delete its key from `strings.toml`. The next `translate`
  removes it everywhere.
- Android special keys (`translatable="false"`, `<string-array>`, `<plurals>`)
  live in the XML, are not managed by lloc and survive `build`.
- No plurals: write number-independent base texts such as `"Items: %d"`.
- In a workspace (`l10n/config.toml` with `members`), commands run from the
  root cover all modules and `--dir <module>` covers one.
```

## Tests

```bash
python -m unittest discover -s tests -t .
```
