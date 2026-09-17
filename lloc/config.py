"""Loading and validation of l10n.toml and strings.toml. See SPEC.md §4.

Workspace mode (SPEC.md §14): a monorepo root keeps everything l10n-related in
one l10n/ directory — l10n/config.toml (members globs + workspace defaults) and
l10n/<tag>.toml group files (any *.toml except config.toml is a group, filename
= tag). Module configs stay as l10n.toml inside module directories and inherit
workspace defaults (strictly two levels). Single-project mode (l10n.toml next
to strings.toml) is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib  # Python 3.11+ (the supported runtime)
except ModuleNotFoundError:  # pragma: no cover - dev environments < 3.11 only
    import tomli as tomllib  # type: ignore

DEFAULT_URL = "http://localhost:11434"
DEFAULT_MAX_RETRIES = 2
DEFAULT_NUM_CTX = 8192
DEFAULT_LENGTH_RATIO = 3.0

# --- model profiles (SPEC.md §6.2) ---------------------------------------
# A profile is the whole ensemble setup in one word: the three model roles
# (A = translator, B = second opinion, C = judge) plus the throughput knobs
# they need. Projects get the production setup without configuring anything;
# any individual key stays overridable in l10n.toml (member wins per key).
PROFILES: dict[str, dict] = {
    # Production default: three independent model families in Ollama Cloud,
    # so their blind spots do not overlap (the lv-incident: one family's
    # judge cannot catch what that same family got wrong). Needs an Ollama
    # Pro subscription; Pro holds three cloud models at once, so there are
    # no swaps and the ensemble runs concurrently per batch.
    "cloud": {
        "model": "glm-5.3-flash:cloud",           # phase A (Zhipu GLM)
        "ensemble_model": "deepseek-v4.1-flash:cloud",  # phase B (DeepSeek)
        "semantic_model": "gpt-oss:120b-cloud",   # phase C judge (OpenAI)
        "batch_size": 20,      # cloud models hold 256K-1M ctx: fewer round-trips
        "timeout_sec": 600,
        "workers": 10,         # no VRAM limit; NB concurrent ensemble keeps
                               # 2 requests in flight per worker (A+B)
    },
    # Offline fallback sized for a 12 GB GPU: two local models, phased (they
    # do not fit in VRAM together), the second translator doubles as judge.
    "local": {
        "model": "gemma4:12b-it-qat",   # A/B winner over qwen3.5:9b (SPEC.md §6.1)
        "ensemble_model": "qwen3.5:9b",
        "semantic_model": None,         # judge = the ensemble model itself
        "batch_size": 10,
        "timeout_sec": 120,
        "workers": 1,
    },
}
DEFAULT_PROFILE = "cloud"

# Back-compat aliases: the defaults of the default profile.
DEFAULT_MODEL = PROFILES[DEFAULT_PROFILE]["model"]
DEFAULT_BATCH_SIZE = PROFILES[DEFAULT_PROFILE]["batch_size"]
DEFAULT_TIMEOUT_SEC = PROFILES[DEFAULT_PROFILE]["timeout_sec"]
DEFAULT_WORKERS = PROFILES[DEFAULT_PROFILE]["workers"]

CONFIG_NAME = "l10n.toml"
STRINGS_NAME = "strings.toml"
WORKSPACE_DIR = "l10n"                    # monorepo root: l10n/config.toml + l10n/<tag>.toml
WORKSPACE_CONFIG = "config.toml"
# sections whose keys a module inherits from workspace defaults (member wins per key)
INHERITED_SECTIONS = ("app", "ollama", "terms", "validation", "glossary", "models")


class ConfigError(Exception):
    pass


def _is_cloud_tag(model: str) -> bool:
    """Ollama Cloud tags: 'gpt-oss:120b-cloud', 'deepseek-v4.1-flash:cloud'."""
    return model.endswith("-cloud") or model.endswith(":cloud")


@dataclass
class StringEntry:
    key: str
    text: str
    context: str = ""
    type: str = ""


@dataclass
class OllamaConfig:
    url: str = DEFAULT_URL
    profile: str = DEFAULT_PROFILE
    model: str = DEFAULT_MODEL
    batch_size: int = DEFAULT_BATCH_SIZE
    max_retries: int = DEFAULT_MAX_RETRIES
    timeout_sec: int = DEFAULT_TIMEOUT_SEC
    workers: int = DEFAULT_WORKERS
    num_ctx: int | None = None  # None = not set explicitly: local models get
                                # DEFAULT_NUM_CTX, cloud models get NO num_ctx
                                # at all (the server sizes the context itself)
    prompts_dir: str | None = None


@dataclass
class TermsConfig:
    min_occurrences: int = 3          # single words: in >= N distinct strings
    min_bigram_occurrences: int = 2
    stopword_df: float = 0.35         # document frequency above which a word is functional


@dataclass
class Sibling:
    """Lightweight view of another module of the same group (SPEC.md §14)."""
    module: str
    root: Path
    strings: dict[str, StringEntry]


@dataclass
class Project:
    root: Path                      # directory containing l10n.toml; all paths relative to it
    description: str = ""
    tone: str = ""
    base_language: str = "en"
    languages: list[str] = field(default_factory=list)
    glossary: dict[str, dict[str, str]] = field(default_factory=dict)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    adapter_name: str = ""
    adapter_config: dict = field(default_factory=dict)
    length_ratio: float = DEFAULT_LENGTH_RATIO
    latin_token_max: int = 5          # [validation]: script-mix heuristic (SPEC.md §7)
    latin_allowlist: list[str] = field(default_factory=list)
    script_check: bool = True         # [validation]: reject wrong-script candidates
                                      # (false = escape hatch for pseudo-localization)
    models: dict[str, str] = field(default_factory=dict)  # [models]: per-language override
    semantic_mode: str = "batch"      # [validation].semantic: "batch" (default) | "off"
    semantic_model: str | None = None  # [validation].semantic_model: separate judge model
    judge_mode: str = "auto"          # [validation].semantic_mode: auto | inline | phased
    ensemble_model: str | None = None  # [validation].ensemble_model: second translator
    ensemble_concurrency: str = "auto"  # [validation].ensemble_concurrency: auto|batch|phased

    @property
    def semantic_inline_judge(self) -> bool:
        """Concurrent inline judge: a separate (small) judge model runs in
        parallel with the main model. Explicit opt-in ('inline') — "auto"
        never risks VRAM by loading two models. Ensemble mode supersedes."""
        return (self.semantic_mode == "batch" and bool(self.semantic_model)
                and self.judge_mode == "inline" and not self.ensemble_model)

    @property
    def semantic_phased(self) -> bool:
        """Phased judge flow (translate all -> swap to judge -> fix): the safe
        default whenever a judge model is configured ("auto" or "phased").
        Ensemble mode supersedes — it has its own arbitration phase."""
        return (self.semantic_mode == "batch" and bool(self.semantic_model)
                and self.judge_mode in ("auto", "phased") and not self.ensemble_model)

    @property
    def ensemble(self) -> bool:
        """Ensemble mode: every string is translated by TWO models with a
        meaning-only arbitration (SPEC.md §13.7)."""
        return self.semantic_mode == "batch" and bool(self.ensemble_model)

    @property
    def ensemble_judge(self) -> str:
        """The arbiter of ensemble phase C: [validation].semantic_model if
        set, else the second translator itself."""
        return self.semantic_model or self.ensemble_model or self.ollama.model

    @property
    def ensemble_batch(self) -> bool:
        """Concurrent per-batch ensemble (SPEC.md §13.7): A and B translate
        the same batch in parallel, the judge follows immediately. Explicit
        "batch" turns it on; "auto" enables it only when ALL THREE models are
        cloud tags (no VRAM, no swaps — the phases were a swap-era artifact).
        Note: "auto" inspects the three configured models, not [models] pins."""
        if not self.ensemble or self.ensemble_concurrency == "phased":
            return False
        if self.ensemble_concurrency == "batch":
            return True
        return all(_is_cloud_tag(m) for m in
                   (self.ollama.model, self.ensemble_model or "", self.ensemble_judge))
    terms: TermsConfig = field(default_factory=TermsConfig)
    strings: dict[str, StringEntry] = field(default_factory=dict)  # insertion-ordered
    # workspace mode (SPEC.md §14); empty/defaults in single-project mode
    module: str = ""                # relative path from the workspace root, e.g. "apps/tv"
    group: str = ""                 # group tag; a module without one is its own group
    group_context: str = ""         # context from l10n/<tag>.toml
    siblings: list[Sibling] = field(default_factory=list)  # same group, other modules

    @property
    def full_context(self) -> str:
        """Layered prompt context: group context -> module description
        (workspace tone comes in via inheritance, string context per entry)."""
        return "\n\n".join(p for p in (self.group_context.strip(), self.description) if p)


@dataclass
class Workspace:
    root: Path                      # monorepo root (the directory containing l10n/)
    members: list[Project]
    workers: int = DEFAULT_WORKERS  # shared worker pool for the whole run


def _load_toml(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"file not found: {path}")
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path.name}: invalid TOML: {e}") from e


def _load_strings(path: Path) -> dict[str, StringEntry]:
    data = _load_toml(path)
    strings: dict[str, StringEntry] = {}
    for key, val in data.items():
        if not isinstance(val, dict):
            raise ConfigError(f"{STRINGS_NAME}: [{key}] must be a table")
        text = val.get("text")
        if isinstance(text, dict):
            raise ConfigError(
                f"{STRINGS_NAME}: [{key}]: plurals are not supported by design; "
                'rephrase the base text so the form does not depend on the number, e.g. "Items: %d" '
                "(SPEC.md §12)"
            )
        if not isinstance(text, str) or not text:
            raise ConfigError(f"{STRINGS_NAME}: [{key}]: missing or empty 'text'")
        strings[key] = StringEntry(
            key=key,
            text=text,
            context=str(val.get("context", "")),
            type=str(val.get("type", "")),
        )
    if not strings:
        raise ConfigError(f"{STRINGS_NAME}: no strings defined")
    return strings


def _resolve_profile(cfg: dict, where: str = CONFIG_NAME) -> tuple[str, dict]:
    """[ollama].profile picks the built-in setup every other key overrides."""
    name = cfg.get("ollama", {}).get("profile", DEFAULT_PROFILE)
    if not isinstance(name, str) or name not in PROFILES:
        raise ConfigError(f"{where}: [ollama].profile must be one of "
                          f"{', '.join(sorted(PROFILES))}, got {name!r}")
    return name, PROFILES[name]


def _model_key(section: dict, key: str, default: str | None,
               where: str, section_name: str) -> tuple[str | None, bool]:
    """A model-name key that the profile may have preset: a name overrides it,
    `false` switches the profile's model off. Returns (value, set_explicitly)."""
    if key not in section:
        return default, False
    val = section[key]
    if val is False:
        return None, True
    if not isinstance(val, str) or not val:
        raise ConfigError(f"{where}: [{section_name}].{key} must be a model name "
                          "or false (to disable the one from [ollama].profile)")
    return val, True


def _build_project(root: Path, cfg: dict, require_strings: bool = True) -> Project:
    app = cfg.get("app", {})
    languages = app.get("languages", [])
    if not languages or not isinstance(languages, list):
        raise ConfigError(f"{CONFIG_NAME}: [app].languages must be a non-empty list")
    base = str(app.get("base_language", "en"))
    if base in languages:
        raise ConfigError(
            f"{CONFIG_NAME}: base_language {base!r} must not be in [app].languages"
        )
    if len(set(languages)) != len(languages):
        raise ConfigError(f"{CONFIG_NAME}: duplicate entries in [app].languages")

    glossary = cfg.get("glossary", {})
    for term, langs in glossary.items():
        if not isinstance(langs, dict):
            raise ConfigError(f"{CONFIG_NAME}: [glossary].{term!r} must be a table of lang -> translation")

    profile_name, prof = _resolve_profile(cfg)
    o = cfg.get("ollama", {})
    ollama = OllamaConfig(
        url=str(o.get("url", DEFAULT_URL)).rstrip("/"),
        profile=profile_name,
        model=str(o.get("model", prof["model"])),
        batch_size=int(o.get("batch_size", prof["batch_size"])),
        max_retries=int(o.get("max_retries", DEFAULT_MAX_RETRIES)),
        timeout_sec=int(o.get("timeout_sec", prof["timeout_sec"])),
        workers=int(o.get("workers", prof["workers"])),
        num_ctx=int(o["num_ctx"]) if "num_ctx" in o else None,
        prompts_dir=o.get("prompts_dir"),
    )
    if ollama.batch_size < 1 or ollama.workers < 1:
        raise ConfigError(f"{CONFIG_NAME}: [ollama] batch_size and workers must be >= 1")

    out = cfg.get("output", {})
    adapter_name = out.get("adapter")
    if not adapter_name:
        raise ConfigError(f"{CONFIG_NAME}: [output].adapter is required")
    adapter_config = out.get(adapter_name, {})

    validation = cfg.get("validation", {})
    length_ratio = float(validation.get("length_ratio", DEFAULT_LENGTH_RATIO))
    latin_token_max = int(validation.get("latin_token_max", 5))
    latin_allowlist = [str(x) for x in validation.get("latin_allowlist", [])]
    script_check = bool(validation.get("script_check", True))
    semantic_mode = str(validation.get("semantic", "batch"))
    if semantic_mode not in ("batch", "off"):
        raise ConfigError(f'{CONFIG_NAME}: [validation].semantic must be "batch" or "off", '
                          f"got {semantic_mode!r}")
    semantic_model, _ = _model_key(validation, "semantic_model",
                                   prof["semantic_model"], CONFIG_NAME, "validation")
    judge_mode = str(validation.get("semantic_mode", "auto"))
    if judge_mode not in ("auto", "inline", "phased"):
        raise ConfigError(f'{CONFIG_NAME}: [validation].semantic_mode must be '
                          f'"auto", "inline" or "phased", got {judge_mode!r}')
    ensemble_model, ensemble_explicit = _model_key(
        validation, "ensemble_model", prof["ensemble_model"], CONFIG_NAME, "validation")
    if ensemble_model is not None and semantic_mode == "off":
        # An explicit pair of contradicting keys is a config error; the same
        # collision with a profile-provided model just means "no ensemble"
        # (semantic = "off" is the documented way to switch the QA off).
        if ensemble_explicit:
            raise ConfigError(f"{CONFIG_NAME}: [validation].ensemble_model requires "
                              'semantic = "batch" (ensemble is a semantic-QA mode)')
        ensemble_model = None
    ensemble_concurrency = str(validation.get("ensemble_concurrency", "auto"))
    if ensemble_concurrency not in ("auto", "batch", "phased"):
        raise ConfigError(f'{CONFIG_NAME}: [validation].ensemble_concurrency must be '
                          f'"auto", "batch" or "phased", got {ensemble_concurrency!r}')

    t = cfg.get("terms", {})
    if "enabled" in t:
        raise ConfigError(
            f"{CONFIG_NAME}: [terms].enabled was removed — the term system is "
            "always on (decision record: SPEC.md §13.6); delete the key")
    terms = TermsConfig(
        min_occurrences=int(t.get("min_occurrences", 3)),
        min_bigram_occurrences=int(t.get("min_bigram_occurrences", 2)),
        stopword_df=float(t.get("stopword_df", 0.35)),
    )

    models = cfg.get("models", {})
    for mlang, mname in models.items():
        if not isinstance(mname, str) or not mname:
            raise ConfigError(f"{CONFIG_NAME}: [models].{mlang} must be a model name")
        if mlang not in languages:
            raise ConfigError(f"{CONFIG_NAME}: [models].{mlang}: "
                              f"language not in [app].languages")

    strings_path = root / STRINGS_NAME
    if require_strings or strings_path.is_file():
        strings = _load_strings(strings_path)
    else:
        strings = {}

    return Project(
        root=root,
        description=str(app.get("description", "")).strip(),
        tone=str(app.get("tone", "")),
        base_language=base,
        languages=list(languages),
        glossary=glossary,
        ollama=ollama,
        models={k: str(v) for k, v in models.items()},
        adapter_name=str(adapter_name),
        adapter_config=adapter_config,
        length_ratio=length_ratio,
        latin_token_max=latin_token_max,
        latin_allowlist=latin_allowlist,
        script_check=script_check,
        semantic_mode=semantic_mode,
        semantic_model=semantic_model,
        judge_mode=judge_mode,
        ensemble_model=ensemble_model,
        ensemble_concurrency=ensemble_concurrency,
        terms=terms,
        strings=strings,
    )


# --- workspace mode (SPEC.md §14) ----------------------------------------

def _workspace_config_path(root: Path) -> Path:
    return root / WORKSPACE_DIR / WORKSPACE_CONFIG


def _merge_member_cfg(defaults: dict, member: dict) -> dict:
    """Strictly two levels: workspace defaults -> module config (member wins
    per key inside inherited sections). [output] and description are module's."""
    out = dict(member)
    for section in INHERITED_SECTIONS:
        base = defaults.get(section)
        if not isinstance(base, dict):
            continue
        merged = dict(base)
        merged.pop("description", None)  # module-specific, never inherited
        merged.update(member.get(section, {}))
        out[section] = merged
    return out


def _member_dirs(root: Path, patterns) -> list[Path]:
    """Explicit globs, no autoscanning: each pattern must yield at least one
    directory containing l10n.toml; matches without a config are ignored."""
    if not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns) \
            or not patterns:
        raise ConfigError(f"{WORKSPACE_DIR}/{WORKSPACE_CONFIG}: "
                          "'members' must be a non-empty list of glob patterns")
    dirs: list[Path] = []
    for pat in patterns:
        found = [d for d in sorted(root.glob(pat))
                 if d.is_dir() and (d / CONFIG_NAME).is_file()]
        if not found:
            raise ConfigError(f"{WORKSPACE_DIR}/{WORKSPACE_CONFIG}: members pattern {pat!r} "
                              f"matched no directory with {CONFIG_NAME}")
        dirs += [d for d in found if d not in dirs]
    return dirs


def _group_context(root: Path, tag: str, explicit: bool) -> str:
    """Group definition lives in l10n/<tag>.toml (one group = one file, edited
    by the product team; the root config stays thin)."""
    p = root / WORKSPACE_DIR / f"{tag}.toml"
    if not p.is_file():
        if explicit:
            raise ConfigError(f"group {tag!r}: file not found: {WORKSPACE_DIR}/{tag}.toml")
        return ""  # module without a group is its own group; no file needed
    data = _load_toml(p)
    return str(data.get("context", "")).strip()


def _load_workspace(root: Path, cfg: dict, require_strings: bool = True) -> Workspace:
    members: list[Project] = []
    for d in _member_dirs(root, cfg.get("members")):
        mcfg = _load_toml(d / CONFIG_NAME)
        if "members" in mcfg:
            raise ConfigError(f"{d / CONFIG_NAME}: nested workspaces are not supported")
        project = _build_project(d, _merge_member_cfg(cfg, mcfg), require_strings)
        project.module = d.relative_to(root).as_posix()
        tag = mcfg.get("group") or mcfg.get("app", {}).get("group")
        if tag is not None and (not isinstance(tag, str) or not tag):
            raise ConfigError(f"{d / CONFIG_NAME}: 'group' must be a non-empty string")
        project.group = tag or project.module
        project.group_context = _group_context(root, project.group, explicit=bool(tag))
        members.append(project)
    for m in members:  # wire same-group siblings (term/few-shot boundary)
        m.siblings = [Sibling(o.module, o.root, o.strings)
                      for o in members if o is not m and o.group == m.group]
    _, prof = _resolve_profile(cfg, f"{WORKSPACE_DIR}/{WORKSPACE_CONFIG}")
    workers = int(cfg.get("ollama", {}).get("workers", prof["workers"]))
    return Workspace(root=root, members=members, workers=max(1, workers))


def _find_workspace_member(root: Path, require_strings: bool) -> Project | None:
    """--dir <module> inside a monorepo: find the workspace root upwards and
    return the module built with inheritance, group and siblings (or None)."""
    for parent in root.parents:
        wcfg_path = _workspace_config_path(parent)
        if not wcfg_path.is_file():
            continue
        cfg = _load_toml(wcfg_path)
        if "members" not in cfg:
            continue
        try:
            dirs = _member_dirs(parent, cfg.get("members"))
        except ConfigError:
            dirs = []
        if root not in dirs:
            return None  # nearest workspace does not list this module -> standalone
        ws = _load_workspace(parent, cfg, require_strings=False)
        me = next(m for m in ws.members if m.root == root)
        if require_strings and not me.strings:
            raise ConfigError(f"file not found: {root / STRINGS_NAME}")
        return me
    return None


def load_project(directory: str | Path, require_strings: bool = True) -> Project:
    """Single project (module) loader; applies workspace inheritance when the
    directory is a member of an enclosing workspace.
    require_strings=False is for bootstrap mode, when strings.toml does not exist yet."""
    root = Path(directory).resolve()
    cfg = _load_toml(root / CONFIG_NAME)
    if "members" in cfg:
        raise ConfigError(f"{CONFIG_NAME}: 'members' belongs to the workspace config "
                          f"({WORKSPACE_DIR}/{WORKSPACE_CONFIG}), not to a module config")
    member = _find_workspace_member(root, require_strings)
    if member is not None:
        return member
    return _build_project(root, cfg, require_strings)


def load(directory: str | Path, require_strings: bool = True) -> Project | Workspace:
    """Entry point for the CLI: workspace root (l10n/config.toml with members)
    or a single project/module directory (l10n.toml)."""
    root = Path(directory).resolve()
    if not (root / CONFIG_NAME).is_file():
        wpath = _workspace_config_path(root)
        if wpath.is_file():
            cfg = _load_toml(wpath)
            if "members" in cfg:
                return _load_workspace(root, cfg, require_strings)
    return load_project(directory, require_strings)
