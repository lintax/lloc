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

DEFAULT_MODEL = "gemma4:12b-it-qat"  # A/B winner over qwen3.5:9b (see SPEC.md §6.1)
DEFAULT_URL = "http://localhost:11434"
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_RETRIES = 2
DEFAULT_TIMEOUT_SEC = 120
DEFAULT_WORKERS = 1
DEFAULT_NUM_CTX = 8192
DEFAULT_LENGTH_RATIO = 3.0

CONFIG_NAME = "l10n.toml"
STRINGS_NAME = "strings.toml"
WORKSPACE_DIR = "l10n"                    # monorepo root: l10n/config.toml + l10n/<tag>.toml
WORKSPACE_CONFIG = "config.toml"
# sections whose keys a module inherits from workspace defaults (member wins per key)
INHERITED_SECTIONS = ("app", "ollama", "terms", "validation", "glossary")


class ConfigError(Exception):
    pass


@dataclass
class StringEntry:
    key: str
    text: str
    context: str = ""
    type: str = ""


@dataclass
class OllamaConfig:
    url: str = DEFAULT_URL
    model: str = DEFAULT_MODEL
    batch_size: int = DEFAULT_BATCH_SIZE
    max_retries: int = DEFAULT_MAX_RETRIES
    timeout_sec: int = DEFAULT_TIMEOUT_SEC
    workers: int = DEFAULT_WORKERS
    num_ctx: int = DEFAULT_NUM_CTX
    prompts_dir: str | None = None


@dataclass
class TermsConfig:
    enabled: bool = True
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

    o = cfg.get("ollama", {})
    ollama = OllamaConfig(
        url=str(o.get("url", DEFAULT_URL)).rstrip("/"),
        model=str(o.get("model", DEFAULT_MODEL)),
        batch_size=int(o.get("batch_size", DEFAULT_BATCH_SIZE)),
        max_retries=int(o.get("max_retries", DEFAULT_MAX_RETRIES)),
        timeout_sec=int(o.get("timeout_sec", DEFAULT_TIMEOUT_SEC)),
        workers=int(o.get("workers", DEFAULT_WORKERS)),
        num_ctx=int(o.get("num_ctx", DEFAULT_NUM_CTX)),
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

    t = cfg.get("terms", {})
    terms = TermsConfig(
        enabled=bool(t.get("enabled", True)),
        min_occurrences=int(t.get("min_occurrences", 3)),
        min_bigram_occurrences=int(t.get("min_bigram_occurrences", 2)),
        stopword_df=float(t.get("stopword_df", 0.35)),
    )

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
        adapter_name=str(adapter_name),
        adapter_config=adapter_config,
        length_ratio=length_ratio,
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
    workers = int(cfg.get("ollama", {}).get("workers", DEFAULT_WORKERS))
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
