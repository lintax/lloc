"""Adapter registry. SPEC.md §8."""
from __future__ import annotations

from ..config import ConfigError, Project
from .android_xml import AndroidXmlAdapter
from .base import Adapter

REGISTRY: dict[str, type[Adapter]] = {
    AndroidXmlAdapter.name: AndroidXmlAdapter,
}


def get_adapter(project: Project) -> Adapter:
    cls = REGISTRY.get(project.adapter_name)
    if cls is None:
        known = ", ".join(sorted(REGISTRY))
        raise ConfigError(f"unknown adapter {project.adapter_name!r} (known: {known})")
    return cls(project)
