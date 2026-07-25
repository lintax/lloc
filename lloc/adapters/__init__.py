"""Adapter registry. SPEC.md §8."""
from __future__ import annotations

from ..config import ConfigError, Project
from .android_xml import AndroidXmlAdapter
from .base import Adapter
from .ios_strings import IosStringsAdapter
from .json_flat import JsonAdapter

REGISTRY: dict[str, type[Adapter]] = {
    AndroidXmlAdapter.name: AndroidXmlAdapter,
    IosStringsAdapter.name: IosStringsAdapter,
    JsonAdapter.name: JsonAdapter,
}


def get_adapter(project: Project) -> Adapter:
    cls = REGISTRY.get(project.adapter_name)
    if cls is None:
        known = ", ".join(sorted(REGISTRY))
        raise ConfigError(f"unknown adapter {project.adapter_name!r} (known: {known})")
    return cls(project)
