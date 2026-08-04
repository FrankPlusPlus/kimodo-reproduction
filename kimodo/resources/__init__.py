"""Pinned, relocatable acquisition of public Kimodo training resources."""

from .config import ResourceCatalog, ResourcePaths, load_catalog, load_paths
from .manager import ResourceManager

__all__ = [
    "ResourceCatalog",
    "ResourceManager",
    "ResourcePaths",
    "load_catalog",
    "load_paths",
]
