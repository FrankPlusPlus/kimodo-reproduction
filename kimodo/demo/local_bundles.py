# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parse and auto-discover local demo checkpoint bundles."""

from __future__ import annotations

import importlib.util
import os
import re
from collections.abc import Iterable
from pathlib import Path


_EXPORT_DIR = re.compile(r"^step-(\d{9})$")
_TRAINER_PT = re.compile(r"^step-(\d{9})\.pt$")
_RUN_ALIASES = {
    "v2-1m-hostnet-wd03-from650k": "wd03",
    "v2-1m-hostnet-wd03-from780k-lr3e6": "wd03-lr3e6",
    "v2-1m-hostnet-kf-smooth-lr1e5": "kf-smooth",
    "v2-1m-hostnet-kf-smooth-lr1e5-step695k": "kf-smooth",
    "v2-1m-hostnet": "hostnet",
    "v2-1m-hostnet-k7-from690k": "k7-from690k",
    "v2-1m-hostnet-k7-reseed696k": "k7-reseed",
}


def parse_local_bundles(raw: str = "", extra: list[str] | None = None) -> dict[str, str]:
    """Parse LABEL=PATH (or LABEL:PATH) pairs from env / CLI into an ordered dict."""
    parts: list[str] = []
    for chunk in (raw or "").replace(";", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    for item in extra or []:
        item = (item or "").strip()
        if item:
            parts.append(item)
    bundles: dict[str, str] = {}
    for part in parts:
        if "=" in part:
            label, path = part.split("=", 1)
        elif ":" in part:
            label, path = part.split(":", 1)
        else:
            raise ValueError(f"Local bundle must be LABEL=PATH, got {part!r}")
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Local bundle label and path must be non-empty: {part!r}")
        bundles[label] = path
    return bundles


def _truthy(raw: str | None, default: bool = True) -> bool:
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _split_roots(raw: str) -> list[Path]:
    return [Path(item.strip()).expanduser() for item in raw.replace(";", ",").split(",") if item.strip()]


def default_export_roots() -> list[Path]:
    raw = os.environ.get("KIMODO_DEMO_EXPORT_ROOTS", "")
    if raw.strip():
        return _split_roots(raw)
    storage = os.environ.get("KIMODO_STORAGE_ROOT", "").strip()
    if storage:
        return [Path(storage).expanduser() / "eval-exports"]
    return []


def default_run_roots() -> list[Path]:
    raw = os.environ.get("KIMODO_DEMO_RUN_ROOTS", "")
    if raw.strip():
        return _split_roots(raw)
    storage = os.environ.get("KIMODO_STORAGE_ROOT", "").strip()
    if storage:
        return [Path(storage).expanduser() / "runs"]
    return []


def default_export_cache() -> Path | None:
    raw = os.environ.get("KIMODO_DEMO_EXPORT_CACHE", "").strip()
    if raw:
        return Path(raw).expanduser()
    storage = os.environ.get("KIMODO_STORAGE_ROOT", "").strip()
    if storage:
        return Path(storage).expanduser() / "eval-exports"
    return None


def is_exported_bundle(path: Path) -> bool:
    path = Path(path)
    return (
        path.is_dir()
        and (path / "config.yaml").is_file()
        and (path / "model.pt").is_file()
        and (path / "stats").exists()
    )


def run_alias(run_name: str) -> str:
    return _RUN_ALIASES.get(run_name, run_name)


def format_step_label(step: int) -> str:
    if step >= 1000 and step % 1000 == 0:
        return f"{step // 1000}k"
    return str(step)


def bundle_label(run_name: str, step: int, *, ready: bool) -> str:
    suffix = "" if ready else " (export on load)"
    return f"{run_alias(run_name)} {format_step_label(step)}{suffix}"


def uses_soma_visuals(model_name: str, local_labels: Iterable[str] | None = None) -> bool:
    """True for official SOMA models and for local training checkpoints (all SOMA-30)."""
    name = model_name or ""
    lowered = name.lower()
    if "g1" in lowered or "smplx" in lowered:
        return False
    if "soma" in lowered:
        return True
    return name in set(local_labels or ())


def skinning_mesh_mode(
    model_name: str,
    *,
    use_soma_layer: bool = False,
    local_labels: Iterable[str] | None = None,
) -> str:
    """Pick the Character mesh mode from a demo model name or local bundle label."""
    name = model_name or ""
    lowered = name.lower()
    if "g1" in lowered:
        return "g1_stl"
    if "smplx" in lowered:
        return "smplx_skin"
    if uses_soma_visuals(name, local_labels):
        return "soma_layer_skin" if use_soma_layer else "soma_skin"
    raise ValueError("The model name is not recognized for skinning.")


def version_options_with_training(
    official_display_names: list[str],
    *,
    skeleton_key: str | None,
    dataset_ui_label: str | None = None,
    local_labels: Iterable[str] | None = None,
) -> list[str]:
    """One Version dropdown: official display names, then local training checkpoints."""
    options = list(official_display_names)
    labels = list(local_labels or ())
    if not labels:
        return options
    skeleton_ok = bool(skeleton_key) and skeleton_key.lower() == "soma"
    dataset_ok = dataset_ui_label in (None, "SEED")
    if skeleton_ok and dataset_ok:
        options.extend(labels)
    return options


def _iter_named_dirs(root: Path, child: str) -> list[Path]:
    if not root.is_dir():
        return []
    if (root / child).is_dir():
        return [root]
    return [path for path in sorted(root.iterdir()) if path.is_dir()]


def discover_exported_bundles(export_roots: list[Path]) -> list[tuple[str, int, Path]]:
    found: list[tuple[str, int, Path]] = []
    for root in export_roots:
        for run_dir in _iter_named_dirs(root, "exports"):
            exports = run_dir / "exports"
            if not exports.is_dir():
                continue
            for path in sorted(exports.iterdir()):
                match = _EXPORT_DIR.fullmatch(path.name)
                if match is None or not is_exported_bundle(path):
                    continue
                found.append((run_dir.name, int(match.group(1)), path))
    return found


def discover_trainer_checkpoints(run_roots: list[Path]) -> list[tuple[str, int, Path]]:
    found: list[tuple[str, int, Path]] = []
    for root in run_roots:
        for run_dir in _iter_named_dirs(root, "checkpoints"):
            checkpoints = run_dir / "checkpoints"
            resolved = run_dir / "config.resolved.yaml"
            if not checkpoints.is_dir() or not resolved.is_file():
                continue
            for path in sorted(checkpoints.iterdir()):
                match = _TRAINER_PT.fullmatch(path.name)
                if match is None or not path.is_file() or not os.access(path, os.R_OK):
                    continue
                found.append((run_dir.name, int(match.group(1)), path))
    return found


def _unique_label(base: str, run_name: str, used: set[str]) -> str:
    if base not in used:
        return base
    candidate = f"{base} [{run_name}]"
    if candidate not in used:
        return candidate
    index = 2
    while f"{candidate}-{index}" in used:
        index += 1
    return f"{candidate}-{index}"


def collect_local_bundles(
    *,
    export_roots: list[Path] | None = None,
    run_roots: list[Path] | None = None,
    explicit: dict[str, str] | None = None,
    auto_discover: bool | None = None,
) -> dict[str, str]:
    """Discover exported bundles and trainer checkpoints; explicit labels win."""
    if auto_discover is None:
        auto_discover = _truthy(os.environ.get("KIMODO_DEMO_AUTO_DISCOVER"))
    if export_roots is None:
        export_roots = default_export_roots()
    if run_roots is None:
        run_roots = default_run_roots()

    ordered: list[tuple[str, str, int, Path, bool]] = []
    seen_keys: set[tuple[str, int]] = set()
    if auto_discover:
        for run_name, step, path in discover_exported_bundles(export_roots):
            key = (run_alias(run_name), step)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            ordered.append((run_name, run_alias(run_name), step, path, True))
        for run_name, step, path in discover_trainer_checkpoints(run_roots):
            key = (run_alias(run_name), step)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            ordered.append((run_name, run_alias(run_name), step, path, False))
        ordered.sort(key=lambda item: (-item[2], item[1], item[0]))

    bundles: dict[str, str] = {}
    used_labels: set[str] = set()
    for run_name, _alias, step, path, ready in ordered:
        label = _unique_label(bundle_label(run_name, step, ready=ready), run_name, used_labels)
        used_labels.add(label)
        bundles[label] = str(path)
    for label, path in (explicit or {}).items():
        bundles[label] = path
    return bundles


def trainer_output_run_dir(checkpoint: Path, export_cache: Path | None = None) -> Path:
    run_dir = checkpoint.expanduser().resolve().parent.parent
    cache = export_cache if export_cache is not None else default_export_cache()
    if cache is not None:
        return cache.expanduser().resolve() / run_dir.name
    return run_dir


def parse_trainer_step(checkpoint: Path) -> int:
    match = _TRAINER_PT.fullmatch(checkpoint.name)
    if match is None:
        raise ValueError(f"Not a trainer checkpoint filename: {checkpoint.name}")
    return int(match.group(1))


def _load_export_bundle():
    script = Path(__file__).resolve().parents[2] / "scripts" / "export_trainer_checkpoint_bundle.py"
    spec = importlib.util.spec_from_file_location("kimodo_demo_export_trainer_checkpoint_bundle", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load exporter from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.export_bundle


def ensure_inference_bundle(
    path: str | Path,
    *,
    export_cache: Path | None = None,
    exporter=None,
) -> Path:
    """Return an exported inference bundle, converting a trainer .pt on demand."""
    bundle = Path(path).expanduser().resolve()
    if is_exported_bundle(bundle):
        return bundle
    if bundle.suffix != ".pt" or not bundle.is_file():
        raise FileNotFoundError(f"Not an exported bundle or trainer checkpoint: {bundle}")
    resolved = bundle.parent.parent / "config.resolved.yaml"
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing resolved config for trainer checkpoint: {resolved}")
    step = parse_trainer_step(bundle)
    output_run_dir = trainer_output_run_dir(bundle, export_cache)
    destination = output_run_dir / "exports" / f"step-{step:09d}"
    if is_exported_bundle(destination):
        return destination
    export_fn = exporter if exporter is not None else _load_export_bundle()
    print(f"Exporting trainer checkpoint {bundle} -> {destination}")
    try:
        return Path(export_fn(
            checkpoint=bundle,
            resolved_config=resolved,
            output_run_dir=output_run_dir,
            step=step,
            force=False,
        ))
    except FileExistsError:
        if is_exported_bundle(destination):
            return destination
        return Path(export_fn(
            checkpoint=bundle,
            resolved_config=resolved,
            output_run_dir=output_run_dir,
            step=step,
            force=True,
        ))
    except PermissionError:
        fallback = Path(
            os.environ.get("KIMODO_DEMO_EXPORT_FALLBACK", str(Path.home() / "demo-exports"))
        ).expanduser() / bundle.parent.parent.name
        fallback_dest = fallback / "exports" / f"step-{step:09d}"
        if is_exported_bundle(fallback_dest):
            return fallback_dest
        print(f"Export cache not writable; falling back to {fallback}")
        return Path(export_fn(
            checkpoint=bundle,
            resolved_config=resolved,
            output_run_dir=fallback,
            step=step,
            force=is_exported_bundle(fallback_dest) is False and fallback_dest.exists(),
        ))
