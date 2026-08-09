# SPDX-License-Identifier: Apache-2.0
"""Build a deterministic, stratified official benchmark testsuite subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from kimodo.common.file_permissions import publish_file

SCHEMA_VERSION = 1
_STAGE_FILES = ("meta.json", "seed_motion.json", "seed_constraints.json")
_GT_FILES = ("gt_motion.npz", "constraints.json")


@dataclass(frozen=True)
class SelectedCase:
    split_path: str
    case_id: str

    @property
    def rel_dir(self) -> str:
        return f"{self.split_path}/{self.case_id}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_unit(seed: int, *parts: str) -> float:
    payload = "\x1f".join((str(seed), *parts)).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return (value + 1) / (2**64 + 1)


def _leaf_group(rel_parts: tuple[str, ...]) -> str:
    if len(rel_parts) < 3:
        raise ValueError(f"Unexpected benchmark path depth: {rel_parts!r}")
    if rel_parts[1] == "text2motion":
        return "/".join(rel_parts[:3])
    if len(rel_parts) < 4:
        raise ValueError(f"Unexpected constraint path depth: {rel_parts!r}")
    return "/".join(rel_parts[:4])


def _quota(full_count: int, rate: float, min_count: int) -> int:
    if full_count <= 0:
        return 0
    return min(full_count, max(min_count, math.ceil(full_count * rate)))


def _index_testsuite(root: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for meta_path in root.rglob("meta.json"):
        rel = meta_path.relative_to(root)
        if len(rel.parts) < 2:
            continue
        groups[_leaf_group(rel.parts[:-1])].append(rel.parts[-2])
    for group in groups:
        groups[group] = sorted(set(groups[group]))
    if not groups:
        raise ValueError(f"No benchmark cases found under {root}")
    return dict(groups)


def _select_cases(
    groups: dict[str, list[str]],
    *,
    rate: float,
    seed: int,
    min_constraint: int,
    min_text2motion: int,
) -> list[SelectedCase]:
    selected: list[SelectedCase] = []
    for group in sorted(groups):
        case_ids = groups[group]
        minimum = min_text2motion if group.split("/")[1] == "text2motion" else min_constraint
        count = _quota(len(case_ids), rate, minimum)
        ranked = sorted(case_ids, key=lambda case_id: _stable_unit(seed, group, case_id))
        for case_id in ranked[:count]:
            selected.append(SelectedCase(split_path=group, case_id=case_id))
    return selected


def _copy_if_present(source: Path, target: Path) -> bool:
    if not source.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def _find_gt_source(rel_dir: str, gt_roots: list[Path]) -> Path | None:
    for root in gt_roots:
        candidate = root / rel_dir
        if (candidate / "gt_motion.npz").is_file():
            return candidate
    return None


def build_benchmark_subset(args) -> dict:
    source = Path(args.source_testsuite).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else output / "proxy_manifest.json"
    gt_roots = [Path(item).expanduser().resolve() for item in args.gt_source_root or ()]

    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")

    groups = _index_testsuite(source)
    selected = _select_cases(
        groups,
        rate=float(args.rate),
        seed=int(args.seed),
        min_constraint=int(args.min_constraint),
        min_text2motion=int(args.min_text2motion),
    )

    output.mkdir(parents=True, exist_ok=True)
    staged_gt = 0
    staged_meta = 0
    per_group_counts = Counter()
    for case in selected:
        per_group_counts[case.split_path] += 1
        rel_dir = case.rel_dir
        src_dir = source / rel_dir
        dst_dir = output / rel_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in _STAGE_FILES:
            if _copy_if_present(src_dir / name, dst_dir / name):
                staged_meta += 1
        gt_source = _find_gt_source(rel_dir, gt_roots)
        if gt_source is not None:
            for name in _GT_FILES:
                if _copy_if_present(gt_source / name, dst_dir / name):
                    if name == "gt_motion.npz":
                        staged_gt += 1

    group_rows = []
    for group in sorted(groups):
        full_count = len(groups[group])
        pick_count = per_group_counts.get(group, 0)
        group_rows.append(
            {
                "group": group,
                "full_count": full_count,
                "selected_count": pick_count,
                "selected_fraction": pick_count / full_count if full_count else 0.0,
                "case_ids": sorted(
                    case.case_id for case in selected if case.split_path == group
                ),
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "builder": "kimodo.devtools.benchmark_subset_cli",
        "name": args.name,
        "seed": int(args.seed),
        "rate": float(args.rate),
        "min_constraint": int(args.min_constraint),
        "min_text2motion": int(args.min_text2motion),
        "selection_policy": {
            "kind": "stratified_proportional_per_leaf_group",
            "ordering": "stable sha256(seed, group, case_id)",
            "quota": "min(group_count, max(min_group, ceil(group_count * rate)))",
            "leaf_group_rules": {
                "text2motion": "split/text2motion/{overview|timeline_single|timeline_multi}",
                "constraints": "split/constraints_{withtext|notext}/.../leaf_subtype",
            },
        },
        "source_testsuite": str(source),
        "source_testsuite_case_count": sum(len(ids) for ids in groups.values()),
        "output": str(output),
        "selected_case_count": len(selected),
        "selected_fraction": len(selected) / sum(len(ids) for ids in groups.values()),
        "gt_prefilled_from": [str(root) for root in gt_roots],
        "gt_prefilled_count": staged_gt,
        "groups": group_rows,
        "producer": {
            "path": "kimodo/devtools/benchmark_subset_cli.py",
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=manifest_path.name + ".", suffix=".tmp", dir=manifest_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        publish_file(temporary)
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)

    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = _sha256(manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-testsuite", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--name", default="stratified-10pct-v1")
    parser.add_argument("--rate", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--min-constraint", type=int, default=10)
    parser.add_argument("--min-text2motion", type=int, default=40)
    parser.add_argument(
        "--gt-source-root",
        action="append",
        default=[],
        help="Optional roots that already contain gt_motion.npz to copy before create_benchmark.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    print(json.dumps(build_benchmark_subset(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
