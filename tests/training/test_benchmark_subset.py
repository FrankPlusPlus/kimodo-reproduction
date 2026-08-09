from __future__ import annotations

import argparse
import json
from pathlib import Path

from kimodo.devtools.benchmark_subset_cli import build_benchmark_subset


def test_benchmark_subset_is_stratified_and_deterministic(tmp_path: Path):
    source = tmp_path / "testsuite"
    for split, parts, case_ids in [
        ("content", ("text2motion", "overview"), ["0001", "0002", "0003", "0004", "0005"]),
        ("content", ("constraints_notext", "root", "path_2dpos"), ["0001", "0002", "0003"]),
    ]:
        for case_id in case_ids:
            case_dir = source / split / Path(*parts) / case_id
            case_dir.mkdir(parents=True)
            (case_dir / "meta.json").write_text("{}", encoding="utf-8")
            (case_dir / "seed_motion.json").write_text("{}", encoding="utf-8")

    output = tmp_path / "subset"
    receipt = build_benchmark_subset(
        argparse.Namespace(
            source_testsuite=str(source),
            output=str(output),
            manifest=str(output / "proxy_manifest.json"),
            name="test-subset",
            rate=0.5,
            seed=7,
            min_constraint=1,
            min_text2motion=2,
            gt_source_root=[],
            overwrite=False,
        )
    )
    assert receipt["selected_case_count"] == 5
    assert (output / "content/text2motion/overview/0001/meta.json").is_file()
    assert (output / "content/constraints_notext/root/path_2dpos/0001/meta.json").is_file()
    manifest = json.loads((output / "proxy_manifest.json").read_text(encoding="utf-8"))
    assert manifest["groups"][0]["selected_count"] >= 2

    second = build_benchmark_subset(
        argparse.Namespace(
            source_testsuite=str(source),
            output=str(tmp_path / "subset2"),
            manifest=str(tmp_path / "subset2" / "proxy_manifest.json"),
            name="test-subset",
            rate=0.5,
            seed=7,
            min_constraint=1,
            min_text2motion=2,
            gt_source_root=[],
            overwrite=False,
        )
    )
    assert second["selected_case_count"] == receipt["selected_case_count"]
