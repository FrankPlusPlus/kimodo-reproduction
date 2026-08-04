"""CLI for pinned public-resource planning, acquisition, and verification."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .config import ResourceConfigError, load_catalog, load_paths
from .manager import ResourceManager, ResourceVerificationError
from .pipeline import PipelineError, _atomic_yaml, prepare_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, help="Pinned remote resource catalog YAML")
    parser.add_argument("--paths", help="Machine-local destinations/existing paths YAML")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init", help="Generate a machine-local paths YAML")
    init.add_argument("--output", required=True)
    init.add_argument("--storage-root", required=True)
    init.add_argument("--legacy-root")
    init.add_argument("--conversion-inventory")
    init.add_argument("--asset-mode", choices=("hardlink", "copy"), default="hardlink")
    for name in ("plan", "verify"):
        command = subparsers.add_parser(name)
        command.add_argument(
            "--group", action="append", default=[], help="Resource group; repeat to combine groups"
        )
    fetch = subparsers.add_parser("fetch")
    fetch.add_argument(
        "--group", action="append", default=[], help="Resource group; repeat to combine groups"
    )
    fetch.add_argument(
        "--local-files-only",
        action="store_true",
        help="Forbid network access and use only the Hugging Face/local_dir cache",
    )
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--dry-run", action="store_true")
    prepare.add_argument(
        "--skip-source-verify",
        action="store_true",
        help="Skip full source hashes only when fetch/verify already succeeded in this session",
    )
    all_command = subparsers.add_parser("all")
    all_command.add_argument("--local-files-only", action="store_true")
    subparsers.add_parser(
        "adopt-legacy",
        help="Adopt pipeline.legacy_bundle_root without rerunning the text encoder",
    )
    bind = subparsers.add_parser(
        "bind-prepared",
        help="Fully verify a relocated train-ready bundle and write a paths YAML",
    )
    bind.add_argument("--prepared-root", required=True)
    bind.add_argument("--run-root", required=True)
    bind.add_argument("--output", required=True)
    return parser


def _init_paths(args, catalog) -> dict:
    output = Path(args.output).expanduser().resolve()
    root = Path(args.storage_root).expanduser().resolve()
    payload = {
        "schema_version": 1,
        "resources": {
            name: {
                "destination": str(root / "sources" / name),
                "existing_path": None,
            }
            for name in catalog.resources
        },
        "pipeline": {
            "dataset_root": str(root / "expanded/bones-seed"),
            "prepared_root": str(root / "prepared/public-seed-soma30-v1"),
            "run_root": str(root / "runs"),
            "repro_paths_yaml": str(root / "config/repro.paths.yaml"),
            "text_device": "cuda:0",
            "motion_workers": 8,
            "threads_per_worker": 2,
            "stats_workers": 16,
            "adoption_asset_mode": args.asset_mode,
        },
    }
    if args.legacy_root:
        payload["pipeline"]["legacy_bundle_root"] = str(
            Path(args.legacy_root).expanduser().resolve()
        )
        payload["pipeline"]["prepared_root"] = str(
            root / "prepared/adopted-legacy-soma30-v1"
        )
    if args.conversion_inventory:
        payload["pipeline"]["legacy_conversion_inventory"] = str(
            Path(args.conversion_inventory).expanduser().resolve()
        )
    _atomic_yaml(output, payload)
    return {"status": "paths_initialized", "paths": str(output)}


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        catalog = load_catalog(args.catalog)
        if args.command == "init":
            result = _init_paths(args, catalog)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "bind-prepared":
            from .adoption import bind_prepared_bundle

            result = bind_prepared_bundle(
                prepared_root=args.prepared_root,
                run_root=args.run_root,
                repro_paths_yaml=args.output,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if not args.paths:
            raise ResourceConfigError(f"{args.command} requires --paths")
        paths = load_paths(args.paths, catalog)
        groups = getattr(args, "group", None) or ["train-minimal"]
        manager = ResourceManager(catalog, paths)
        if args.command == "adopt-legacy":
            from .adoption import adopt_legacy_bundle

            if paths.pipeline is None or paths.pipeline.legacy_bundle_root is None:
                raise ResourceConfigError(
                    "adopt-legacy requires pipeline.legacy_bundle_root in paths YAML"
                )
            result = adopt_legacy_bundle(
                legacy_root=paths.pipeline.legacy_bundle_root,
                output_root=paths.pipeline.prepared_root,
                run_root=paths.pipeline.run_root,
                repro_paths_yaml=paths.pipeline.repro_paths_yaml,
                conversion_inventory=paths.pipeline.legacy_conversion_inventory,
                asset_mode=paths.pipeline.adoption_asset_mode,
            )
        elif args.command == "plan":
            result = manager.plan(groups)
        elif args.command == "verify":
            result = manager.verify(groups, raise_on_error=False)
            if not result["ok"]:
                print(json.dumps(result, indent=2, sort_keys=True))
                return 2
        elif args.command == "fetch":
            result = manager.fetch(groups, local_files_only=args.local_files_only)
        elif args.command == "prepare":
            if not args.skip_source_verify:
                manager.verify(["train-minimal"])
            result = prepare_pipeline(catalog, paths, dry_run=args.dry_run)
        else:
            manager.fetch(["train-minimal"], local_files_only=args.local_files_only)
            result = prepare_pipeline(catalog, paths)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        FileNotFoundError,
        TypeError,
        ValueError,
        ResourceConfigError,
        ResourceVerificationError,
        PipelineError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        print(json.dumps({"status": "error", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
