"""CLI for pinned public-resource planning, acquisition, and verification."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from .config import ResourceConfigError, load_catalog, load_paths
from .manager import ResourceManager, ResourceVerificationError
from .pipeline import PipelineError, prepare_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, help="Pinned remote resource catalog YAML")
    parser.add_argument("--paths", required=True, help="Machine-local destinations/existing paths YAML")
    subparsers = parser.add_subparsers(dest="command", required=True)
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
    return parser


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        catalog = load_catalog(args.catalog)
        paths = load_paths(args.paths, catalog)
        groups = getattr(args, "group", None) or ["train-minimal"]
        manager = ResourceManager(catalog, paths)
        if args.command == "plan":
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
