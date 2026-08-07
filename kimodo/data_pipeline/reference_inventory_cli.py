# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build or fully verify a manifest reference inventory."""

from __future__ import annotations

import argparse
import json

from .reference_inventory import build_reference_inventory, verify_reference_inventory_full


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Hash all references and create an inventory")
    build.add_argument("--manifest", required=True)
    build.add_argument("--output", required=True)
    verify = subparsers.add_parser("verify", help="Re-hash and verify every inventory reference")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--inventory", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "build":
        result = build_reference_inventory(args.manifest, args.output)
    else:
        result = verify_reference_inventory_full(args.manifest, args.inventory)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
