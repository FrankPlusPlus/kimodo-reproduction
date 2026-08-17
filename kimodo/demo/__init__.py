# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: I001
import argparse
import os


def main() -> None:
    from kimodo.model import DEFAULT_MODEL
    from kimodo.model.registry import resolve_model_name

    from .app import Demo
    from .local_bundles import collect_local_bundles, parse_local_bundles

    parser = argparse.ArgumentParser(description="Run the kimodo demo UI.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Default model to load (e.g. Kimodo-SOMA-SEED-v1.1, kimodo-soma-seed, or a local bundle label).",
    )
    parser.add_argument(
        "--local-bundle",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Extra LABEL=PATH bundle. Repeatable. Also reads KIMODO_DEMO_LOCAL_BUNDLES.",
    )
    parser.add_argument(
        "--no-auto-discover",
        action="store_true",
        help="Do not scan eval-exports / run checkpoints; only use --local-bundle.",
    )
    args = parser.parse_args()

    local_bundles = collect_local_bundles(
        explicit=parse_local_bundles(
            os.environ.get("KIMODO_DEMO_LOCAL_BUNDLES", ""),
            extra=args.local_bundle,
        ),
        auto_discover=False if args.no_auto_discover else None,
    )
    print(f"Training checkpoints in UI: {len(local_bundles)}")
    for label, path in local_bundles.items():
        print(f"  {label}: {path}")
    if args.model in local_bundles:
        default_model_name = args.model
    else:
        default_model_name = resolve_model_name(args.model, "Kimodo")
    demo = Demo(default_model_name=default_model_name, local_bundles=local_bundles)
    demo.run()


if __name__ == "__main__":
    main()
