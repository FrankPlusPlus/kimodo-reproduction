#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export an EMA inference bundle from a full trainer checkpoint (CPU-only).

Official eval/watcher consumes ``exports/step-XXXXXXXXX/`` bundles, not raw
``.pt`` trainer checkpoints. Training only auto-exports every milestone
(default 100k). This tool lets a sidecar export mid-run checkpoints for
proxy monitoring without touching the live DDP trainer.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf


def _publish_mode(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def _from_scratch_inference_config(model: dict) -> dict:
    return {
        "_target_": "kimodo.model.Kimodo",
        "denoiser": {
            "_target_": "kimodo.model.TwostageDenoiser",
            "motion_rep": {
                "_target_": "kimodo.motion_rep.KimodoMotionRep",
                "skeleton": {
                    "_target_": "kimodo.skeleton.registry.build_skeleton",
                    "nbjoints": int(model["skeleton_joints"]),
                },
                "fps": int(model["fps"]),
                "stats_path": "${checkpoint_dir}/stats",
            },
            "motion_mask_mode": model["motion_mask_mode"],
            "ckpt_path": "${checkpoint_dir}/model.pt",
            "detach_root_for_body": bool(model["detach_root_for_body"]),
            "llm_shape": [int(model["llm_tokens"]), int(model["llm_dim"])],
            "use_text_mask": bool(model["use_text_mask"]),
            "latent_dim": int(model["latent_dim"]),
            "ff_size": int(model["ff_size"]),
            "num_layers": int(model["num_layers"]),
            "num_heads": int(model["num_heads"]),
            "activation": model["activation"],
            "dropout": 0.0,
            "pe_dropout": 0.0,
            "norm_first": bool(model["norm_first"]),
            "num_text_tokens_override": int(model["num_text_tokens_override"]),
            "input_first_heading_angle": bool(model["input_first_heading_angle"]),
        },
        "text_encoder": None,
        "num_base_steps": int(model["num_diffusion_steps"]),
        "cfg_type": "separated",
    }


def export_bundle(
    *,
    checkpoint: Path,
    resolved_config: Path,
    output_run_dir: Path,
    step: int | None,
    force: bool,
) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    resolved_config = resolved_config.expanduser().resolve()
    output_run_dir = output_run_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not resolved_config.is_file():
        raise FileNotFoundError(resolved_config)

    with resolved_config.open("r", encoding="utf-8") as handle:
        resolved = yaml.safe_load(handle)
    model_cfg = resolved.get("model") or {}
    stats_source = Path(str(model_cfg.get("stats_path", ""))).expanduser()
    if not stats_source.is_dir():
        raise FileNotFoundError(f"stats_path missing: {stats_source}")
    if model_cfg.get("checkpoint_dir"):
        raise ValueError(
            "This exporter currently supports from-scratch runs "
            "(model.checkpoint_dir must be null)."
        )

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    global_step = int(step if step is not None else state["global_step"])
    ema = state.get("ema")
    if not isinstance(ema, dict) or not isinstance(ema.get("shadow"), dict):
        raise ValueError(f"Checkpoint has no EMA shadow weights: {checkpoint}")
    weights = dict(ema["shadow"])

    exports = output_run_dir / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    _publish_mode(exports, 0o755)
    destination = exports / f"step-{global_step:09d}"
    if destination.exists():
        if not force:
            raise FileExistsError(f"Bundle already exists: {destination}")
        shutil.rmtree(destination)

    temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".", dir=exports))
    try:
        model_path = temporary / "model.pt"
        torch.save(weights, model_path)
        _publish_mode(model_path, 0o644)
        OmegaConf.save(OmegaConf.create(_from_scratch_inference_config(model_cfg)), temporary / "config.yaml")
        shutil.copytree(stats_source, temporary / "stats")
        (temporary / "TRAINING_PROVENANCE.txt").write_text(
            "Exported offline from trainer checkpoint for sidecar benchmark monitoring.\n"
            f"source_checkpoint={checkpoint}\n"
            f"resolved_config={resolved_config}\n"
            f"global_step={global_step}\n"
            f"ema_updates={(ema or {}).get('num_updates')}\n",
            encoding="utf-8",
        )
        (temporary / "export_meta.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_checkpoint": str(checkpoint),
                    "resolved_config": str(resolved_config),
                    "global_step": global_step,
                    "ema_updates": int((ema or {}).get("num_updates") or 0),
                    "weights": "ema.shadow",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        for path in [destination, *destination.rglob("*")]:
            _publish_mode(path, 0o755 if path.is_dir() else 0o644)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Trainer checkpoint step-*.pt")
    parser.add_argument(
        "--resolved-config",
        required=True,
        help="Training run config.resolved.yaml",
    )
    parser.add_argument(
        "--output-run-dir",
        required=True,
        help="Directory that will receive exports/step-XXXXXXXXX",
    )
    parser.add_argument("--step", type=int, help="Override step in export dirname")
    parser.add_argument("--force", action="store_true", help="Replace existing export")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    destination = export_bundle(
        checkpoint=Path(args.checkpoint),
        resolved_config=Path(args.resolved_config),
        output_run_dir=Path(args.output_run_dir),
        step=args.step,
        force=bool(args.force),
    )
    print(json.dumps({"event": "export_complete", "bundle": str(destination)}, sort_keys=True))


if __name__ == "__main__":
    main()
