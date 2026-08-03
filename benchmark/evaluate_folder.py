# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Step (4) of evaluation pipeline.

This script recursively computes metrics for generated and ground-truth motions within a test suite folder tree. 
Saves metrics json files per test case and per group of test cases in the folder tree.
"""

import argparse
import json
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from kimodo.constraints import load_constraints_lst
from kimodo.meta import parse_prompts_from_meta
from kimodo.metrics import (
    PAPER_PELVIS_TO_SMOOTH_ROOT_2D,
    ContraintFollow,
    FootContactConsistency,
    FootSkateFromContacts,
    FootSkateFromHeight,
    FootSkateRatio,
    TMR_EmbeddingMetric,
    aggregate_metrics,
    clear_metrics,
    compute_metrics,
    compute_paper_constraint_errors,
    compute_tmr_per_sample_retrieval,
    compute_tmr_retrieval_metrics,
)
from kimodo.skeleton import build_skeleton
from kimodo.skeleton.definitions import SOMASkeleton30
from kimodo.tools import load_json, to_torch

DEFAULT_FPS = 30.0


def discover_motion_folders(root: Path) -> list[tuple[Path, Path]]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {root}")
    out: list[tuple[Path, Path]] = []
    for meta_path in root.rglob("meta.json"):
        sample_dir = meta_path.parent
        if (sample_dir / "motion.npz").is_file() and (sample_dir / "gt_motion.npz").is_file():
            rel = sample_dir.relative_to(root)
            out.append((sample_dir, rel))
    return sorted(out, key=lambda x: str(x[1]))


def group_by_parent(examples: list[tuple[Path, Path]]) -> list[list[tuple[Path, Path]]]:
    def parent_key(item: tuple[Path, Path]) -> Path:
        return item[1].parent if len(item[1].parts) > 1 else Path(".")

    sorted_examples = sorted(examples, key=parent_key)
    groups: list[list[tuple[Path, Path]]] = []
    for _key, group in groupby(sorted_examples, key=parent_key):
        groups.append(list(group))
    return groups


def _to_scalar(t: torch.Tensor) -> float:
    return float(t.mean().item()) if t.numel() > 0 else float(t.item())


def _to_p95(t: torch.Tensor) -> float:
    if t.numel() == 0:
        return float("nan")
    return float(torch.nanquantile(t, torch.tensor(0.95, device=t.device), dim=0).item())


def _per_sample_metrics_from_saved(metrics_list: list, n: int) -> list[dict[str, float]]:
    per_sample: list[dict[str, float]] = [{} for _ in range(n)]
    for metric in metrics_list:
        for key, lst in metric.saved_metrics.items():
            for i, t in enumerate(lst):
                if i >= n:
                    break
                per_sample[i][key] = _to_scalar(t)
    return per_sample


def _load_pair_embeddings(
    sample_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    motion_emb_path = sample_dir / "motion_embedding.npy"
    text_emb_path = sample_dir / "text_embedding.npy"
    gt_motion_emb_path = sample_dir / "gt_motion_embedding.npy"
    if not (motion_emb_path.is_file() and text_emb_path.is_file()):
        return None

    motion_emb = np.load(motion_emb_path)
    text_emb = np.load(text_emb_path)
    if motion_emb.ndim == 3 and motion_emb.shape[0] == 1:
        motion_emb = motion_emb[0]
    if text_emb.ndim == 3 and text_emb.shape[0] == 1:
        text_emb = text_emb[0]

    gt_motion_emb = None
    if gt_motion_emb_path.is_file():
        gt_motion_emb = np.load(gt_motion_emb_path)
        if gt_motion_emb.ndim == 3 and gt_motion_emb.shape[0] == 1:
            gt_motion_emb = gt_motion_emb[0]

    return motion_emb, text_emb, gt_motion_emb


def _load_npz_motion(
    npz_path: Path,
    device: str,
    soma30_skel: SOMASkeleton30 | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load posed_joints and foot_contacts from an NPZ, upscaling SOMA30 to SOMA77 if needed."""
    data = np.load(npz_path)
    posed_joints = to_torch(data["posed_joints"], device=device)
    foot_contacts = to_torch(data["foot_contacts"], device=device)

    if posed_joints.shape[-2] == 30 and soma30_skel is not None:
        local_rot_mats = to_torch(data["local_rot_mats"], device=device)
        root_positions = to_torch(data["root_positions"], device=device)
        out77 = soma30_skel.output_to_SOMASkeleton77(
            {"local_rot_mats": local_rot_mats, "root_positions": root_positions, "foot_contacts": foot_contacts}
        )
        posed_joints = out77["posed_joints"]
        foot_contacts = out77["foot_contacts"]

    return posed_joints, foot_contacts


def _load_npz_paper_motion(
    npz_path: Path,
    device: str,
    soma30_skel: SOMASkeleton30 | None = None,
) -> dict[str, torch.Tensor]:
    """Load fields required by the paper's Sec. 6.1 constraint metrics.

    Unlike the legacy metric loader, this deliberately refuses to derive the
    generated smooth-root trajectory from the pelvis.  Those are separate
    quantities in the paper and conflating them invalidates both root columns.
    """
    data = np.load(npz_path)
    required = {"posed_joints", "global_rot_mats", "smooth_root_pos"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise ValueError(f"Paper protocol requires {missing} in {npz_path}")

    output = {
        "posed_joints": to_torch(data["posed_joints"], device=device),
        "global_rot_mats": to_torch(data["global_rot_mats"], device=device),
        "smooth_root_pos": to_torch(data["smooth_root_pos"], device=device),
    }
    if output["posed_joints"].shape[-2] == 30 and soma30_skel is not None:
        if "local_rot_mats" in data.files and "root_positions" in data.files:
            local_rot_mats = to_torch(data["local_rot_mats"], device=device)
            root_positions = to_torch(data["root_positions"], device=device)
        elif "root_positions" in data.files:
            global_rot_mats = output["global_rot_mats"]
            local_rot_mats = soma30_skel.global_rots_to_local_rots(global_rot_mats)
            root_positions = to_torch(data["root_positions"], device=device)
        else:
            raise ValueError(
                f"Paper protocol cannot upscale SOMA30 rotations without root_positions: {npz_path}"
            )
        out77 = soma30_skel.output_to_SOMASkeleton77(
            {"local_rot_mats": local_rot_mats, "root_positions": root_positions}
        )
        output["posed_joints"] = out77["posed_joints"]
        output["global_rot_mats"] = out77["global_rot_mats"]
    return output


def _aggregate_paper_constraint_values(values: dict[str, list[torch.Tensor]]) -> dict[str, Any]:
    """Aggregate raw constraint points exactly once over a complete suite."""
    result: dict[str, Any] = {"means": {}, "counts": {}}
    for key, chunks in sorted(values.items()):
        if not chunks:
            continue
        flattened = torch.cat([chunk.detach().cpu().reshape(-1) for chunk in chunks])
        result["means"][key] = float(flattened.mean())
        result["counts"][key] = int(flattened.numel())
        if key == PAPER_PELVIS_TO_SMOOTH_ROOT_2D:
            result["pelvis_to_smooth_root_2d_p95_m"] = float(torch.quantile(flattened, 0.95))
    return result


def compute_paper_constraint_suite(
    examples: list[tuple[Path, Path]],
    skeleton: torch.nn.Module,
    device: str,
    soma30_skel: SOMASkeleton30 | None = None,
) -> dict[str, Any]:
    """Compute Sec. 6.1 errors over all constraint points in ``examples``."""
    generated: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
    ground_truth: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
    constrained_motions = 0
    for sample_dir, _ in examples:
        constraints_path = sample_dir / "constraints.json"
        if not constraints_path.is_file():
            continue
        constraints = load_constraints_lst(str(constraints_path), skeleton=skeleton)
        if not constraints:
            continue
        constrained_motions += 1
        for filename, accumulator in (("motion.npz", generated), ("gt_motion.npz", ground_truth)):
            motion = _load_npz_paper_motion(sample_dir / filename, device, soma30_skel)
            errors = compute_paper_constraint_errors(
                posed_joints=motion["posed_joints"],
                global_rot_mats=motion["global_rot_mats"],
                smooth_root_pos=motion["smooth_root_pos"],
                constraints_lst=constraints,
                root_idx=skeleton.root_idx,
            )
            for key, value in errors.items():
                accumulator[key].append(value)
    return {
        "protocol": "kimodo-paper-sec-6.1-constraint-points-v1",
        "aggregation": "mean over every constrained frame/joint; pelvis p95 over every root-constraint point",
        "num_constrained_motions": constrained_motions,
        "generated": _aggregate_paper_constraint_values(generated),
        "ground_truth": _aggregate_paper_constraint_values(ground_truth),
    }


def compute_paper_retrieval_set(examples: list[tuple[Path, Path]]) -> dict[str, Any]:
    """Compute TMR retrieval and FID once over a complete prompt test set."""
    motion_embeddings: list[np.ndarray] = []
    text_embeddings: list[np.ndarray] = []
    gt_embeddings: list[np.ndarray] = []
    missing: list[str] = []
    for sample_dir, _ in examples:
        loaded = _load_pair_embeddings(sample_dir)
        if loaded is None or loaded[2] is None:
            missing.append(str(sample_dir))
            continue
        motion, text, gt = loaded
        motion_embeddings.append(np.asarray(motion).reshape(-1))
        text_embeddings.append(np.asarray(text).reshape(-1))
        gt_embeddings.append(np.asarray(gt).reshape(-1))
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(
            f"Paper full-set retrieval requires generated/text/GT embeddings for every sample; "
            f"missing {len(missing)} ({preview})."
        )
    if not motion_embeddings:
        raise ValueError("Paper full-set retrieval received an empty test set.")
    metrics = compute_tmr_retrieval_metrics(
        np.stack(motion_embeddings),
        np.stack(text_embeddings),
        np.stack(gt_embeddings),
    )
    return {
        "protocol": "kimodo-paper-sec-6.1-full-test-set-tmr-v1",
        "aggregation": "single retrieval matrix and single FID fit over the complete prompt test set",
        "num_motions": len(examples),
        "metrics": metrics,
        "paper_reported_columns": {
            "R@3_generated_percent": metrics["TMR/t2m_R/R03"],
            "R@3_ground_truth_percent": metrics["TMR/t2m_gt_R/R03"],
            "FID_generated_vs_ground_truth": metrics["TMR/FID/gen_gt"],
        },
        "paper_table_note": "Kimodo Tables 1-2 multiply FID by 100 for readability.",
    }


def write_paper_protocol_outputs(
    folder: Path,
    examples: list[tuple[Path, Path]],
    skeleton: torch.nn.Module,
    device: str,
    soma30_skel: SOMASkeleton30 | None = None,
) -> list[Path]:
    """Write separately named paper-protocol artifacts without changing legacy columns."""
    written: list[Path] = []
    retrieval_groups: defaultdict[tuple[str, str], list[tuple[Path, Path]]] = defaultdict(list)
    constraint_groups: defaultdict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for example in examples:
        parts = example[1].parts
        if len(parts) >= 3 and parts[1] == "text2motion":
            retrieval_groups[(parts[0], parts[2])].append(example)
        if len(parts) >= 2 and parts[1] in {"constraints_withtext", "constraints_notext"}:
            constraint_groups[parts[0]].append(example)

    for (split, category), group in sorted(retrieval_groups.items()):
        path = folder / split / "text2motion" / category / "paper_retrieval.json"
        _write_json(path, compute_paper_retrieval_set(group))
        written.append(path)
    for split, group in sorted(constraint_groups.items()):
        path = folder / split / "paper_constraint_metrics.json"
        _write_json(path, compute_paper_constraint_suite(group, skeleton, device, soma30_skel))
        written.append(path)
    return written


def _run_eval_on_group(
    group: list[tuple[Path, Path]],
    skeleton: torch.nn.Module,
    metrics_list: list,
    device: str,
    group_name: str = "",
    soma30_skel: SOMASkeleton30 | None = None,
) -> tuple[
    list[dict[str, float]],
    list[dict[str, float]],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    list[dict[str, Any]],
]:
    """Run two passes: gen (motion.npz + embeddings) and GT (gt_motion.npz only). Return
    per_sample_gen, per_sample_gt, aggregated_gen, aggregated_gt, tmr_metrics, tmr_per_sample.
    """
    n = len(group)
    sample_ids: list[str] = []
    texts: list[str] = []
    motion_embs: list[np.ndarray] = []
    text_embs: list[np.ndarray] = []

    # ----- Pass 1: generation (motion.npz + all embeddings) -----
    clear_metrics(metrics_list)
    desc = f"Samples ({group_name})" if group_name else "Samples"
    for sample_dir, rel_path in tqdm(group, desc=desc, unit="motion"):
        stem = rel_path.name
        sample_ids.append(stem)
        meta_path = sample_dir / "meta.json"
        meta = load_json(meta_path)
        texts_parsed, _ = parse_prompts_from_meta(meta)
        texts.append(texts_parsed[0] if texts_parsed else "")

        posed_joints, foot_contacts = _load_npz_motion(sample_dir / "motion.npz", device, soma30_skel)
        nframes = posed_joints.shape[0]
        lengths = torch.tensor(nframes, dtype=torch.long, device=device)
        constraints_path = sample_dir / "constraints.json"
        constraints_lst = (
            load_constraints_lst(str(constraints_path), skeleton=skeleton) if constraints_path.is_file() else []
        )
        metrics_in: dict[str, Any] = {
            "posed_joints": posed_joints,
            "foot_contacts": foot_contacts,
            "lengths": lengths,
            "constraints_lst": constraints_lst,
        }
        text_this = texts_parsed[0] if texts_parsed else ""
        embs = _load_pair_embeddings(sample_dir)
        if (text_this or "").strip() and embs is not None:
            motion_emb, text_emb, gt_motion_emb = embs
            metrics_in["motion_emb"] = motion_emb
            metrics_in["text_emb"] = text_emb
            if gt_motion_emb is not None:
                metrics_in["gt_motion_emb"] = gt_motion_emb
            motion_embs.append(motion_emb)
            text_embs.append(text_emb)

        compute_metrics(metrics_list, metrics_in)

    per_sample_gen = _per_sample_metrics_from_saved(metrics_list, n)
    raw_aggregated_gen = aggregate_metrics(metrics_list)
    aggregated_gen = {}
    tmr_metrics: dict[str, float] = {}
    has_text = len(motion_embs) == n and len(text_embs) == n
    for key, v in raw_aggregated_gen.items():
        val = _to_scalar(v)
        if key.startswith("TMR/"):
            if has_text:
                tmr_metrics[key] = val
        else:
            aggregated_gen[key] = val
    if "constraint_root2d_err" in raw_aggregated_gen:
        aggregated_gen["constraint_root2d_err_p95"] = _to_p95(raw_aggregated_gen["constraint_root2d_err"])

    tmr_per_sample: list[dict[str, Any]] = []
    if has_text and motion_embs and text_embs and len(motion_embs) == n and len(text_embs) == n:
        motion_emb_stack = np.stack(motion_embs, axis=0)
        text_emb_stack = np.stack(text_embs, axis=0)
        tmr_per_sample = compute_tmr_per_sample_retrieval(motion_emb_stack, text_emb_stack, sample_ids, texts, top_k=5)

    # ----- Pass 2: GT (gt_motion.npz only, no embeddings) -----
    clear_metrics(metrics_list)
    for sample_dir, rel_path in tqdm(group, desc=f"GT ({group_name})" if group_name else "GT", unit="motion"):
        posed_joints, foot_contacts = _load_npz_motion(sample_dir / "gt_motion.npz", device, soma30_skel)
        nframes = posed_joints.shape[0]
        lengths = torch.tensor(nframes, dtype=torch.long, device=device)
        constraints_path = sample_dir / "constraints.json"
        constraints_lst = (
            load_constraints_lst(str(constraints_path), skeleton=skeleton) if constraints_path.is_file() else []
        )
        metrics_in = {
            "posed_joints": posed_joints,
            "foot_contacts": foot_contacts,
            "lengths": lengths,
            "constraints_lst": constraints_lst,
        }
        compute_metrics(metrics_list, metrics_in)

    per_sample_gt = _per_sample_metrics_from_saved(metrics_list, n)
    raw_aggregated_gt = aggregate_metrics(metrics_list)
    aggregated_gt = {}
    for key, v in raw_aggregated_gt.items():
        if key.startswith("TMR/"):
            continue
        aggregated_gt[key] = _to_scalar(v)
    if "constraint_root2d_err" in raw_aggregated_gt:
        aggregated_gt["constraint_root2d_err_p95"] = _to_p95(raw_aggregated_gt["constraint_root2d_err"])

    return (
        per_sample_gen,
        per_sample_gt,
        aggregated_gen,
        aggregated_gt,
        tmr_metrics,
        tmr_per_sample,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Recursively evaluate generated motions; write metrics.json per folder and <name>.json per parent.",
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Root folder to search recursively for meta.json + motion.npz + gt_motion.npz",
    )
    parser.add_argument("--device", default=None, help="cuda/cpu. Default: auto")
    parser.add_argument(
        "--paper-protocol",
        action="store_true",
        help=(
            "Additionally run the strict Sec. 6.1 evaluator: full-set TMR/FID and raw-point "
            "constraint means/p95. Missing smooth-root, rotation, or embedding assets are errors."
        ),
    )
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"Folder does not exist: {folder}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    examples = discover_motion_folders(folder)
    if not examples:
        raise SystemExit(f"No directories with meta.json, motion.npz, and gt_motion.npz found under {folder}")
    print(f"Discovered {len(examples)} motion folders.")

    first_posed = np.load(examples[0][0] / "motion.npz")["posed_joints"]
    num_joints = first_posed.shape[-2]

    # SOMA models could generate 30-joint output; upscale to 77 for evaluation
    soma30_skel: SOMASkeleton30 | None = None
    if num_joints == 30:
        soma30_skel = SOMASkeleton30().to(device)
        _ = soma30_skel.somaskel77  # trigger lazy init
        soma30_skel.somaskel77.to(device)
        skeleton = soma30_skel.somaskel77
        print("Detected SOMA30 motions; will upscale to SOMA77 for evaluation.")
    else:
        skeleton = build_skeleton(num_joints).to(device)

    fps = DEFAULT_FPS
    kwargs = {"skeleton": skeleton, "fps": fps}
    metrics_list = [
        FootSkateFromHeight(**kwargs),
        FootSkateFromContacts(**kwargs),
        FootContactConsistency(**kwargs),
        FootSkateRatio(**kwargs),
        ContraintFollow(**kwargs),
        TMR_EmbeddingMetric(**kwargs),
    ]

    groups = group_by_parent(examples)
    for group in tqdm(groups, desc="Evaluating folders"):
        sample_dirs = [g[0] for g in group]
        folder_for_group = sample_dirs[0].parent
        folder_name = folder_for_group.name

        (
            per_sample_gen,
            per_sample_gt,
            aggregated_gen,
            aggregated_gt,
            tmr_metrics,
            tmr_per_sample,
        ) = _run_eval_on_group(group, skeleton, metrics_list, device, group_name=folder_name, soma30_skel=soma30_skel)

        texts = []
        for sample_dir, _ in group:
            meta = load_json(sample_dir / "meta.json")
            texts_parsed, _ = parse_prompts_from_meta(meta)
            texts.append(texts_parsed[0] if texts_parsed else "")

        for i, (sample_dir, _) in enumerate(group):
            metrics_path = sample_dir / "metrics.json"
            out = {
                "num_motions": 1,
                "folder": str(sample_dir),
                "per_motion_mean_gen": per_sample_gen[i] if i < len(per_sample_gen) else {},
                "per_motion_mean_gt": per_sample_gt[i] if i < len(per_sample_gt) else {},
            }
            if i < len(tmr_per_sample):
                out["tmr"] = {
                    "t2m_rank": tmr_per_sample[i]["rank"],
                    "text": texts[i] if i < len(texts) else "",
                    "top5_retrieved": tmr_per_sample[i]["top_k"],
                }
            _write_json(metrics_path, out)

        parent_json_path = folder_for_group.parent / f"{folder_name}.json"
        full_metrics = {
            "num_motions": len(group),
            "folder": str(folder_for_group),
            "per_motion_mean_gen": aggregated_gen,
            "per_motion_mean_gt": aggregated_gt,
        }
        if tmr_metrics:
            full_metrics["tmr"] = tmr_metrics
        _write_json(parent_json_path, full_metrics)

    if args.paper_protocol:
        paper_paths = write_paper_protocol_outputs(folder, examples, skeleton, device, soma30_skel)
        print(f"Wrote {len(paper_paths)} separately named Sec. 6.1 paper-protocol files.")

    print(f"Wrote metrics.json in each of {len(examples)} folders and folder-level JSONs for {len(groups)} groups.")


if __name__ == "__main__":
    main()
