# SPDX-License-Identifier: Apache-2.0
"""Pin one visible GPU per torchrun rank before importing torch."""

from __future__ import annotations

import os


def pin_local_cuda_device(environ: dict[str, str] | None = None) -> str:
    """Restrict this process to a single GPU and keep LLM2Vec off cuda:0-of-the-node.

    generate_eval imports torch at module load. If eight ranks still see eight
    devices, LLM2Vec ``device=auto`` becomes ``device_map=cuda:0`` and every
    copy lands on GPU 0. Call this before any torch import.
    """
    env = os.environ if environ is None else environ
    local_rank = int(env.get("LOCAL_RANK", "0"))
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [item.strip() for item in visible.split(",") if item.strip()]
        if len(devices) > 1:
            if not 0 <= local_rank < len(devices):
                raise ValueError(
                    f"LOCAL_RANK={local_rank} is outside CUDA_VISIBLE_DEVICES={visible}"
                )
            chosen = devices[local_rank]
        elif devices:
            chosen = devices[0]
        else:
            chosen = str(local_rank)
    else:
        chosen = str(local_rank)
    env["CUDA_VISIBLE_DEVICES"] = chosen
    env.setdefault("TEXT_ENCODER_DEVICE", "cuda:0")
    return chosen
