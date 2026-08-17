#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""torchrun entry: pin one GPU, then run benchmark/generate_eval.py."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

from kimodo.evaluation.rank_cuda import pin_local_cuda_device

if __name__ == "__main__":
    chosen = pin_local_cuda_device()
    print(
        f"generate_eval_rank: LOCAL_RANK={os.environ.get('LOCAL_RANK', '0')} "
        f"CUDA_VISIBLE_DEVICES={chosen} "
        f"TEXT_ENCODER_DEVICE={os.environ.get('TEXT_ENCODER_DEVICE')}",
        flush=True,
    )
    target = Path(__file__).resolve().parents[1] / "benchmark" / "generate_eval.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
