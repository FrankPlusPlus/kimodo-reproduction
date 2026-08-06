# Kimodo reproduction three-stage validation (2026-08-06)

## Executive conclusion

The three-stage validation completed successfully on two NVIDIA H200 NVL GPUs. The result is a
strong **training-pipeline and scale-down learning signal**, not a claim that the 10-hour/30k-step
model reproduces the released model's public benchmark quality.

The decisive stage used a 10.001-hour Core10 subset, trained from random initialization for 20k
text-only steps followed by 10k constraint-curriculum steps. On a 512-motion validation split that
was excluded from training by `take_name`, the EMA checkpoint improved as follows:

| Fixed held-out objective | Random initialization | Phase 1, step 20k | Phase 2, step 30k |
| --- | ---: | ---: | ---: |
| Text-only denoising total | 7.134681 | 0.161987 | 0.121654 |
| Phase-2-conditioned denoising total | 6.877775 | 0.162267 | 0.116747 |

Phase 2 reduced the same fixed conditioned objective by 28.1% relative to the Phase-1 boundary,
while the text-only objective improved by 24.9% rather than regressing. This satisfies the planned
overall proxy go/no-go criterion. A real DDIM-100 inference with the local LLM2Vec stack also
completed and produced a finite SOMA-77 motion with valid rotations.

These results establish that the reconstructed data, model, losses, curriculum, optimizer, EMA,
checkpoint/resume, export, text encoder, and inference paths work together. They do not establish
TMR R@3, FID, foot-skate, or generated constraint-error parity with the released
`Kimodo-SOMA-SEED-v1.1`; the public benchmark remains the required quality comparison.

## Experiment and data boundary

- Project: `/home/yezitao/PublicWorkspace/yzt/kimodo-reproduction`
- Core10 package: `/home/yezitao/PublicWorkspace/yzt/kimodo-core10-v1`
- Run: `/home/yezitao/PublicWorkspace/yzt/kimodo-validation-runs/core10-from-scratch-20k-10k`
- GPUs: physical devices 0 and 2, two NVIDIA H200 NVL GPUs
- Precision: BF16
- Model: 283,281,777 parameters, 408 state tensors, SOMA-30 369D representation
- Effective global batch: 512 = 2 ranks × 128 local batch × 2 gradient accumulation
- Optimizer: Adam-atan2, learning rate `1e-5`
- EMA: decay 0.995, update every 10 optimizer steps
- Curriculum: 20,000 Phase-1 steps + 10,000 Phase-2 steps

Core10 contains 4,751 motions and 52,592 cached rows totaling 10.001324 hours. Its training split
contains 4,239 motions, 46,796 rows, and 8.989454 hours. Its validation split contains 512 motions,
5,796 rows, and 1.011870 hours. Train/validation `take_name` overlap is exactly zero.

The model consumed 15.36 million effective samples over 30k optimizer steps. Relative to the 46,796
training rows this is about 328 row-equivalent passes, so the run deliberately tests rapid learning
and pipeline behavior, not data-diverse generalization at released-model scale.

## Stage 1: released checkpoint compatibility

Released bundle:

```text
/storage/data/metaiot_data/yzt/kimodo-repro/models/Kimodo-SOMA-SEED-v1.1
```

The released model loaded strictly with all 408 tensors. A fixed forward and backward pass was
finite, and the instantiated model had 283,281,777 parameters. This validates architecture and
state-dict compatibility; it does not validate the reconstructed private training recipe.

## Stage 2: released-checkpoint Phase-2 update smoke

Run:

```text
/home/yezitao/PublicWorkspace/yzt/kimodo-validation-runs/phase2-official-3000-lr5e6
```

The released weights were used to initialize a fresh optimizer and EMA, then updated for 3,000
Phase-2 steps at learning rate `5e-6` and effective global batch 2,048. The fixed denoising objective
fell from 0.123695 to 0.065164.

This result is intentionally classified only as an **update-path smoke test**. The continuation used
the complete public training manifest, which contains the motions later selected for the Core10
validation split. It therefore does not constitute held-out model-quality improvement and must not
be described as outperforming the released model. It proves that gradients, optimizer updates,
constraint conditioning, EMA, save, and load paths are active.

The separate resume-equivalence experiment showed that a 50→100 resumed run differed less from a
continuous 100-step run than two independent continuous 100-step H200 BF16/DDP reruns differed from
each other. Resume therefore passed at numerical tolerance, not bit-exact determinism.

## Stage 3: Core10 from random initialization

### Completion and artifacts

The final checkpoint and portable inference export completed normally:

```text
checkpoints/step-000005000.pt
checkpoints/step-000010000.pt
checkpoints/step-000015000.pt
checkpoints/step-000020000.pt
checkpoints/step-000025000.pt
checkpoints/step-000030000.pt
exports/step-000030000/
```

The final trainer checkpoint SHA-256 is
`cbb79f3800f327af07b537c460ffdc010c74097851768ad4b9483f6f60688513`. The final EMA inference
weights SHA-256 is `65a61815899fc0fa0ea7021756d061dc0e7c4c31260f0fd9cce2e8c16decec84`.

The export was loaded through `load_checkpoint_bundle` without the trainer. It contained 408 finite
tensors, 283,281,777 parameters, and a 369D motion representation. This validates that it is a real
portable inference bundle rather than merely a trainer checkpoint.

### Runtime and interruption record

The initial job was launched in a temporary execution session. That session was reclaimed after
about two hours, after step 6,470, while the last complete checkpoint was step 5,000. Training was
then resumed exactly from step 5,000 in a persistent `tmux` session. The discarded 5,000→6,470 work
was recomputed. The log therefore contains a duplicate range; analysis must use the post-resume
segment for that interval.

Checkpoint effectiveness was independently verified: the original and resumed step-5,010 loss was
identical (`0.33523369995909813`), while steps 5,020 and 5,030 differed by only about `2–3e-5`, within
normal H200 BF16/DDP nondeterminism. The checkpoint contained model, all 408 optimizer states, EMA,
both-rank RNG state, epoch, batch cursor, and micro-step cursor.

Effective compute time was approximately 8.17 hours: about 1.46 hours to the retained step-5,000
checkpoint plus 6.71 hours from resumed step 5,000 to step 30,000. Wall-clock duration was longer
because of the gap before the persistent-session restart.

### Curriculum audit

Phase 1 had exactly zero motion constraints. Mean text-condition dropout was 9.99%, matching the
configured 10% CFG condition dropout.

Across all 1,000 Phase-2 log windows:

| Statistic | Measured | Intended |
| --- | ---: | ---: |
| Constraint-present fraction | 90.049% | 90% |
| Two-pattern fraction | 25.107% | 25% |
| Text condition dropout | 9.972% | 10% |
| Joint text+constraint branch | 81.065% | 81% |
| Constraint-only branch | 8.984% | 9% |
| Text-only branch | 8.963% | 9% |
| Unconditional branch | 0.988% | 1% |

The maximum sparse-keyframe schedule started at one at the Phase-2 boundary and reached 20 near the
end. The model's internal Transformer dropout switches from the Phase-1 value to zero in Phase 2;
the approximately 10% `text_dropout_fraction` above is a separate classifier-free-guidance
conditioning dropout and is expected to remain active.

### Held-out fixed denoising results

All rows below use the same 256 Core10 validation samples, seed, diffusion noise, timesteps, and EMA
selection. The validation motions were not present in the Core10 training split.

| Checkpoint | Text-only total | Phase-2-conditioned total |
| --- | ---: | ---: |
| Random initialization | 7.134681 | 6.877775 |
| Step 5k EMA | 0.447914 | 0.445832 |
| Step 20k EMA | 0.161987 | 0.162267 |
| Step 30k EMA | 0.121654 | 0.116747 |

From step 20k to 30k, the same conditioned objective changed by constraint family as follows:

| Samples containing family | Step 20k | Step 30k | Relative change |
| --- | ---: | ---: | ---: |
| Full-body sparse | 0.294169 | 0.164651 | -44.0% |
| Root sparse | 0.143831 | 0.078919 | -45.1% |
| Foot-contact sparse | 0.168301 | 0.110734 | -34.2% |
| Root dense | 0.113816 | 0.110782 | -2.7% |
| End-effector sparse | 0.105761 | 0.111678 | +5.6% |

The family values are total reconstruction losses on samples containing a family, not direct
generated constraint errors. Consequently, the weak root-dense change and the 5.6% end-effector
increase are warning signals to investigate with generated root/EE constraint metrics, but they do
not by themselves show that constraint following regressed.

For scale only, the released model previously measured 0.123695 on a similar fixed conditioned
proxy, while the Core10 model measured 0.116747. This is **not a valid released-model win**: the two
models use different normalization statistics and training exposure, and fixed denoising loss is not
the public generation benchmark.

### Real inference sanity check

The final portable export was paired with the locally pinned Llama-3 LLM2Vec foundation, MNTP
adapter, and supervised adapter. DDIM-100 generation with separated CFG `[2, 2]`, no postprocessing,
and prompt `A person walks forward naturally and waves with the right hand.` produced:

```text
exports/step-000030000/validation_samples/walk_and_wave_seed20260806.npz
```

- output shape: 120 frames × 77 SOMA joints × 3 coordinates;
- all arrays finite;
- local-rotation orthogonality maximum error: `9.54e-7`;
- four-second root displacement: 4.50 m;
- horizontal root path length: 4.54 m;
- maximum per-frame root step: 0.0556 m;
- predicted foot-contact fraction: 44.9%.

The root path is physically plausible for walking and the right wrist had more relative motion than
the left in this single sample. A single prompt cannot establish text alignment or visual quality;
the NPZ should be viewed in the Kimodo demo and followed by TMR/public-benchmark evaluation.

## What is established and what remains

Established with high confidence:

1. The released architecture and weights are compatible with the reconstructed model.
2. Core10 is train/validation disjoint at `take_name` and enters the normal cached-data pipeline.
3. Random initialization learns strongly on held-out motions.
4. Phase 2 improves the overall held-out conditioned proxy without degrading the text-only proxy.
5. Constraint/CFG sampling frequencies and sparse-keyframe curriculum match the encoded strategy.
6. Exact-state checkpoint resume, EMA export, local LLM2Vec loading, and DDIM-100 inference work.

Not established:

1. Public `content`/`repetition` TMR R@3 and FID parity.
2. Generated full-body, end-effector, root, and pelvis constraint-error parity.
3. Foot-skate/contact parity on the public benchmark.
4. Equivalence to NVIDIA's undisclosed data augmentation and private trainer details.
5. Released-model quality from only 10 hours of data and 30k steps.

The next decisive evaluation is a stratified public-benchmark proxy followed, if it passes, by the
full 22,474-case public suite. Both this model and `Kimodo-SOMA-SEED-v1.1` must use identical seeds,
DDIM-100, batch size one, separated CFG `[2,2]`, no postprocessing, and the same LLM2Vec/TMR
precision. Direct generated constraint metrics, especially end-effector and dense-root cases, are
the first priority.
