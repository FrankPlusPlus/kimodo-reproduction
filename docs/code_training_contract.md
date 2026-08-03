# Kimodo upstream public-code training contract (historical audit)

Status: **historical upstream audit; superseded for current reproduction behavior**  
Scope: the pristine NVIDIA public snapshot examined before `kimodo/training/` was implemented. Existing inference behavior remains a compatibility boundary, but current paper-method decisions and acceptance status are authoritative only in `paper_training_parity_audit.md` and `training_reproduction_spec.md`.

## 1. Executive conclusion

The pristine upstream repository contained the **model architecture and inference-time mathematical primitives**, but not an end-to-end training or post-training implementation. The current reproduction tree now adds a clean-room `kimodo/training/` implementation; it is not NVIDIA's original trainer.

In particular, it exposes:

- the two-stage root/body denoiser (`kimodo/model/twostage_denoiser.py`);
- the transformer backbone and its mask convention (`kimodo/model/backbone.py`);
- the forward noising operation `Diffusion.q_sample` and the cosine schedule (`kimodo/model/diffusion.py`);
- motion feature construction, normalization, skeleton assets, and constraint conversion (`kimodo/motion_rep`, `kimodo/skeleton`, `kimodo/constraints.py`);
- frozen text encoding and inference-time classifier-free guidance (`kimodo/model/llm2vec`, `kimodo/model/cfg.py`);
- checkpoint/config loading and benchmark evaluation.

The pristine upstream snapshot did **not** expose a training dataset, training collator, loss, optimizer construction, update loop, EMA, curriculum/phase controller, constraint sampler for training, resume checkpoint writer, or training CLI. Those components now exist only as this project's clean-room reconstruction. Therefore the upstream `model/` directory means that the forward network was open, not that NVIDIA's training script was open. The public artifacts still cannot by themselves prove undisclosed private-trainer details.

Reproducible repository check:

```bash
cd kimodo-reproduction
rg -n "class .*Trainer|configure_optimizers|optimizer\.step|loss\.backward|manual_backward" kimodo benchmark
find kimodo -maxdepth 2 -type d | sort
```

The first command has no Kimodo training-loop implementation. `benchmark/generate_eval.py` has the only public `Dataset`/`DataLoader`, and it is explicitly an evaluation loader (`benchmark/generate_eval.py:4-8,107-159`).

## 2. Compatibility boundary

The implementation may add a new `kimodo/training/` package and a new CLI, but it must not change these public contracts:

1. `kimodo.model.load_model(...)` continues to load existing Hugging Face and local inference bundles (`kimodo/model/load_model.py:108-145`).
2. Existing model short names and version resolution remain valid (`kimodo/model/registry.py:13-26,91-137,359-478`).
3. `Kimodo.generate`, `_generate`, and the four existing console scripts keep their current signatures and semantics.
4. Existing checkpoint formats remain readable: raw PyTorch state dict, a dict containing `state_dict`, and safetensors (`kimodo/model/loading.py:44-68`).
5. Existing motion/constraint JSON and NPZ formats remain accepted.

The trainer must operate on a **bare `TwostageDenoiser`**, not on `Kimodo.denoising_step` and not on `ClassifierFreeGuidedModel`. `Kimodo` describes itself as a test-time helper, wraps the denoiser for CFG, and executes denoising under `torch.inference_mode()` (`kimodo/model/kimodo_model.py:25-53,75-120`). Training through that path cannot work correctly.

## 3. Model construction and config contract

### 3.1 Canonical object graph

The inference bundle's `config.yaml` is a recursively-instantiated Hydra object graph. Every mapping with `_target_` is passed to Hydra (`kimodo/model/loading.py:29-41`), and `load_model` merges runtime `checkpoint_dir` and text-encoder settings before instantiation (`kimodo/model/load_model.py:177-206`). A trainer should construct the same nested objects:

```text
Kimodo (inference wrapper only)
└── TwostageDenoiser (train this module)
    ├── KimodoMotionRep
    │   └── one supported SkeletonBase subclass
    ├── root_model: TransformerEncoderBlock
    └── body_model: TransformerEncoderBlock
```

The supported skeleton constructors are `SOMASkeleton30`, `SOMASkeleton77`, `SMPLXSkeleton22`, and `G1Skeleton34`; joint-count dispatch is fixed at 30/77/22/34 (`kimodo/skeleton/registry.py:17-42`). Packaged skeleton assets resolve relative to the installed package (`kimodo/assets.py:4-15`) and are included by `pyproject.toml:53-58`. Skeleton tensor buffers are non-persistent and reloaded from those assets (`kimodo/skeleton/base.py:38-99`), so they must not be treated as learned checkpoint parameters.

### 3.2 Denoiser dimensions

`TwostageDenoiser` derives every input/output dimension from the motion representation (`kimodo/model/twostage_denoiser.py:18-61`):

- full feature width `D = motion_rep.motion_rep_dim`;
- root output width `D_root = 5` for `KimodoMotionRep`;
- local-root width `D_local_root = 4`;
- body output width `D - 5`;
- with `motion_mask_mode == "concat"`, root input width is `2D`, and the body input also appends the full `D`-wide mask.

The transformer config is validated with unknown keys forbidden (`kimodo/model/backbone.py:60-99`, via `kimodo/tools.py:19-50`). A reproduction config must preserve at least:

```text
llm_shape, use_text_mask, latent_dim, ff_size, num_layers,
num_heads, activation, dropout, pe_dropout, norm_first,
num_text_tokens_override, input_first_heading_angle,
motion_mask_mode, num_base_steps, fps, skeleton, stats_path
```

Do not infer architectural values from checkpoint tensor shapes when an official `config.yaml` is available. For a local bundle, `load_model` requires `<bundle>/config.yaml` (`kimodo/model/load_model.py:177-181`).

### 3.3 Checkpoint key contract

`TwostageDenoiser.load_ckpt` loads strictly after removing the literal prefix `denoiser.backbone.` from every key (`kimodo/model/twostage_denoiser.py:63-71`). The canonical inference export should therefore contain the raw denoiser state dict with keys rooted at:

```text
root_model.*
body_model.*
```

Legacy official-style keys beginning `denoiser.backbone.root_model.*` / `denoiser.backbone.body_model.*` are also accepted by that loader. A full wrapper state dict rooted at `denoiser.model.*` is **not** compatible with this function and must not be exported as the inference weights without an explicit conversion test.

Recommended separation:

- `trainer_state.pt`: resume-only state (raw denoiser, EMA denoiser, optimizer, scheduler, scaler, RNG, step, phase, resolved config, data/stats fingerprints);
- `denoiser.safetensors`: inference-only EMA weights using raw `TwostageDenoiser.state_dict()` names;
- `config.yaml` plus split stats: inference construction.

Diffusion, stats, and skeleton buffers are declared non-persistent (`kimodo/model/diffusion.py:32-41,60-94`; `kimodo/motion_rep/stats.py:60-63`; `kimodo/skeleton/base.py:70-99`). Reproducing them depends on config/assets, not on `state_dict` contents.

### 3.4 Bundle acceptance test

Every exported bundle must pass a network-free round trip:

```bash
CHECKPOINT_DIR=/absolute/path/to/export-root python - <<'PY'
from kimodo.model import load_model
m = load_model("kimodo-soma-seed-v1.1", device="cpu", text_encoder=object())
print(type(m).__name__, m.motion_rep.motion_rep_dim, m.fps)
PY
```

The export root must contain either the registry display folder (for example `Kimodo-SOMA-SEED-v1.1`) or its short-key fallback (`kimodo/model/load_model.py:159-175`). The test should normally inject a stub conforming text encoder rather than `object()` when running generation; the snippet only checks construction.

## 4. Motion representation and statistics

### 4.1 Exact feature order

`KimodoMotionRep` packs features in this immutable order (`kimodo/motion_rep/reps/kimodo_motionrep.py:23-48,94-104`):

| Block | Width |
|---|---:|
| `smooth_root_pos` | 3 |
| `global_root_heading` | 2 |
| `local_joints_positions` | `3J` |
| `global_rot_data` (continuous 6D) | `6J` |
| `velocities` | `3J` |
| `foot_contacts` | 4 |

Thus `D = 12J + 9`: SOMA-30 is 369, G1-34 is 417, and SMPLX-22 is 273. The global-root slice is the first 5 channels. The body slice starts at channel 5. No trainer, loss, stats builder, or constraint sampler may reorder these fields.

Feature construction requires local rotation matrices `[B,T,J,3,3]`, root positions `[B,T,3]`, and valid lengths (`kimodo/motion_rep/reps/kimodo_motionrep.py:50-111`). It performs FK, smooth-root extraction, heading calculation, velocities, contacts, and 6D rotation conversion. Batched training must always pass `lengths`; the velocity helpers' fallback is intended only for unbatched input and uses fragile length inference (`kimodo/motion_rep/feature_utils.py:38-72,75-108`). Enforce `length >= 2` because both helpers copy index `length-2` into the last valid frame.

### 4.2 Required split-stats layout

An inference-compatible stats directory is:

```text
stats/
├── global_root/mean.npy   # [5]
├── global_root/std.npy    # [5]
├── local_root/mean.npy    # [4]
├── local_root/std.npy     # [4]
├── body/mean.npy          # [D-5]
└── body/std.npy           # [D-5]
```

The loader requires these subdirectories and reconstructs full global-feature stats by concatenating global-root then body (`kimodo/motion_rep/reps/base.py:19-33,72-83`). Save float32 arrays in exact feature order. `std.npy` is a standard deviation, not a variance; normalization is

```text
(x - mean) / sqrt(std**2 + 1e-5)
```

(`kimodo/motion_rep/stats.py:15-20,43-77`). Stats must be fitted on the finalized training split only, after the exact motion preprocessing chosen for training, then frozen. Include a SHA-256 fingerprint of all six files in every resume checkpoint and run manifest.

The local-root stats are not a slice of the global representation. They are computed from predicted/target global root as `[angular_velocity, planar_velocity_xz, root_height]` and normalized separately (`kimodo/motion_rep/reps/base.py:113-157`).

## 5. Text encoder contract

The public local encoder is intentionally frozen: it calls `.eval()`, sets every parameter's `requires_grad=False`, and encodes under `torch.no_grad()` (`kimodo/model/llm2vec/llm2vec_wrapper.py:49-51,65-82`). It returns `(features, lengths)`, where local LLM2Vec produces `[B,1,4096]` and lengths of 1 (`kimodo/model/llm2vec/llm2vec_wrapper.py:84-95`). The remote API permits variable token length and pads to `[B,L,E]` (`kimodo/model/text_encoder_api.py:38-73`).

Trainer rules:

1. Text-encoder parameters are never added to the optimizer or trainer checkpoint.
2. The provider interface remains general `[B,L,E]`; do not hard-code `L=1`.
3. `E` must equal `root_model.llm_shape[-1]` and `body_model.llm_shape[-1]`.
4. For an empty prompt or classifier-free text dropout, zero the embedding tensor **and set its valid length to zero**, matching inference `_generate` behavior rather than treating the local encoder's empty-string result as a valid token (`kimodo/model/kimodo_model.py:584-600`).
5. Precompute/cache sanitized text embeddings with encoder repo IDs, revisions, dtype, sanitizer version, and content hash in the cache key.
6. For v1.1 reproduction, use float32 text embeddings. The public benchmark documentation says v1.1 was trained with float32 encodings (`docs/source/benchmark/pipeline.md:76-83`; also `docs/source/benchmark/results.md:7`). The local loader otherwise defaults to bfloat16 (`kimodo/model/load_model.py:22-34`).

The local wrapper deliberately fixes its internal batch size to 1 because batch size and low precision alter embeddings (`kimodo/model/llm2vec/llm2vec_wrapper.py:71-80`). Cached features must be created with the same rule.

## 6. Diffusion training contract

`Diffusion` defines the cosine schedule (`kimodo/model/diffusion.py:12-41`) and the public forward process (`kimodo/model/diffusion.py:96-110`):

```python
noise = torch.randn_like(x0)
t = torch.randint(0, diffusion.num_base_steps, (B,), device=x0.device, dtype=torch.long)
xt = diffusion.q_sample(x0, t, noise)
x0_pred = denoiser(
    xt, motion_valid, text_feat, text_valid, t,
    first_heading_angle, feature_constraint_mask, observed_motion,
)
```

The denoiser predicts **clean `x0`**, not noise `epsilon` and not velocity `v`. This is also the quantity passed into the DDIM sampler as `pred_xstart` (`kimodo/model/diffusion.py:113-133`; `kimodo/model/kimodo_model.py:102-120`). Any loss adapter must therefore receive `pred_x0` and `target_x0` explicitly.

Freeze these tensor contracts:

- `x0`, `xt`, `noise`, `pred_x0`: `[B,T,D]`, floating point;
- `t`: `[B]`, `torch.long`, same device as the diffusion buffers;
- `0 <= t < num_base_steps`;
- `motion_valid`: `[B,T]`, bool, **True means valid**;
- `text_valid`: `[B,L]`, bool, **True means valid**;
- `feature_constraint_mask`: `[B,T,D]`, bool, **True means observed/constrained**.

The apparently unusual reciprocal-square-root expressions at `diffusion.py:86-93` algebraically evaluate to `sqrt(alpha_bar)` and `sqrt(1-alpha_bar)`; do not “correct” their names by changing numerical behavior.

## 7. Padding and masks

The code uses positive validity masks. Transformer internals invert the concatenated prefix/motion mask only immediately before `src_key_padding_mask` (`kimodo/model/backbone.py:136-154,172-186,210-229`). `length_to_mask` also returns True for valid frames (`kimodo/motion_rep/feature_utils.py:129-162`).

Collation requirements:

- crop every sequence to the configured maximum before batching;
- pad feature values with zero only after normalization/feature construction;
- generate `motion_valid = arange(T) < lengths[:,None]`;
- set loss weight to zero on invalid frames for every loss term;
- keep padded constraints false and padded text tokens invalid;
- pass a Python integer as `max_len` to `length_to_mask`.

Do not rely on `pad_x_and_mask_to_fixed_size` to crop. Its over-length branch changes `cur_max_size` but assigns the still-long `x`/`mask` into a shorter slice (`kimodo/model/backbone.py:31-56`). Also inspect `use_text_mask`: when false, the backbone forces all padded text slots valid (`kimodo/model/backbone.py:178-184`). Reproduction must take this value from the official config and ensure padded embeddings are deterministic zeros.

## 8. Constraint to `(observed_motion, mask)` contract

The public conversion path is:

```text
ConstraintSet.update_constraints
  -> build_condition_dicts
  -> KimodoMotionRep.create_conditions
  -> observed_motion [T,D] + motion_mask [T,D]
```

Evidence: `kimodo/motion_rep/conditioning.py:10-28`, `kimodo/motion_rep/reps/base.py:251-299`, and `kimodo/motion_rep/reps/kimodo_motionrep.py:222-306`.

Semantics:

- `observed_motion` is normalized in the same full representation as `x0` when `to_normalize=True`;
- `motion_mask` is bool and selects which observed channels are meaningful;
- unselected values in `observed_motion` are allowed to be nonzero after normalization and must always be ignored through the mask;
- an unconstrained sample is represented by all-false mask and an arbitrary (preferably zero/normalized-zero) observation tensor;
- the denoiser in `concat` mode first imputes constrained values into noisy `x`, then concatenates the full mask (`kimodo/model/twostage_denoiser.py:98-104,132-139`).

Specific public semantics that a training sampler must match:

- root 2D fills only smooth-root X and Z (`kimodo/motion_rep/reps/kimodo_motionrep.py:242-251`);
- heading fills its two cosine/sine channels (`kimodo/motion_rep/reps/kimodo_motionrep.py:262-269`);
- rotation constraints convert matrices to continuous 6D for selected joints (`kimodo/motion_rep/reps/kimodo_motionrep.py:271-282`);
- global joint position constraints require root XZ at the same frames and are expressed relative to smooth root (`kimodo/motion_rep/reps/kimodo_motionrep.py:284-302`);
- `FullBodyConstraintSet` contributes all joint positions, root XZ/Y, and heading, but explicitly not global rotations (`kimodo/constraints.py:233-259`);
- end-effector constraints contribute the expanded selected position/rotation chains plus root information (`kimodo/constraints.py:387-427`).

For performance, a training constraint sampler may generate the normalized tensors directly from ground truth, but every family must have a parity test against the public conversion path.

## 9. Two-stage forward and `self.training`

The root network first predicts normalized global root. That prediction is converted to normalized local-root features and fed to the body network. The initial upstream snapshot detached this bridge in training mode, but the paper explicitly describes end-to-end training. The current reconstruction therefore exposes `detach_root_for_body`:

- paper profile: `detach_root_for_body=false`; body loss can update `root_model` through the local-root bridge;
- upstream-compatibility ablation: `detach_root_for_body=true`; conversion is detached during training;
- eval mode keeps the bridge differentiable for guidance.

Consequences:

1. The update step must call `denoiser.train()` before every training phase/epoch.
2. Validation that calls `.eval()` must restore `.train()` afterward.
3. A gradient-partition test is mandatory for both branches: paper mode must pass body-loss gradients into root; compatibility mode must block that bridge.
4. Never substitute ground-truth local-root features into the body stage unless the experiment is explicitly a non-compatible ablation. The public forward uses the predicted root.

## 10. Required new modules and proposed interfaces

These were proposed additions during the upstream audit. They now exist in the current reproduction under `kimodo/training/`; this section is retained only as historical design context.

### 10.1 Data and batch types

```python
@dataclass(frozen=True)
class MotionSample:
    sample_id: str
    local_rot_mats: torch.Tensor   # [T,J,3,3], float32
    root_positions: torch.Tensor   # [T,3], float32
    text: str
    length: int

@dataclass
class TrainBatch:
    sample_ids: list[str]
    x0: torch.Tensor               # [B,T,D], normalized
    lengths: torch.Tensor          # [B], long
    motion_valid: torch.Tensor      # [B,T], bool; True=valid
    text_feat: torch.Tensor         # [B,L,E], preferably fp32
    text_valid: torch.Tensor        # [B,L], bool; True=valid
    first_heading_angle: torch.Tensor | None  # [B]

def collate_motion_batch(
    samples: list[MotionSample], *, motion_rep, text_provider,
    max_frames: int,
) -> TrainBatch: ...
```

Workers should parse/crop CPU data. Device transfer belongs in the trainer step, because public evaluation also avoids skeleton/CUDA work in DataLoader workers (`benchmark/generate_eval.py:107-110,316-320`).

### 10.2 Frozen text provider

```python
class TextFeatureProvider(Protocol):
    def encode(self, texts: list[str]) -> tuple[torch.Tensor, list[int]]: ...
    # output [B,L,E] and valid lengths; never participates in autograd
```

Provide implementations for a precomputed cache and the public LLM2Vec/API adapters. The training loop consumes only this protocol.

### 10.3 Constraint sampler

```python
@dataclass
class ConditioningBatch:
    observed_motion: torch.Tensor  # [B,T,D], same dtype/device as x0
    motion_mask: torch.Tensor       # [B,T,D], bool
    family: list[str]
    metadata: list[dict]

class ConstraintSampler(Protocol):
    def sample(
        self, batch: TrainBatch, *, global_step: int,
        generator: torch.Generator,
    ) -> ConditioningBatch: ...
```

The phase/curriculum policy belongs outside `TwostageDenoiser`; the denoiser interface already supports constrained and unconstrained batches.

### 10.4 Diffusion step and loss adapter

```python
@dataclass
class DenoisingBatch:
    x0: torch.Tensor
    xt: torch.Tensor
    noise: torch.Tensor
    timesteps: torch.Tensor
    pred_x0: torch.Tensor
    valid: torch.Tensor
    conditioning: ConditioningBatch

def diffusion_train_forward(
    denoiser, diffusion, batch: TrainBatch,
    conditioning: ConditioningBatch, *, generator: torch.Generator,
) -> DenoisingBatch: ...

class KimodoLoss(Protocol):
    def __call__(
        self, denoising: DenoisingBatch, *, motion_rep,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...
```

Naming `pred_x0`/`x0` in the interface prevents accidental epsilon-target training. The exact paper loss formulation and weighting are experiment configuration, not denoiser behavior.

### 10.5 Trainer, stats, and export

```python
class StatsBuilder:
    def update(self, unnormalized_features, lengths): ...
    def finalize(self) -> dict[str, tuple[Tensor, Tensor]]: ...
    def save_split_layout(self, output_dir: Path): ...

class KimodoTrainer:
    def train_step(self, batch: TrainBatch) -> dict[str, Tensor]: ...
    def validate(self) -> dict[str, float]: ...
    def save_resume(self, path: Path): ...
    def resume(self, path: Path): ...

class InferenceBundleExporter:
    def export(
        self, *, ema_denoiser, hydra_config, stats_dir: Path,
        output_dir: Path,
    ) -> Path: ...
```

The exporter must validate strict reload through `TwostageDenoiser.load_ckpt`, then validate the unchanged `load_model` path.

## 11. Registry and asset rules

The registry is generated from a canonical list of Hugging Face repository IDs (`kimodo/model/registry.py:13-26,91-129`). Versionless aliases resolve to the latest registered version (`registry.py:123-127,359-365`). Training and experiment manifests must therefore use an explicit target such as `kimodo-soma-seed-v1.1`, never the floating `kimodo-soma-seed` or default `kimodo-soma-rp` alias.

Do not add an experimental checkpoint to the public registry just to make local loading work. Export it into an existing explicit model folder under `CHECKPOINT_DIR` for compatibility testing, or add a separate explicit local-config CLI path. A registry addition implies a real downloadable model and stable metadata.

The model skeleton for SOMA prediction is normally SOMA-30 and inference expands outputs to SOMA-77 (`kimodo/model/kimodo_model.py:60-65`; `kimodo/skeleton/definitions.py:239-283`). Dataset ingestion must perform any 77-to-30 mapping before motion-feature construction and record the mapping revision. The benchmark output boundary remains SOMA-77.

## 12. Benchmark boundary

The `benchmark/` package is evaluation-only:

1. build held-out test cases;
2. generate outputs;
3. embed with pretrained TMR;
4. evaluate metrics;
5. aggregate results (`docs/source/benchmark/pipeline.md:3-11`).

It is not a training-data implementation. The benchmark docs define a train split and two held-out test splits and explicitly require training only on `train_split_paths.txt` (`docs/source/benchmark/introduction.md:13-22`). `create_benchmark.py` labels itself evaluation step 1, is SOMA-only, and writes `gt_motion.npz`/constraints for test cases (`benchmark/create_benchmark.py:4-9,56-128`). Do not ingest those generated test folders into training or stats.

Acceptance evaluation should:

- use the public split manifests;
- keep `content` and `repetition` fully held out;
- evaluate raw model output without post-processing for fair motion metrics (`docs/source/benchmark/pipeline.md:76-83`);
- use batch size 1 for exact seed reproducibility (`docs/source/benchmark/pipeline.md:72-74`);
- output SOMA-77 `posed_joints` and `foot_contacts` at minimum (`docs/source/benchmark/pipeline.md:96-98`).

The public benchmark is not identical to the paper's test suite (`docs/source/benchmark/introduction.md:7-9`). Passing it establishes public-code compatibility, not exact paper-number reproduction.

## 13. `pyproject.toml` and CLI contract

The base dependencies are inference-oriented and do not even list `torch` explicitly (`pyproject.toml:12-33`). Optional extras currently cover only demo/SOMA integrations (`pyproject.toml:35-45`). Existing console entry points are exactly:

```text
kimodo_gen
kimodo_demo
kimodo_textencoder
kimodo_convert
```

(`pyproject.toml:47-51`).

Recommended additive packaging:

```toml
[project.optional-dependencies]
train = [
  # pin after the optimizer/launcher decisions are resolved
  "safetensors>=...",
  "tensorboard>=...",       # or the selected logger, not both by accident
]

[project.scripts]
kimodo_train = "kimodo.training.cli:main"
kimodo_export = "kimodo.training.export_cli:main"
```

Do not alter the four existing entry points. The training CLI should require explicit `--config`, `--model-target` (including version), `--train-split`, `--stats`, and `--output`. It should provide `--resume`, `--seed`, `--precision`, `--num-workers`, and a dry-run/one-step validation mode. Resolve paths to absolute paths in the saved manifest. A CLI default must never download proprietary RP data or silently fall back from SEED to RP.

## 14. Compatibility traps

### P0 — blocks a valid reproduction

1. **Wrong diffusion target:** public denoiser predicts `x0`, not epsilon.
2. **Wrong mode/wrapper:** training `Kimodo`/CFG or leaving the bare denoiser in eval mode changes gradient semantics or disables gradients.
3. **Mask polarity inversion:** all public input masks use True=valid/observed.
4. **Stats mismatch:** wrong feature order, unsplit stats, variance saved as std, non-fp32 stats, or test data included invalidates every input and output.
5. **Checkpoint prefix/config mismatch:** wrapper keys or a non-Hydra config cannot be loaded by existing inference APIs.
6. **Text incompatibility:** fine-tuning LLM2Vec, treating dropped text as one valid empty token, or using the wrong v1.1 embedding precision changes conditioning.
7. **Root/body gradient-policy confusion:** paper profile must remain end-to-end (`detach=false`), while `detach=true` is only an explicitly labeled upstream-compatibility ablation. Silently switching either policy changes the trained model.

### P1 — likely silent degradation or runtime failure

1. `MotionRepBase.randomize_first_heading` creates its random target on CPU (`kimodo/motion_rep/reps/base.py:192-204`); a GPU augmentation should generate it on `features.device` instead of calling this method unchanged.
2. Constraint constructors interpret a 3D `smooth_root_2d` via indices `[0,1]`, although the rest of the code uses XZ (`kimodo/constraints.py:91-93,199-201,354-356`). Always pass an explicit `[K,2]` XZ tensor.
3. `ConstraintSet.to(dtype=...)` casts index tensors as well as values (`kimodo/constraints.py:151-159,288-298,468-480`). Keep `frame_indices`, `pos_indices`, and `rot_indices` as `torch.long`; load with `dtype=None`.
4. Duplicate constraint resolution claims “first” but uses `scatter_`, so later duplicates can win (`kimodo/motion_rep/conditioning.py:18-28`). Generate unique `(frame,joint)` keys or define/test precedence.
5. Backbone fixed-size cropping is unsafe; crop in collate.
6. `use_text_mask=False` marks padded text slots valid; ensure deterministic zero padding and preserve the official config.
7. Variable-length helpers require at least two valid frames and explicit lengths.
8. Versionless model aliases float to the newest registry entry; record an explicit version.

## 15. Minimum test suite

These tests are required before long training:

1. **Feature-layout test:** instantiate J=22/30/34 reps; assert `D=12J+9`, root width 5, local-root width 4, and exact slices.
2. **Stats round trip:** fit/save/load all six float32 arrays; `unnormalize(normalize(x)) ~= x`; verify file hashes and dimensions.
3. **Motion conversion test:** known rotation/root sequence to normalized features and inverse; padded batch equals per-sample valid prefixes; length-1 input is rejected.
4. **Diffusion test:** fixed generator makes `q_sample` deterministic; manual formula equals implementation; `t` shape/dtype validation; training target is `x0`.
5. **Forward-shape test:** constrained and unconstrained calls return `[B,T,D]` for every supported predictor skeleton.
6. **Gradient-partition test:** in paper mode body loss reaches `root_model`; in `detach=true` compatibility mode it cannot; eval-mode bridge remains differentiable outside inference mode.
7. **Mask test:** corrupting padded motion/text values does not alter valid predictions under the chosen official config; all loss terms are zero on padding.
8. **Text test:** provider returns `[B,L,E]`; local encoder is frozen; dropped/empty text has zero features and zero valid length; cache key changes with revision/dtype/sanitizer.
9. **Constraint parity tests:** root-only, root+heading, full-body, and each end-effector family; compare direct sampler tensors bit-for-bit/tolerance with `create_conditions_from_constraints_batched`; verify no constraints.
10. **Constraint dtype test:** all indices remain long after device transfer; explicit XZ is preserved; duplicate keys are rejected or deterministically resolved.
11. **Checkpoint tests:** strict raw state reload; legacy `denoiser.backbone.` prefix reload; reject wrapper prefixes; resume restores optimizer/EMA/step/phase/RNG exactly.
12. **Inference-bundle smoke test:** existing `load_model` loads the exported EMA bundle under `CHECKPOINT_DIR`; one short generation completes with the public API.
13. **Registry test:** explicit version stays fixed while versionless alias resolves as documented.
14. **CLI regression:** the four old `--help` commands still work; `kimodo_train --help`, one-step dry run, resume, and export work.
15. **Benchmark smoke test:** run a tiny held-out subset, no post-processing, batch size 1, and validate the required SOMA-77 NPZ fields.

## 16. Blocking questions (maximum three)

1. **Resolved after this historical audit:** the compatibility oracle is frozen to `Kimodo-SOMA-SEED-v1.1`; its config, weights and stats are fingerprinted, and strict-load/forward/backward tests pass. This resolves artifact compatibility only, not undisclosed NVIDIA training details.
2. **What is the authoritative training recipe for losses and phase transition?** Public code has no loss, curriculum, optimizer, EMA, or update schedule. The implementer needs one resolved spec for component losses/weights and their normalized versus physical space, constraint/text dropout distributions, phase boundary, Adam-atan2 implementation, learning-rate schedule, and EMA decay.
3. **What is the canonical BONES-SEED ingestion/annotation manifest?** Freeze exact dataset revision, train split revision, SOMA-77→30 conversion, crop/window policy, prompt/timeline sampling, and exclusions before computing stats. Otherwise two implementations can satisfy the model API but train on materially different examples.

## 17. Evidence commands

Run from `kimodo-reproduction`:

```bash
# Historical upstream training-loop absence / inference primitives.
# Run the first command only in a pristine upstream checkout; the current
# reproduction intentionally contains kimodo/training and will return matches.
rg -n "class .*Trainer|configure_optimizers|optimizer\.step|loss\.backward|manual_backward" kimodo benchmark
rg -n "def q_sample|class TwostageDenoiser|torch\.inference_mode|requires_grad = False" kimodo/model

# Public API and package contract
nl -ba pyproject.toml | sed -n '1,90p'
nl -ba kimodo/model/load_model.py | sed -n '108,215p'
nl -ba kimodo/model/twostage_denoiser.py | sed -n '18,153p'
nl -ba kimodo/model/diffusion.py | sed -n '12,133p'

# Representation, stats, masks, and constraints
nl -ba kimodo/motion_rep/reps/kimodo_motionrep.py | sed -n '23,111p'
nl -ba kimodo/motion_rep/reps/kimodo_motionrep.py | sed -n '222,306p'
nl -ba kimodo/motion_rep/reps/base.py | sed -n '19,157p'
nl -ba kimodo/motion_rep/stats.py | sed -n '15,110p'
nl -ba kimodo/motion_rep/feature_utils.py | sed -n '38,162p'
nl -ba kimodo/constraints.py | sed -n '75,592p'

# Benchmark split/evaluation boundary
nl -ba docs/source/benchmark/introduction.md | sed -n '3,27p'
nl -ba docs/source/benchmark/pipeline.md | sed -n '3,110p'
```

This contract is intentionally sufficient for adding a trainer without altering existing inference interfaces. It is not a claim that exact paper training is reproducible until the three blocking inputs above are frozen.
