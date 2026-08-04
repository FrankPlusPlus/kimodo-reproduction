# Pinned resource acquisition

This directory is the network/resource layer for the public BONES-SEED
engineering reconstruction. It deliberately does not contain training
hyperparameters and does not claim to recreate the paper's unpublished Qwen
prompt, mixture, or diffusion-transition recipe.

The configuration is split into two parts:

- `catalog.public.yaml` is committed and immutable in spirit: Hugging Face repo
  IDs, commit revisions, required file paths, sizes, and SHA-256 digests.
- a paths YAML is machine-local: managed download destinations or
  `existing_path` locations on shared storage. Copy `paths.example.yaml` to
  `paths.local.yaml` and edit it. `paths.local.yaml` is ignored by Git.

Set `existing_path` to reuse a server snapshot without copying it. `fetch`
fully hashes such a snapshot and never modifies it. Leave `existing_path: null`
to let the tool download into `destination`. Authentication is read by
`huggingface_hub` from its normal token store/environment; tokens are not part
of either YAML schema and are never printed by this tool.

## Fresh-clone flow

```bash
scripts/resources/setup_env.sh \
  --flowmatching-repo /path/to/kimodo-flowmatching
cp resources/paths.example.yaml resources/paths.local.yaml
# Edit resources/paths.local.yaml.

scripts/resources/resources.sh \
  --paths resources/paths.local.yaml plan
scripts/resources/resources.sh \
  --paths resources/paths.local.yaml fetch
scripts/resources/resources.sh \
  --paths resources/paths.local.yaml verify
scripts/resources/resources.sh \
  --paths resources/paths.local.yaml prepare
```

The training/resource environment skips the optional MotionCorrection C++
postprocessor, so a fresh server does not need CMake or a compiler. Add
`--with-motion-correction` only for inference postprocessing; the setup script
then checks both build tools before installing.

The default group is `train-minimal`. It downloads BONES-SEED SOMA Uniform,
the official train split, and the three LLM2Vec components. It does **not**
download Qwen3-32B or the official Kimodo checkpoint.

Optional groups must be requested explicitly:

```bash
# Compatibility oracle only; not a resumable training checkpoint.
scripts/resources/resources.sh --paths resources/paths.local.yaml \
  fetch --group official-oracle

# Research tool only; not required for DDPM/FM training.
scripts/resources/resources.sh --paths resources/paths.local.yaml \
  fetch --group paper-exploration
```

`plan` is intentionally fast and checks presence/known sizes. `verify` and
`fetch` calculate full SHA-256 digests. Hugging Face handles partial-transfer
resumption in each managed destination. A verified receipt is written below
`.cache/kimodo/`, outside the model-content identity used by text caching.

`fetch` only acquires pinned source files. `prepare` then performs safe archive
extraction, SOMA77/120 Hz to SOMA30/30 Hz conversion, portable raw/cached
manifest construction, offline LLM2Vec caching, normalization statistics and
the portable reference inventory. It writes the resolved repro training paths
YAML configured by `pipeline.repro_paths_yaml` and a `resource-state.json`
receipt below the prepared root. Re-running validates and reuses complete
stages; partial output pairs fail closed. Before writing `repro_train_ready`,
the pipeline fully re-hashes the portable reference inventory and validates all
six statistics arrays. This is intentionally slower than `plan` but prevents a
missing/corrupt derived asset from being reported as trainable.

The canonical motion adapter is shared with the FM project, so the environment
setup accepts an FM checkout at any path. This is an explicit preprocessing-only
dependency; repro and FM training do not require adjacent clones or a shared
virtual environment. The combined shorthand is:

```bash
scripts/resources/resources.sh --paths resources/paths.local.yaml all
```
