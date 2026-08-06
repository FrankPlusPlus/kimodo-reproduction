# Documentation

## Training reconstruction

The training implementation is a clean-room reconstruction layered on the released
Kimodo inference code. Read these documents in this order:

1. [`data_pipeline_flow.md`](data_pipeline_flow.md): visual, artifact-by-artifact path
   from BONES-SEED downloads and BVH conversion through LLM2Vec and the final batch.
2. [`two_stage_training_flow.md`](two_stage_training_flow.md): tensor-level visual
   walkthrough of the root/body Transformers, diffusion objective, losses and updates,
   including a standalone 3600 px model-architecture SVG with every dimension change.
3. [`portable_training_setup.md`](portable_training_setup.md): one-command fresh
   setup, verified legacy-cache adoption, relocated-bundle binding, and training.
4. [`reproduction_end_to_end_guide.md`](reproduction_end_to_end_guide.md): complete
   resource, data, model, training, artifact and paper-alignment walkthrough.
5. [`training_reproduction_runbook.md`](training_reproduction_runbook.md): operational
   commands for a fresh server and two-H200 training.
6. [`paper_training_parity_audit.md`](paper_training_parity_audit.md): clause-by-clause
   paper gate and known unavailable inputs.
7. [`training_reproduction_spec.md`](training_reproduction_spec.md): detailed
   `PAPER`/`CODE`/`RECONSTRUCTION` decisions.
8. [`h200_training_benchmark.md`](h200_training_benchmark.md): measured performance,
   memory and effective-batch analysis.
9. [`dancing_augmentation.md`](dancing_augmentation.md): zero-copy small dancing
   oversampling without adding another trainer data path.

## Local build

Install doc dependencies:

```bash
pip install -r docs/requirements.txt
```

Build HTML:

```bash
cd docs
make html
```

Open the output at `docs/build/html/index.html`.

## API reference generation

Generate API stubs from the Python packages:

```bash
cd docs
make apidoc
make html
```

Note: generated stubs are written to `docs/source/api_reference/_generated` and are not
included in the default navigation. Add them to a toctree if you want to expose them.
