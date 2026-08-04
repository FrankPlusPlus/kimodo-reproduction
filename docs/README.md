# Documentation

## Training reconstruction

The training implementation is a clean-room reconstruction layered on the released
Kimodo inference code. Read these documents in this order:

1. [`reproduction_end_to_end_guide.md`](reproduction_end_to_end_guide.md): complete
   resource, data, model, training, artifact and paper-alignment walkthrough.
2. [`training_reproduction_runbook.md`](training_reproduction_runbook.md): operational
   commands for a fresh server and two-H200 training.
3. [`paper_training_parity_audit.md`](paper_training_parity_audit.md): clause-by-clause
   paper gate and known unavailable inputs.
4. [`training_reproduction_spec.md`](training_reproduction_spec.md): detailed
   `PAPER`/`CODE`/`RECONSTRUCTION` decisions.
5. [`h200_training_benchmark.md`](h200_training_benchmark.md): measured performance,
   memory and effective-batch analysis.

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
