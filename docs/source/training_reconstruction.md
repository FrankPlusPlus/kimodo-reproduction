# Training reconstruction

This repository includes a clean-room Kimodo training reconstruction in addition to the
released inference code. It is not NVIDIA's unpublished trainer, and the public
BONES-SEED/SOMA30 profile must not be reported as an exact reproduction of the private
Rigplay experiments.

Read the full Chinese guide on GitHub:

- [Paper alignment, engineering implementation, data flow, operations, and audit findings](https://github.com/FrankPlusPlus/kimodo-reproduction/blob/main/docs/reproduction_end_to_end_guide.md)
- [Portable training runbook](https://github.com/FrankPlusPlus/kimodo-reproduction/blob/main/docs/training_reproduction_runbook.md)
- [Clause-by-clause paper parity audit](https://github.com/FrankPlusPlus/kimodo-reproduction/blob/main/docs/paper_training_parity_audit.md)
- [Small BONES-SEED dancing augmentation](https://github.com/FrankPlusPlus/kimodo-reproduction/blob/main/docs/dancing_augmentation.md)

The strict profile intentionally remains blocked when the unpublished Qwen paraphrases,
cross-motion stitching, and diffusion-transition assets are unavailable. Use the public
profile for the auditable BONES-SEED engineering baseline.
