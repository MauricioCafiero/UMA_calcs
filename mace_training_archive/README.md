# MACE model fine-tuning

This folder is the fine-tuning project (the older training/active-learning/committee
notebook remains in git history).

`TrainMace_CafChem.ipynb`:
- Accepts the student's own reference data (extended XYZ with energies and forces)
  plus an optional starting structure
- Fine-tunes the MACE-MP-0 foundation model (`"small"`) on it, with a seeded
  train/valid/test split
- Runs a short MD simulation before and after fine-tuning with the same starting
  structure and conditions, and plots energy per atom and temperature for both models
- Shows both trajectories in py3Dmol

Helper code is in `code/mace_train.py` (`train_mace`, `make_train_file`, `simpleMD`,
`view_traj`); training-config templates are in `data/`. Example data lives in `data/`
and is used automatically if the student uploads nothing.

## Requirements
- Runs on Colab with a T4 GPU; Python 3.12 is fine (no XTB needed)
- Dependencies are installed by the notebook (mace-torch, aseMolec, py3Dmol,
  cuequivariance)

### Adapted from [here](https://colab.research.google.com/drive/1oCSVfMhWrqHTeHbKgUSQN9hTKxLzoNyb#scrollTo=XYamp68VxNVI)