# Mol-TDL
Molecular Topological Deep Learning for Polymer Property Prediction

This repository provides a documented, end-to-end pretraining pipeline and task-specific fine-tuning for topological molecular modeling. We release:
* Exact configs under configs/
* Deterministic splits under splits/<dataset>/
* Pretraining scripts and final weights with logs
* One-command examples for pretraining and fine-tuning
* Environment/automation for full reproducibility

# 1) Environment
* conda env create -f environment.yml
* conda activate mol-tdl

# 2) Repository Layout (key folders)
configs/               # YAML or config files (hyperparameters & I/O paths)
data/                  # raw CSVs, generated SDFs, and 3D TXT files
data/processed/        # processed PyG shards: <dataset>_agu_{train,vali}{i}.pt (+ _mask)
pretrain/
  data_process.py      # CSV -> SDF + 3D TXT (train/vali)
  creat_data_DC.py     # 3D TXT -> simplicial complexes -> PyG .pt shards
  training_GCN.py      # contrastive pretraining (variant 1)
  training_GCN2.py     # contrastive pretraining (variant 2; use the one in your repo)
finetune/
  training_finetune_*.py  # task-specific fine-tuning scripts (e.g., E_ea)
outputs/               # saved checkpoints, npy, and curves

# 3) Pretraining — Step by Step
## 3.1 CSV → SDF & 3D coordinate TXT
python pretrain/data_process.py

# Contacts
If you have any questions or comments, please feel free to email Cong Shen (cshen[at]hnu[dot]edu[dot]cn).
