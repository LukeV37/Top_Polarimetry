# Agent Runbook: U_10M Seven-Model Training Study

This runbook explains how an agent should launch the full U_10M training study.

Do **not** submit these jobs unless the user explicitly asks you to run them. This
file is a guide for how to submit them correctly when requested.

## Goal

Train seven independent single-task models:

1. `top` with `MSE`
2. `down` with `MSE`
3. `down` with `CosSim`
4. `down` with `vMF`
5. `bottom` with `MSE`
6. `bottom` with `CosSim`
7. `bottom` with `vMF`

Each model should be trained as a separate run. Do not combine tasks or losses in
one model.

## Required dataset

Use the local preprocessed U_10M samples in this repository:

```text
model/WS_U_10M_R1_5_pT250/datasets_AllFrame/dataset_combined.pt
```

Do **not** use the sibling workspace:

```text
/mnt/md0/lvaughan/Top_Polarimetry/model/...
```

The training configs in `configs/u10m_100epoch/` intentionally set:

```bash
dir_datasets="WS_${PY_tag}/datasets_AllFrame"
```

This is relative to `model/`, because `run_job.sh` changes into `model/` before
calling `New_Training.py`.

Before launching anything, verify the local dataset exists:

```bash
test -f model/WS_U_10M_R1_5_pT250/datasets_AllFrame/dataset_combined.pt
```

## Training settings

All seven full-study configs use:

```bash
epochs=100
embed_dim=64
log_every_batches=500
bypass_madgraph=true
bypass_pythia=true
bypass_preprocessing=true
bypass_train=false
```

The bypass settings mean the jobs assume preprocessing is already complete and
only run training.

## Config files

Use these configs:

```text
configs/u10m_100epoch/job_top_MSE.config
configs/u10m_100epoch/job_down_MSE.config
configs/u10m_100epoch/job_down_CosSim.config
configs/u10m_100epoch/job_down_vMF.config
configs/u10m_100epoch/job_bottom_MSE.config
configs/u10m_100epoch/job_bottom_CosSim.config
configs/u10m_100epoch/job_bottom_vMF.config
```

Each config writes to a unique directory under:

```text
model/WS_U_10M_R1_5_pT250/
```

For example:

```text
model/WS_U_10M_R1_5_pT250/training_top_MSE_100epoch_64embed/
model/WS_U_10M_R1_5_pT250/training_down_CosSim_100epoch_64embed/
model/WS_U_10M_R1_5_pT250/training_bottom_vMF_100epoch_64embed/
```

Final named checkpoints should look like:

```text
model_final_top_MSE.torch
model_final_down_MSE.torch
model_final_down_CosSim.torch
model_final_down_vMF.torch
model_final_bottom_MSE.torch
model_final_bottom_CosSim.torch
model_final_bottom_vMF.torch
```

## Submission policy

Run only **one** GPU job at a time. Do not run these in parallel.

Before each job:

```bash
nvidia-smi
```

If the GPU is busy, wait and check again later. Do not start another training
process until the previous command has exited.

## Sequential command list

Run these commands from the repository root, one at a time:

```bash
./run_job.sh configs/u10m_100epoch/job_top_MSE.config

./run_job.sh configs/u10m_100epoch/job_down_MSE.config
./run_job.sh configs/u10m_100epoch/job_down_CosSim.config
./run_job.sh configs/u10m_100epoch/job_down_vMF.config

./run_job.sh configs/u10m_100epoch/job_bottom_MSE.config
./run_job.sh configs/u10m_100epoch/job_bottom_CosSim.config
./run_job.sh configs/u10m_100epoch/job_bottom_vMF.config
```

Recommended operator workflow:

```bash
nvidia-smi
./run_job.sh configs/u10m_100epoch/job_top_MSE.config

nvidia-smi
./run_job.sh configs/u10m_100epoch/job_down_MSE.config

nvidia-smi
./run_job.sh configs/u10m_100epoch/job_down_CosSim.config

nvidia-smi
./run_job.sh configs/u10m_100epoch/job_down_vMF.config

nvidia-smi
./run_job.sh configs/u10m_100epoch/job_bottom_MSE.config

nvidia-smi
./run_job.sh configs/u10m_100epoch/job_bottom_CosSim.config

nvidia-smi
./run_job.sh configs/u10m_100epoch/job_bottom_vMF.config
```

If any job fails, stop and inspect its log before launching the next one.

## Logs

Each training directory contains a task/loss-specific log:

```text
training_<analysis_type>_<loss_label>.log
```

Examples:

```text
training_top_MSE.log
training_down_CosSim.log
training_bottom_vMF.log
```

The training log includes timestamps, batch progress, elapsed time, ETA, and
checkpoint save paths.

## Verification after each job

After each run, verify the named final checkpoint exists. For example:

```bash
test -f model/WS_U_10M_R1_5_pT250/training_top_MSE_100epoch_64embed/model_final_top_MSE.torch
test -f model/WS_U_10M_R1_5_pT250/training_down_CosSim_100epoch_64embed/model_final_down_CosSim.torch
test -f model/WS_U_10M_R1_5_pT250/training_bottom_vMF_100epoch_64embed/model_final_bottom_vMF.torch
```

## Important notes for future agents

- Use only the local `Top_Polarimetry_vMF/model/WS_U_10M...` preprocessed samples.
- Do not use `/mnt/md0/lvaughan/Top_Polarimetry/model/...` for this study.
- Do not submit jobs in parallel.
- Do not start these long jobs unless the user explicitly asks you to run them.
- If you update configs, keep `epochs=100` and `embed_dim=64` for this full study.
