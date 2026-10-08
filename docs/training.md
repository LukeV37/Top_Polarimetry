# Training Guide

This repository now supports **three independent training tasks**:

- `top` → predict the top-quark 4-vector: `(px, py, pz, E)`
- `down` → predict the down-quark 3-vector direction: `(px, py, pz)`
- `bottom` → predict the bottom-quark 3-vector direction: `(px, py, pz)`

These are **not** trained as a multi-task model.

Instead, the intended design is:

- run **one training job per task**
- produce **one checkpoint per task**
- store each task in its own training directory

## Model design

The training code uses `analysis_type` to choose a **single-task model**.

### Supported tasks

```bash
analysis_type="top"
analysis_type="down"
analysis_type="bottom"
```

### Task behavior

- `top`
  - output dimension: `4`
  - target: `top_labels`
  - loss: `MSELoss`

- `down`
  - output dimension: `3`
  - target: `down_labels`
  - configurable loss: `mse`, `cossim`, or `vmf`
  - output is normalized to a unit vector

- `bottom`
  - output dimension: `3`
  - target: `bottom_labels`
  - configurable loss: `mse`, `cossim`, or `vmf`
  - output is normalized to a unit vector

## Loss configuration

The loss is configured with:

```bash
loss_type="cossim"
```

Supported values are:

```bash
loss_type="mse"
loss_type="cossim"
loss_type="vmf"
```

Rules:

- `analysis_type="top"` always uses `loss_type="mse"`.
- `analysis_type="down"` supports `mse`, `cossim`, and `vmf`.
- `analysis_type="bottom"` supports `mse`, `cossim`, and `vmf`.

For `vmf`, the model predicts:

- `mu`: the normalized 3-vector direction
- `kappa`: the positive vMF concentration parameter

The training loss is the vMF negative log-likelihood. Evaluation uses `mu` as the predicted direction and also saves the predicted `kappa` values.

## Training outputs

The training directory is built from the task name:

```bash
training_tag="_${analysis_type}_${loss_type}_${epochs}epoch_${embed_dim}embed"
dir_training="WS_${PY_tag}/training${training_tag}"
```

So the three trainings naturally write to different directories, for example:

```text
WS_U_10M_R1_5_pT250/training_top_mse_60epoch_64embed
WS_U_10M_R1_5_pT250/training_down_cossim_60epoch_64embed
WS_U_10M_R1_5_pT250/training_bottom_vmf_60epoch_64embed
```

Each directory contains its own:

- `model_final.torch`
- epoch checkpoints in `models/`
- loss plots
- task-specific prediction plots

## How training is launched

Training is launched through:

```bash
./run_job.sh <config-file>
```

The script reads the config file, checks `analysis_type`, and runs:

```bash
python -u New_Training.py "$PY_tag" "$epochs" "$embed_dim" "$dir_datasets" "$dir_training" "$analysis_type" "$loss_type" "$log_every_batches"
```

## Important config settings

In the config file, these fields control the training task:

```bash
analysis_type="down"
loss_type="cossim"
epochs=60
embed_dim=64
log_every_batches=500
training_tag="_${analysis_type}_${loss_type}_${epochs}epoch_${embed_dim}embed"
dir_training="WS_${PY_tag}/training${training_tag}"
```

To actually run training, make sure:

```bash
bypass_train=false
```

If your dataset is already prepared and you only want to train, you will usually also want:

```bash
bypass_madgraph=true
bypass_pythia=true
bypass_preprocessing=true
```

That avoids regenerating data when you only want a new model fit.

## Logging

Training logs now include timestamped progress messages for long runs:

- dataset path and loaded event count
- task, loss, output directory, batch size, and device
- train/validation/test batch progress
- elapsed time and ETA
- epoch checkpoint path
- final named checkpoint path

Control the logging cadence with:

```bash
log_every_batches=500
```

For each run, `run_job.sh` writes a task/loss-specific log file:

```text
training_<analysis_type>_<loss_label>.log
```

For example:

```text
training_down_CosSim.log
training_bottom_vMF.log
```

Final checkpoints are saved both as the compatibility name:

```text
model_final.torch
```

and as a unique named checkpoint:

```text
model_final_<analysis_type>_<loss_label>.torch
```

For example:

```text
model_final_top_MSE.torch
model_final_down_CosSim.torch
model_final_bottom_vMF.torch
```

## Recommended way to run the three trainings

The simplest workflow is to make three config files, one for each task.

For example:

- `job_top.config`
- `job_down.config`
- `job_bottom.config`

Each file should set a different `analysis_type`.

### Example: top training config

```bash
analysis_type="top"
loss_type="mse"
bypass_train=false
```

### Example: down training config

```bash
analysis_type="down"
loss_type="cossim"
bypass_train=false
```

### Example: bottom training config

```bash
analysis_type="bottom"
loss_type="cossim"
bypass_train=false
```

## Commands to run the jobs

### Option 1: edit one config file and run it three times

Set `analysis_type` in the config, then run:

```bash
./run_job.sh job.config
```

Repeat for:

```bash
analysis_type="top"
analysis_type="down"
analysis_type="bottom"
```

### Option 2: use three dedicated config files

Run the jobs directly:

```bash
./run_job.sh job_top.config
./run_job.sh job_down.config
./run_job.sh job_bottom.config
```

This is the safest approach because it avoids accidentally overwriting or reusing the wrong task setting.

## Example training-only config snippet

If datasets already exist and you only want to train models, a config can look like:

```bash
#!/bin/bash

bypass_madgraph=true
bypass_pythia=true
bypass_preprocessing=true
bypass_train=false

process="U"
MG_tag="${process}_10M_gen"
num_runs=40
num_events_per_run="250k"
max_cpu_cores=40
seed=12

R="1.5"
minJetpT="250"
PY_tag="${process}_10M_R${R/./_}_pT${minJetpT}"
dir_datasets="WS_${PY_tag}/datasets_AllFrame"

analysis_type="top"
loss_type="mse"
epochs=60
embed_dim=64
log_every_batches=500
training_tag="_${analysis_type}_${loss_type}_${epochs}epoch_${embed_dim}embed"
dir_training="WS_${PY_tag}/training${training_tag}"
```

For `down` and `bottom`, change both the task and, if desired, the loss:

```bash
analysis_type="down"
loss_type="cossim"
```

or

```bash
analysis_type="bottom"
loss_type="vmf"
```

## Running a down/bottom loss study

For a loss study, run separate jobs for each task/loss combination.

The top model only has one case. In `job_top_mse.config`, set:

```bash
analysis_type="top"
loss_type="mse"
```

Then run:

```bash
./run_job.sh job_top_mse.config
```

For down, create one config per loss:

```bash
# job_down_mse.config
analysis_type="down"
loss_type="mse"

# job_down_cossim.config
analysis_type="down"
loss_type="cossim"

# job_down_vmf.config
analysis_type="down"
loss_type="vmf"
```

Then run:

```bash
./run_job.sh job_down_mse.config
./run_job.sh job_down_cossim.config
./run_job.sh job_down_vmf.config
```

For bottom, create one config per loss:

```bash
# job_bottom_mse.config
analysis_type="bottom"
loss_type="mse"

# job_bottom_cossim.config
analysis_type="bottom"
loss_type="cossim"

# job_bottom_vmf.config
analysis_type="bottom"
loss_type="vmf"
```

Then run:

```bash
./run_job.sh job_bottom_mse.config
./run_job.sh job_bottom_cossim.config
./run_job.sh job_bottom_vmf.config
```

Each config should keep:

```bash
training_tag="_${analysis_type}_${loss_type}_${epochs}epoch_${embed_dim}embed"
```

so that every task/loss case writes to a different output directory.

## Full U_10M 100-epoch loss study

For the full seven-model U_10M study, use the root runbook:

```text
AGENTS.md
```

That runbook is intentionally written as a guided submission procedure for future
agents/operators. It does **not** imply the jobs have already been submitted.

The full-study configs are:

```text
configs/u10m_100epoch/job_top_MSE.config
configs/u10m_100epoch/job_down_MSE.config
configs/u10m_100epoch/job_down_CosSim.config
configs/u10m_100epoch/job_down_vMF.config
configs/u10m_100epoch/job_bottom_MSE.config
configs/u10m_100epoch/job_bottom_CosSim.config
configs/u10m_100epoch/job_bottom_vMF.config
```

They assume the local preprocessed samples already exist at:

```text
model/WS_U_10M_R1_5_pT250/datasets_AllFrame/dataset_combined.pt
```

and use:

```bash
epochs=100
embed_dim=64
dir_datasets="WS_${PY_tag}/datasets_AllFrame"
bypass_madgraph=true
bypass_pythia=true
bypass_preprocessing=true
bypass_train=false
```

Run the seven jobs sequentially, not in parallel:

```bash
./run_job.sh configs/u10m_100epoch/job_top_MSE.config

./run_job.sh configs/u10m_100epoch/job_down_MSE.config
./run_job.sh configs/u10m_100epoch/job_down_CosSim.config
./run_job.sh configs/u10m_100epoch/job_down_vMF.config

./run_job.sh configs/u10m_100epoch/job_bottom_MSE.config
./run_job.sh configs/u10m_100epoch/job_bottom_CosSim.config
./run_job.sh configs/u10m_100epoch/job_bottom_vMF.config
```

Expected output directories include:

```text
model/WS_U_10M_R1_5_pT250/training_top_MSE_100epoch_64embed
model/WS_U_10M_R1_5_pT250/training_down_CosSim_100epoch_64embed
model/WS_U_10M_R1_5_pT250/training_bottom_vMF_100epoch_64embed
```

## One-epoch U_10M debug loss study

The checked-in debug configs under `configs/debug_u10m/` run one epoch against the existing U_10M workspace and skip generation/preprocessing.

Run them sequentially, not in parallel:

```bash
./run_job.sh configs/debug_u10m/job_top_MSE.config

./run_job.sh configs/debug_u10m/job_down_MSE.config
./run_job.sh configs/debug_u10m/job_down_CosSim.config
./run_job.sh configs/debug_u10m/job_down_vMF.config

./run_job.sh configs/debug_u10m/job_bottom_MSE.config
./run_job.sh configs/debug_u10m/job_bottom_CosSim.config
./run_job.sh configs/debug_u10m/job_bottom_vMF.config
```

Progress and results from the debug run are tracked in:

```text
docs/debug_u10m_loss_study.md
```

## Evaluating a trained model

Use `model/evaluate.py` with the matching task:

```bash
python model/evaluate.py \
  --left-dataset model/WS_L_1M_R1_5_pT250/datasets_AllFrame/dataset_combined.pt \
  --right-dataset model/WS_R_1M_R1_5_pT250/datasets_AllFrame/dataset_combined.pt \
  --model model/WS_U_10M_R1_5_pT250/training_top_mse_60epoch_64embed/model_final.torch \
  --task top \
  --output-dir workspaces/eval_top
```

For the other tasks:

```bash
python model/evaluate.py --left-dataset <L_dataset> --right-dataset <R_dataset> --model <down_model> --task down --output-dir <eval_dir>
python model/evaluate.py --left-dataset <L_dataset> --right-dataset <R_dataset> --model <bottom_model> --task bottom --output-dir <eval_dir>
```

## Notes

- `run_job.sh` now validates that `analysis_type` is one of `top`, `down`, or `bottom`.
- `run_job.sh` validates that `loss_type` is compatible with the selected task.
- The new model is single-task by construction, so each checkpoint is tied to one task.
- A `top` checkpoint should be evaluated with `--task top`, and similarly for `down` and `bottom`.
- vMF evaluation saves `pred_kappa_<task>_L.npy` and `pred_kappa_<task>_R.npy` in addition to direction predictions.
- The current `job.config` in your working tree may have `bypass_train=true`; if so, training will be skipped until you change it to `false`.
