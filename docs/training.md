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
  - loss: `CosineEmbeddingLoss`
  - output is normalized to a unit vector

- `bottom`
  - output dimension: `3`
  - target: `bottom_labels`
  - loss: `CosineEmbeddingLoss`
  - output is normalized to a unit vector

## Training outputs

The training directory is built from the task name:

```bash
training_tag="_${analysis_type}_${epochs}epoch_${embed_dim}embed"
dir_training="WS_${PY_tag}/training${training_tag}"
```

So the three trainings naturally write to different directories, for example:

```text
WS_U_10M_R1_5_pT250/training_top_60epoch_64embed
WS_U_10M_R1_5_pT250/training_down_60epoch_64embed
WS_U_10M_R1_5_pT250/training_bottom_60epoch_64embed
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
python -u New_Training.py "$PY_tag" "$epochs" "$embed_dim" "$dir_datasets" "$dir_training" "$analysis_type"
```

## Important config settings

In the config file, these fields control the training task:

```bash
analysis_type="down"
epochs=60
embed_dim=64
training_tag="_${analysis_type}_${epochs}epoch_${embed_dim}embed"
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
bypass_train=false
```

### Example: down training config

```bash
analysis_type="down"
bypass_train=false
```

### Example: bottom training config

```bash
analysis_type="bottom"
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
epochs=60
embed_dim=64
training_tag="_${analysis_type}_${epochs}epoch_${embed_dim}embed"
dir_training="WS_${PY_tag}/training${training_tag}"
```

For `down` and `bottom`, only change:

```bash
analysis_type="down"
```

or

```bash
analysis_type="bottom"
```

## Evaluating a trained model

Use `model/evaluate.py` with the matching task:

```bash
python model/evaluate.py \
  --left-dataset model/WS_L_1M_R1_5_pT250/datasets_AllFrame/dataset_combined.pt \
  --right-dataset model/WS_R_1M_R1_5_pT250/datasets_AllFrame/dataset_combined.pt \
  --model model/WS_U_10M_R1_5_pT250/training_top_60epoch_64embed/model_final.torch \
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
- The new model is single-task by construction, so each checkpoint is tied to one task.
- A `top` checkpoint should be evaluated with `--task top`, and similarly for `down` and `bottom`.
- The current `job.config` in your working tree may have `bypass_train=true`; if so, training will be skipped until you change it to `false`.
