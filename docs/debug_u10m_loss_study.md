# U_10M One-Epoch Debug Loss Study

This file tracks the one-epoch debug runs over the existing U_10M workspace.

## Dataset and output locations

Dataset used by all runs:

```text
/mnt/md0/lvaughan/Top_Polarimetry_vMF/model/WS_U_10M_R1_5_pT250/datasets_AllFrame/dataset_combined.pt
```

Correction note: the first sequential debug pass was accidentally run against
the sibling workspace `/mnt/md0/lvaughan/Top_Polarimetry/model/...` after a broad
dataset search found that path. The configs have been corrected to use the
current `Top_Polarimetry_vMF` workspace path shown above. The completed statuses
below refer to the first pass and should be rerun against the corrected vMF
dataset if these debug results are used for comparison.

Debug outputs are written under:

```text
/mnt/md0/lvaughan/Top_Polarimetry_vMF/model/WS_U_10M_R1_5_pT250/
```

Each model is saved in a unique training directory containing the analysis type and loss label.
Corrected vMF-dataset reruns use directory names beginning with
`training_debug_current_...` to keep them separate from the first-pass outputs.

## Logging improvements used for this study

`New_Training.py` now logs:

- timestamped messages
- dataset path and dataset size
- task, loss, output directory, batch size, and device
- train/validation/test batch progress every `log_every_batches`
- elapsed time and ETA for each phase
- epoch checkpoint save path
- final named checkpoint save path

Final checkpoints are saved both as:

```text
model_final.torch
```

and as a unique named file:

```text
model_final_<analysis_type>_<loss_label>.torch
```

For example:

```text
model_final_down_CosSim.torch
model_final_bottom_vMF.torch
```

## Sequential run order

Only one GPU training job should be run at a time.

| Order | Config | Task | Loss | Status |
| --- | --- | --- | --- | --- |
| 1 | `configs/debug_u10m/job_top_MSE.config` | `top` | `MSE` | Completed first pass on sibling dataset; rerun needed on corrected vMF dataset |
| 2 | `configs/debug_u10m/job_down_MSE.config` | `down` | `MSE` | Completed first pass on sibling dataset; rerun needed on corrected vMF dataset |
| 3 | `configs/debug_u10m/job_down_CosSim.config` | `down` | `CosSim` | Completed first pass on sibling dataset; rerun needed on corrected vMF dataset |
| 4 | `configs/debug_u10m/job_down_vMF.config` | `down` | `vMF` | Completed first pass on sibling dataset; rerun needed on corrected vMF dataset |
| 5 | `configs/debug_u10m/job_bottom_MSE.config` | `bottom` | `MSE` | Completed first pass on sibling dataset; rerun needed on corrected vMF dataset |
| 6 | `configs/debug_u10m/job_bottom_CosSim.config` | `bottom` | `CosSim` | Completed first pass on sibling dataset; rerun needed on corrected vMF dataset |
| 7 | `configs/debug_u10m/job_bottom_vMF.config` | `bottom` | `vMF` | Completed first pass on sibling dataset; rerun needed on corrected vMF dataset |

## Commands

Run these sequentially, not in parallel:

```bash
./run_job.sh configs/debug_u10m/job_top_MSE.config
./run_job.sh configs/debug_u10m/job_down_MSE.config
./run_job.sh configs/debug_u10m/job_down_CosSim.config
./run_job.sh configs/debug_u10m/job_down_vMF.config
./run_job.sh configs/debug_u10m/job_bottom_MSE.config
./run_job.sh configs/debug_u10m/job_bottom_CosSim.config
./run_job.sh configs/debug_u10m/job_bottom_vMF.config
```

## Notes from this run

- Prepared one-epoch configs for all 7 task/loss cases.
- `debug.config` has also been pointed at the U_10M workspace as a one-epoch `top/MSE` template.
- All 7 one-epoch debug jobs completed successfully in the first pass, run sequentially.
- The first pass used the sibling dataset by mistake. That loaded dataset contained 198,386 events, split into 148,790 training, 9,920 validation, and 39,676 test events.
- The corrected vMF dataset path is now configured for the next pass.
