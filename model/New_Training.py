import math
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from new_model import (
    Model,
    VALID_ANALYSIS_TYPES,
    VALID_LOSS_TYPES,
    validate_analysis_type,
    validate_loss_type,
)
from DataLoader_Parallel import CustomDataset  # noqa: F401 - needed to load serialized datasets


if len(sys.argv) not in (7, 8, 9):
    raise SystemExit(
        "Usage: python New_Training.py <tag> <epochs> <embed_dim> "
        "<dir_dataset> <dir_training> <analysis_type> [loss_type] [log_every_batches]"
    )

tag = str(sys.argv[1])
epochs = int(sys.argv[2])
embed_dim = int(sys.argv[3])
dir_dataset = Path(sys.argv[4])
dir_training = Path(sys.argv[5])
train_type = validate_analysis_type(sys.argv[6])
loss_type = validate_loss_type(sys.argv[7] if len(sys.argv) >= 8 else None, train_type)
log_every_batches = int(sys.argv[8]) if len(sys.argv) == 9 else 500

if epochs <= 0:
    raise ValueError("epochs must be positive")
if log_every_batches <= 0:
    raise ValueError("log_every_batches must be positive")

dir_training.mkdir(parents=True, exist_ok=True)
(dir_training / "models").mkdir(parents=True, exist_ok=True)

batch_size = 256
learning_rate = 0.0001
num_heads = 4
step_size = 160
Gamma = 0.1

TASK_FEATURES = {
    "top": ["top_px", "top_py", "top_pz", "top_e"],
    "down": ["down_px", "down_py", "down_pz"],
    "bottom": ["bottom_px", "bottom_py", "bottom_pz"],
}

TASK_RANGES = {
    "top_px": (-1000, 1000),
    "top_py": (-1000, 1000),
    "top_pz": (-1000, 1000),
    "top_e": (0, 1500),
    "down_px": (-1.1, 1.1),
    "down_py": (-1.1, 1.1),
    "down_pz": (-1.1, 1.1),
    "bottom_px": (-1.1, 1.1),
    "bottom_py": (-1.1, 1.1),
    "bottom_pz": (-1.1, 1.1),
}

LOSS_LABELS = {
    "mse": "MSE",
    "cossim": "CosSim",
    "vmf": "vMF",
}


def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def model_file_suffix(analysis_type, loss_type):
    return f"{analysis_type}_{LOSS_LABELS[loss_type]}"


def select_target(top_labels, down_labels, bottom_labels, analysis_type):
    if analysis_type == "top":
        return top_labels
    if analysis_type == "down":
        return down_labels
    if analysis_type == "bottom":
        return bottom_labels
    raise ValueError(f"analysis_type must be one of {VALID_ANALYSIS_TYPES}, got {analysis_type!r}")


def prediction_direction(model_output):
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


def vmf_nll(mu, kappa, target):
    """Negative log-likelihood for S^2 von Mises-Fisher predictions.

    mu:     (B, 3) unit vectors predicted by the model
    kappa:  (B,) positive concentration predicted by the model
    target: (B, 3) target unit vectors
    """
    target = F.normalize(target, dim=-1)
    dot = (mu * target).sum(-1)

    # log C_3(kappa) = log(kappa) - log(4*pi*sinh(kappa))
    #                  = log(kappa) - log(2*pi) - kappa - log(1-exp(-2*kappa))
    # The expm1 form is stable near kappa=0.
    log_norm = (
        torch.log(kappa)
        - math.log(2 * math.pi)
        - kappa
        - torch.log(-torch.expm1(-2 * kappa))
    )
    return -(log_norm + kappa * dot).mean()


def task_loss(pred, target, analysis_type, loss_type, mse_loss_fn, cos_sim_loss_fn):
    if analysis_type == "top":
        return mse_loss_fn(pred, target)

    pred_direction = prediction_direction(pred)

    if loss_type == "mse":
        return mse_loss_fn(pred_direction, target)
    if loss_type == "cossim":
        cos_target = torch.ones(
            pred_direction.shape[0], dtype=pred_direction.dtype, device=pred_direction.device
        )
        return cos_sim_loss_fn(pred_direction, target, cos_target)
    if loss_type == "vmf":
        mu, kappa = pred
        return vmf_nll(mu, kappa, target)

    raise ValueError(f"loss_type must be one of {VALID_LOSS_TYPES}, got {loss_type!r}")


def validate_predictions(true, pred, var_names):
    for i, var in enumerate(var_names):
        var_range = TASK_RANGES[var]

        plt.figure()
        plt.hist(
            np.ravel(true[:, i]),
            histtype="step",
            color="r",
            label="True Distribution",
            bins=50,
            range=var_range,
        )
        plt.hist(
            np.ravel(pred[:, i]),
            histtype="step",
            color="b",
            label="Predicted Distribution",
            bins=50,
            range=var_range,
        )
        plt.title(f"Predicted Output Distribution: {var}")
        plt.legend()
        plt.yscale("log")
        plt.xlabel(var, loc="right")
        plt.savefig(dir_training / f"pred_1d_{var}.png")
        plt.close()

        fig, ax = plt.subplots()
        plt.title(f"Output Distribution: {var}")
        ax.hist2d(
            np.ravel(pred[:, i]),
            np.ravel(true[:, i]),
            bins=100,
            norm=mcolors.LogNorm(),
            range=(var_range, var_range),
        )
        plt.xlabel(f"Predicted {var}", loc="right")
        plt.ylabel(f"True {var}", loc="top")
        diff = var_range[1] - var_range[0]
        plt.text(
            var_range[1] - 0.3 * diff,
            var_range[0] + 0.2 * diff,
            "$R^2$ value: " + str(round(r2_score(np.ravel(true[:, i]), np.ravel(pred[:, i])), 3)),
            backgroundcolor="r",
            color="k",
        )
        plt.savefig(dir_training / f"pred_2d_{var}.png")
        plt.close()


def train(model, optimizer, scheduler, train_loader, val_loader, device, epochs=40):
    combined_history = []
    mse_loss_fn = nn.MSELoss()
    cos_sim_loss_fn = nn.CosineEmbeddingLoss()
    suffix = model_file_suffix(train_type, loss_type)

    for e in range(epochs):
        epoch_start = time.time()
        model.train()
        cumulative_loss_train = 0
        num_train = len(train_loader)
        log(f"Epoch {e + 1}/{epochs}: starting train phase ({num_train} batches)")

        phase_start = time.time()
        for batch_idx, (probe_jet, constituents, event, top_labels, down_labels, bottom_labels, _direct_labels, _track_labels) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            pred = model(probe_jet.to(device), constituents.to(device), event.to(device))
            target = select_target(top_labels, down_labels, bottom_labels, train_type).to(device)
            loss = task_loss(pred, target, train_type, loss_type, mse_loss_fn, cos_sim_loss_fn)

            loss.backward()
            optimizer.step()

            cumulative_loss_train += loss.detach().cpu().numpy().mean()

            if batch_idx == 1 or batch_idx % log_every_batches == 0 or batch_idx == num_train:
                elapsed = time.time() - phase_start
                batches_per_sec = batch_idx / elapsed if elapsed > 0 else 0
                remaining = (num_train - batch_idx) / batches_per_sec if batches_per_sec > 0 else 0
                running_loss = cumulative_loss_train / batch_idx
                log(
                    f"Epoch {e + 1}/{epochs} train batch {batch_idx}/{num_train} "
                    f"loss={loss.detach().item():.6g} running_loss={running_loss:.6g} "
                    f"elapsed={format_duration(elapsed)} eta={format_duration(remaining)}"
                )

        cumulative_loss_train = cumulative_loss_train / num_train

        model.eval()
        cumulative_loss_val = 0
        num_val = len(val_loader)
        log(f"Epoch {e + 1}/{epochs}: starting validation phase ({num_val} batches)")
        phase_start = time.time()
        with torch.inference_mode():
            for batch_idx, (probe_jet, constituents, event, top_labels, down_labels, bottom_labels, _direct_labels, _track_labels) in enumerate(val_loader, start=1):
                pred = model(probe_jet.to(device), constituents.to(device), event.to(device))
                target = select_target(top_labels, down_labels, bottom_labels, train_type).to(device)
                loss = task_loss(pred, target, train_type, loss_type, mse_loss_fn, cos_sim_loss_fn)

                cumulative_loss_val += loss.detach().cpu().numpy().mean()

                if batch_idx == 1 or batch_idx % log_every_batches == 0 or batch_idx == num_val:
                    elapsed = time.time() - phase_start
                    batches_per_sec = batch_idx / elapsed if elapsed > 0 else 0
                    remaining = (num_val - batch_idx) / batches_per_sec if batches_per_sec > 0 else 0
                    running_loss = cumulative_loss_val / batch_idx
                    log(
                        f"Epoch {e + 1}/{epochs} val batch {batch_idx}/{num_val} "
                        f"loss={loss.detach().item():.6g} running_loss={running_loss:.6g} "
                        f"elapsed={format_duration(elapsed)} eta={format_duration(remaining)}"
                    )

        cumulative_loss_val = cumulative_loss_val / num_val
        combined_history.append([cumulative_loss_train, cumulative_loss_val])

        scheduler.step()

        log(
            f"Epoch {e + 1}/{epochs} complete: "
            f"train_loss={cumulative_loss_train:.6g} val_loss={cumulative_loss_val:.6g} "
            f"epoch_time={format_duration(time.time() - epoch_start)}"
        )

        epoch_model_path = dir_training / "models" / f"model_Epoch_{e + 1}_{suffix}.torch"
        torch.save(model, epoch_model_path)
        log(f"Saved epoch checkpoint: {epoch_model_path}")

    return np.array(combined_history)


dataset_path = dir_dataset / "dataset_combined.pt"
if not dataset_path.is_file():
    raise FileNotFoundError(f"Dataset not found: {dataset_path}")

log(f"Loading dataset: {dataset_path}")
load_start = time.time()
dset = torch.load(dataset_path, weights_only=False)
log(f"Loaded dataset with {len(dset)} events in {format_duration(time.time() - load_start)}")

generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset = torch.utils.data.random_split(
    dset, [0.75, 0.25], generator=generator
)
val_dataset, test_dataset = torch.utils.data.random_split(
    test_dataset, [0.2, 0.8], generator=generator
)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

log(f"Training tag: {tag}")
log(f"Training task: {train_type}")
log(f"Loss type: {loss_type} ({LOSS_LABELS[loss_type]})")
log(f"Training directory: {dir_training}")
log(f"Batch size: {batch_size}")
log(f"Log every batches: {log_every_batches}")
log(f"GPU Available: {torch.cuda.is_available()}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log(f"Device: {device}")

model = Model(embed_dim, num_heads, train_type, loss_type).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=Gamma)

log(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
log(f"Number of Training Events: {len(train_dataset)}")
log(f"Number of Validation Events: {len(val_dataset)}")
log(f"Number of Test Events: {len(test_dataset)}")

training_start = time.time()
history = train(model, optimizer, scheduler, train_loader, val_loader, device, epochs=epochs)
log(f"Training loop finished in {format_duration(time.time() - training_start)}")

final_model_path = dir_training / f"model_final_{model_file_suffix(train_type, loss_type)}.torch"
torch.save(model, final_model_path)
torch.save(model, dir_training / "model_final.torch")
log(f"Saved final named checkpoint: {final_model_path}")
log(f"Saved final compatibility checkpoint: {dir_training / 'model_final.torch'}")

plt.figure()
plt.plot(history[:, 0], label="Train")
plt.plot(history[:, 1], label="Val")
plt.title(f"Loss: {train_type} ({loss_type})")
plt.legend()
plt.yscale("log")
plt.savefig(dir_training / "loss_curve_total.png")
plt.close()

plt.figure()
plt.plot(history[int(epochs / 2) :, 0], label="Train")
plt.plot(history[int(epochs / 2) :, 1], label="Val")
plt.title(f"Loss: {train_type} ({loss_type})")
plt.legend()
plt.yscale("log")
plt.savefig(dir_training / "loss_curve_second_half.png")
plt.close()

num_feats = len(TASK_FEATURES[train_type])
pred_batches = []
true_batches = []

model.eval()
log(f"Starting test prediction pass ({len(test_loader)} batches)")
test_start = time.time()
with torch.inference_mode():
    for batch_idx, (probe_jet, constituents, event, top_labels, down_labels, bottom_labels, _direct_labels, _track_labels) in enumerate(test_loader, start=1):
        pred = model(probe_jet.to(device), constituents.to(device), event.to(device))
        target = select_target(top_labels, down_labels, bottom_labels, train_type)

        pred_batches.append(prediction_direction(pred).detach().cpu().numpy())
        true_batches.append(target.detach().cpu().numpy())

        if batch_idx == 1 or batch_idx % log_every_batches == 0 or batch_idx == len(test_loader):
            elapsed = time.time() - test_start
            batches_per_sec = batch_idx / elapsed if elapsed > 0 else 0
            remaining = (len(test_loader) - batch_idx) / batches_per_sec if batches_per_sec > 0 else 0
            log(
                f"Test batch {batch_idx}/{len(test_loader)} "
                f"elapsed={format_duration(elapsed)} eta={format_duration(remaining)}"
            )

if not pred_batches:
    raise ValueError("Test dataset is empty")

pred_labels = np.concatenate(pred_batches, axis=0).reshape(-1, num_feats)
true_labels = np.concatenate(true_batches, axis=0).reshape(-1, num_feats)

log("Creating prediction plots")
validate_predictions(true_labels, pred_labels, TASK_FEATURES[train_type])
log(f"Finished training job for {train_type}/{loss_type}")
