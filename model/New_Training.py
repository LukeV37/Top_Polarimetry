import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from new_model import Model, VALID_ANALYSIS_TYPES, validate_analysis_type
from DataLoader_Parallel import CustomDataset  # noqa: F401 - needed to load serialized datasets


if len(sys.argv) != 7:
    raise SystemExit(
        "Usage: python New_Training.py <tag> <epochs> <embed_dim> "
        "<dir_dataset> <dir_training> <analysis_type>"
    )

tag = str(sys.argv[1])
epochs = int(sys.argv[2])
embed_dim = int(sys.argv[3])
dir_dataset = Path(sys.argv[4])
dir_training = Path(sys.argv[5])
train_type = validate_analysis_type(sys.argv[6])

if epochs <= 0:
    raise ValueError("epochs must be positive")

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


def select_target(top_labels, down_labels, bottom_labels, analysis_type):
    if analysis_type == "top":
        return top_labels
    if analysis_type == "down":
        return down_labels
    if analysis_type == "bottom":
        return bottom_labels
    raise ValueError(f"analysis_type must be one of {VALID_ANALYSIS_TYPES}, got {analysis_type!r}")


def task_loss(pred, target, analysis_type, mse_loss_fn, cos_sim_loss_fn):
    if analysis_type == "top":
        return mse_loss_fn(pred, target)

    cos_target = torch.ones(pred.shape[0], dtype=pred.dtype, device=pred.device)
    return cos_sim_loss_fn(pred, target, cos_target)


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

    for e in range(epochs):
        model.train()
        cumulative_loss_train = 0
        num_train = len(train_loader)

        for probe_jet, constituents, event, top_labels, down_labels, bottom_labels, _direct_labels, _track_labels in train_loader:
            optimizer.zero_grad()

            pred = model(probe_jet.to(device), constituents.to(device), event.to(device))
            target = select_target(top_labels, down_labels, bottom_labels, train_type).to(device)
            loss = task_loss(pred, target, train_type, mse_loss_fn, cos_sim_loss_fn)

            loss.backward()
            optimizer.step()

            cumulative_loss_train += loss.detach().cpu().numpy().mean()

        cumulative_loss_train = cumulative_loss_train / num_train

        model.eval()
        cumulative_loss_val = 0
        num_val = len(val_loader)
        with torch.inference_mode():
            for probe_jet, constituents, event, top_labels, down_labels, bottom_labels, _direct_labels, _track_labels in val_loader:
                pred = model(probe_jet.to(device), constituents.to(device), event.to(device))
                target = select_target(top_labels, down_labels, bottom_labels, train_type).to(device)
                loss = task_loss(pred, target, train_type, mse_loss_fn, cos_sim_loss_fn)

                cumulative_loss_val += loss.detach().cpu().numpy().mean()

        cumulative_loss_val = cumulative_loss_val / num_val
        combined_history.append([cumulative_loss_train, cumulative_loss_val])

        scheduler.step()

        print(
            "Epoch:",
            e + 1,
            "\tTrain Loss:",
            round(cumulative_loss_train, 6),
            "\tVal Loss:",
            round(cumulative_loss_val, 6),
        )
        print()

        torch.save(model, dir_training / "models" / f"model_Epoch_{e + 1}.torch")

    return np.array(combined_history)


dataset_path = dir_dataset / "dataset_combined.pt"
if not dataset_path.is_file():
    raise FileNotFoundError(f"Dataset not found: {dataset_path}")

dset = torch.load(dataset_path, weights_only=False)

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

print("Training tag:", tag)
print("Training task:", train_type)
print("GPU Available: ", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = Model(embed_dim, num_heads, train_type).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=Gamma)

print("Trainable Parameters :", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("Number of Training Events: ", len(train_dataset))
print("Number of Validation Events: ", len(val_dataset))
print("Number of Test Events: ", len(test_dataset))
print()

history = train(model, optimizer, scheduler, train_loader, val_loader, device, epochs=epochs)

torch.save(model, dir_training / "model_final.torch")

plt.figure()
plt.plot(history[:, 0], label="Train")
plt.plot(history[:, 1], label="Val")
plt.title(f"Loss: {train_type}")
plt.legend()
plt.yscale("log")
plt.savefig(dir_training / "loss_curve_total.png")
plt.close()

plt.figure()
plt.plot(history[int(epochs / 2) :, 0], label="Train")
plt.plot(history[int(epochs / 2) :, 1], label="Val")
plt.title(f"Loss: {train_type}")
plt.legend()
plt.yscale("log")
plt.savefig(dir_training / "loss_curve_second_half.png")
plt.close()

num_feats = len(TASK_FEATURES[train_type])
pred_batches = []
true_batches = []

model.eval()
with torch.inference_mode():
    for probe_jet, constituents, event, top_labels, down_labels, bottom_labels, _direct_labels, _track_labels in test_loader:
        pred = model(probe_jet.to(device), constituents.to(device), event.to(device))
        target = select_target(top_labels, down_labels, bottom_labels, train_type)

        pred_batches.append(pred.detach().cpu().numpy())
        true_batches.append(target.detach().cpu().numpy())

if not pred_batches:
    raise ValueError("Test dataset is empty")

pred_labels = np.concatenate(pred_batches, axis=0).reshape(-1, num_feats)
true_labels = np.concatenate(true_batches, axis=0).reshape(-1, num_feats)

validate_predictions(true_labels, pred_labels, TASK_FEATURES[train_type])
