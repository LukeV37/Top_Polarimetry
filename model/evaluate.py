"""Evaluate one independently trained task model on L and R datasets.

Example:
    python model/evaluate.py \
        --left-dataset model/WS_L_1M_R1_5_pT250/datasets_AllFrame/dataset_combined.pt \
        --right-dataset model/WS_R_1M_R1_5_pT250/datasets_AllFrame/dataset_combined.pt \
        --model model/WS_U_10M_R1_5_pT250/training_down_60epoch_64embed/model_final.torch \
        --task down \
        --output-dir workspaces/eval_down

The output directory contains NumPy arrays and comparison plots for the selected
task only. Supported tasks are:
    top    -> 4-vector regression
    down   -> normalized 3-vector direction regression
    bottom -> normalized 3-vector direction regression

For vMF down/bottom checkpoints, predicted kappa arrays are also saved.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from DataLoader_Parallel import CustomDataset  # noqa: F401 - needed to load serialized datasets
from new_model import Model, VALID_ANALYSIS_TYPES, validate_analysis_type  # noqa: F401


TASK_FEATURES = {
    "top": ("top_px", "top_py", "top_pz", "top_e"),
    "down": ("down_px", "down_py", "down_pz"),
    "bottom": ("bottom_px", "bottom_py", "bottom_pz"),
}

TASK_RANGES = {
    "top": ((-1000, 1000), (-1000, 1000), (-1000, 1000), (0, 1500)),
    "down": ((-1.1, 1.1),) * 3,
    "bottom": ((-1.1, 1.1),) * 3,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one independently trained task model on L and R datasets."
    )
    parser.add_argument("--left-dataset", required=True, help="Path to the L dataset_combined.pt")
    parser.add_argument("--right-dataset", required=True, help="Path to the R dataset_combined.pt")
    parser.add_argument(
        "--model",
        "--checkpoint",
        dest="model_path",
        required=True,
        help="Path to the trained model_final.torch checkpoint",
    )
    parser.add_argument(
        "--task",
        choices=VALID_ANALYSIS_TYPES,
        default="down",
        help="Which task to evaluate: top, down, or bottom (default: down)",
    )
    parser.add_argument("--output-dir", required=True, help="New workspace directory for output arrays")
    parser.add_argument("--batch-size", type=int, default=256, help="Evaluation batch size (default: 256)")
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, or a torch device string (default: auto)",
    )
    return parser.parse_args()


def get_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({device}) but CUDA is not available")
    return device


def select_target(batch, task):
    if task == "top":
        return batch[3]
    if task == "down":
        return batch[4]
    if task == "bottom":
        return batch[5]
    raise ValueError(f"task must be one of {VALID_ANALYSIS_TYPES}, got {task!r}")


def select_prediction(model_output, task):
    """Return selected prediction and optional vMF kappa.

    Handles new single-output models, new vMF `(mu, kappa)` outputs, and older
    multi-output checkpoints.
    """
    if not isinstance(model_output, tuple):
        return model_output, None

    if (
        len(model_output) == 2
        and torch.is_tensor(model_output[0])
        and torch.is_tensor(model_output[1])
        and model_output[1].ndim == 1
    ):
        return model_output[0], model_output[1]

    if task == "top":
        return model_output[0], None
    if task in ("down", "bottom"):
        return model_output[1], None
    raise ValueError(f"task must be one of {VALID_ANALYSIS_TYPES}, got {task!r}")


def plot_predictions(true, pred, feature_names, ranges, title, output_path):
    """Save notebook-style 1D overlays and 2D true-versus-predicted plots."""
    num_features = len(feature_names)
    fig, axes = plt.subplots(
        2,
        num_features,
        figsize=(6 * num_features, 10),
        squeeze=False,
    )

    for index, (feature_name, feature_range) in enumerate(zip(feature_names, ranges)):
        true_values = np.asarray(true[:, index]).ravel()
        pred_values = np.asarray(pred[:, index]).ravel()
        finite = np.isfinite(true_values) & np.isfinite(pred_values)
        true_values = true_values[finite]
        pred_values = pred_values[finite]

        ax_1d = axes[0, index]
        true_counts, _, _ = ax_1d.hist(
            true_values,
            histtype="step",
            color="r",
            label="True Distribution",
            bins=50,
            range=feature_range,
        )
        ax_1d.hist(
            pred_values,
            histtype="step",
            color="b",
            label="Predicted Distribution",
            bins=50,
            range=feature_range,
        )
        ax_1d.set_title(f"1D Comparison: {feature_name}")
        ax_1d.set_xlabel(feature_name, loc="right")
        ax_1d.set_ylim(bottom=0, top=max(1.0, float(true_counts.max()) * 1.2))
        ax_1d.legend()

        ax_2d = axes[1, index]
        hist = ax_2d.hist2d(
            pred_values,
            true_values,
            bins=100,
            norm=LogNorm(),
            range=(feature_range, feature_range),
        )
        ax_2d.set_title(f"2D Comparison: {feature_name}")
        ax_2d.set_xlabel(f"Predicted {feature_name}", loc="right")
        ax_2d.set_ylabel(f"True {feature_name}", loc="top")
        if len(true_values) > 1:
            score = r2_score(true_values, pred_values)
        else:
            score = float("nan")
        ax_2d.text(
            0.04,
            0.96,
            f"$R^2$: {score:.3f}",
            transform=ax_2d.transAxes,
            va="top",
            backgroundcolor="white",
            color="black",
        )
        fig.colorbar(hist[3], ax=ax_2d)

    fig.suptitle(title, fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_dataset(dataset_path, sample, model, device, task, batch_size, output_dir):
    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"{sample} dataset not found: {dataset_path}")

    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    true_batches = []
    pred_batches = []
    kappa_batches = []

    model.eval()
    with torch.inference_mode():
        for batch in data_loader:
            probe_jet, constituents, event = (tensor.to(device) for tensor in batch[:3])
            target = select_target(batch, task)

            model_output = model(probe_jet, constituents, event)
            pred, kappa = select_prediction(model_output, task)

            true_batches.append(target.cpu().numpy())
            pred_batches.append(pred.cpu().numpy())
            if kappa is not None:
                kappa_batches.append(kappa.cpu().numpy())

    if not true_batches:
        raise ValueError(f"{sample} dataset is empty: {dataset_path}")

    true_values = np.concatenate(true_batches, axis=0)
    pred_values = np.concatenate(pred_batches, axis=0)

    np.save(output_dir / f"true_{task}_{sample}.npy", true_values)
    np.save(output_dir / f"pred_{task}_{sample}.npy", pred_values)
    if kappa_batches:
        kappa_values = np.concatenate(kappa_batches, axis=0)
        np.save(output_dir / f"pred_kappa_{task}_{sample}.npy", kappa_values)

    plot_predictions(
        true_values,
        pred_values,
        TASK_FEATURES[task],
        TASK_RANGES[task],
        f"{task.capitalize()} Results: {sample}",
        output_dir / f"{task}_comparison_{sample}.png",
    )

    print(f"{sample}: evaluated {len(dataset)} events from {dataset_path}")


def main():
    args = parse_args()
    task = validate_analysis_type(args.task)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    model_path = Path(args.model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = torch.load(model_path, map_location=device, weights_only=False)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "Expected a serialized torch.nn.Module checkpoint saved by New_Training.py."
        )

    checkpoint_task = getattr(model, "analysis_type", None)
    if checkpoint_task is not None and checkpoint_task != task:
        raise ValueError(
            f"Checkpoint was trained for analysis_type={checkpoint_task!r}, "
            f"but --task={task!r} was requested"
        )
    model.analysis_type = task
    model = model.to(device)

    print(f"Using device: {device}")
    print(f"Evaluating task: {task}")
    print(f"Checkpoint loss type: {getattr(model, 'loss_type', 'unknown')}")
    print(f"Saving NumPy arrays to: {output_dir}")

    evaluate_dataset(args.left_dataset, "L", model, device, task, args.batch_size, output_dir)
    evaluate_dataset(args.right_dataset, "R", model, device, task, args.batch_size, output_dir)
    print(f"Saved comparison plots to: {output_dir}")


if __name__ == "__main__":
    main()
