"""Evaluate a U-trained model on independent L and R datasets.

Example:
    python model/evaluate.py \
        --left-dataset model/WS_L_1M_R1_5_pT250/datasets_AllFrame/dataset_combined.pt \
        --right-dataset model/WS_R_1M_R1_5_pT250/datasets_AllFrame/dataset_combined.pt \
        --model model/WS_U_10M_R1_5_pT250/training_down_60epoch_64embed/model_final.torch \
        --task down \
        --output-dir workspaces/eval_down

The output directory contains NumPy arrays for true/predicted top and
task-specific quark vectors, and true/reconstructed cos(theta), separately for
L and R. It also contains per-sample top/quark comparison plots and a combined
L/R cos(theta) plot. Each input dataset is evaluated in full; no split is made.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from DataLoader_Parallel import CustomDataset  # noqa: F401 - needed to load serialized datasets
from new_model import Model  # noqa: F401 - needed to load serialized model checkpoints


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a model trained on unpolarized U data on L and R datasets."
    )
    parser.add_argument("--left-dataset", required=True, help="Path to the L dataset_combined.pt")
    parser.add_argument("--right-dataset", required=True, help="Path to the R dataset_combined.pt")
    parser.add_argument(
        "--model",
        "--checkpoint",
        dest="model_path",
        required=True,
        help="Path to the U-trained model_final.torch checkpoint",
    )
    parser.add_argument(
        "--task",
        choices=("down", "bottom"),
        default="down",
        help="Which quark target to compare against (default: down)",
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


def reconstructed_cos_theta(top, quark):
    """Cosine between the top momentum and quark directions, event by event."""
    top_momentum = top[:, :3]
    top_norm = np.linalg.norm(top_momentum, axis=1, keepdims=True)
    quark_norm = np.linalg.norm(quark, axis=1, keepdims=True)

    top_direction = np.divide(
        top_momentum,
        top_norm,
        out=np.full_like(top_momentum, np.nan, dtype=np.float64),
        where=top_norm > 0,
    )
    quark_direction = np.divide(
        quark,
        quark_norm,
        out=np.full_like(quark, np.nan, dtype=np.float64),
        where=quark_norm > 0,
    )
    cos_theta = np.sum(top_direction * quark_direction, axis=1)
    # Guard against tiny floating-point excursions outside the cosine range.
    return np.clip(cos_theta, -1.0, 1.0)


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


def plot_cos_theta(results_by_sample, output_path):
    """Save the combined L/R true/predicted cos(theta) histogram."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"L": "r", "R": "b"}
    for sample in ("L", "R"):
        results = results_by_sample[sample]
        color = colors[sample]
        ax.hist(
            results["true_cos_theta"],
            histtype="step",
            bins=30,
            range=(-1, 1),
            color=color,
            linestyle="-",
            label=f"True {{{sample}}}",
        )
        ax.hist(
            results["pred_cos_theta"],
            histtype="step",
            bins=30,
            range=(-1, 1),
            color=color,
            linestyle="--",
            label=f"Pred {{{sample}}}",
        )

    ax.set_title("Cos Theta")
    ax.set_xlabel(r"$\cos\theta$")
    ax.set_ylabel("Events")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_dataset(dataset_path, sample, model, device, task, batch_size, output_dir):
    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"{sample} dataset not found: {dataset_path}")

    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    true_top_batches = []
    pred_top_batches = []
    true_quark_batches = []
    pred_quark_batches = []
    quark_label_index = 4 if task == "down" else 5

    model.eval()
    with torch.inference_mode():
        for batch in data_loader:
            probe_jet, constituents, event = (tensor.to(device) for tensor in batch[:3])
            top_labels = batch[3]
            quark_labels = batch[quark_label_index]

            top_pred, quark_pred, _, _ = model(probe_jet, constituents, event)

            true_top_batches.append(top_labels.cpu().numpy())
            pred_top_batches.append(top_pred.cpu().numpy())
            true_quark_batches.append(quark_labels.cpu().numpy())
            pred_quark_batches.append(quark_pred.cpu().numpy())

    if not true_top_batches:
        raise ValueError(f"{sample} dataset is empty: {dataset_path}")

    true_top = np.concatenate(true_top_batches, axis=0)
    pred_top = np.concatenate(pred_top_batches, axis=0)
    true_quark = np.concatenate(true_quark_batches, axis=0)
    pred_quark = np.concatenate(pred_quark_batches, axis=0)
    true_cos_theta = reconstructed_cos_theta(true_top, true_quark)
    pred_cos_theta = reconstructed_cos_theta(pred_top, pred_quark)

    arrays = {
        f"true_top_{sample}": true_top,
        f"pred_top_{sample}": pred_top,
        f"true_{task}_{sample}": true_quark,
        f"pred_{task}_{sample}": pred_quark,
        f"true_cos_theta_{sample}": true_cos_theta,
        f"pred_cos_theta_{sample}": pred_cos_theta,
    }
    for name, values in arrays.items():
        np.save(output_dir / f"{name}.npy", values)

    top_names = ("top_px", "top_py", "top_pz", "top_e")
    top_ranges = ((-1000, 1000), (-1000, 1000), (-1000, 1000), (0, 1500))
    quark_names = tuple(f"{task}_{component}" for component in ("px", "py", "pz"))
    quark_ranges = ((-1.1, 1.1),) * 3
    plot_predictions(
        true_top,
        pred_top,
        top_names,
        top_ranges,
        f"Top Results: {sample}",
        output_dir / f"top_comparison_{sample}.png",
    )
    plot_predictions(
        true_quark,
        pred_quark,
        quark_names,
        quark_ranges,
        f"{task.capitalize()} Results: {sample}",
        output_dir / f"{task}_comparison_{sample}.png",
    )

    print(f"{sample}: evaluated {len(dataset)} events from {dataset_path}")
    return {
        "true_cos_theta": true_cos_theta,
        "pred_cos_theta": pred_cos_theta,
    }


def main():
    args = parse_args()
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
            "Expected a serialized torch.nn.Module checkpoint. "
            "This script currently supports the full-model checkpoints saved by New_Training.py."
        )
    model = model.to(device)
    print(f"Using device: {device}")
    print(f"Evaluating task: {args.task}")
    print(f"Saving NumPy arrays to: {output_dir}")

    results_by_sample = {}
    results_by_sample["L"] = evaluate_dataset(
        args.left_dataset, "L", model, device, args.task, args.batch_size, output_dir
    )
    results_by_sample["R"] = evaluate_dataset(
        args.right_dataset, "R", model, device, args.task, args.batch_size, output_dir
    )
    plot_cos_theta(results_by_sample, output_dir / "cos_theta_LR.png")
    print(f"Saved comparison plots to: {output_dir}")


if __name__ == "__main__":
    main()
