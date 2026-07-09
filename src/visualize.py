import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.dataset import MovingMNISTDataset
from src.model import PredictionModel

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "mnist_test_seq.npy")
BEST_MODEL_PATH = os.path.join(ROOT_DIR, "results", "best_model.pth")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")


def _to_uint8(frame_tensor):
    return (frame_tensor.squeeze(0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


@torch.no_grad()
def generate_comparison_grid(model, dataset, n_samples=5, device="cpu"):
    """Save one PNG per sample with 3 rows: input frames, real target frames, predicted frames."""
    model.eval()
    indices = random.sample(range(len(dataset)), n_samples)
    row_labels = ["Input", "Real", "Predicted"]

    saved_paths = []
    for sample_idx, idx in enumerate(indices):
        inputs, targets = dataset[idx]
        inputs_batch = inputs.unsqueeze(0).to(device)
        predictions = model(inputs_batch, future_frames=targets.shape[0]).clamp(0, 1).squeeze(0).cpu()

        rows = [inputs, targets, predictions]
        n_frames = inputs.shape[0]

        fig, axes = plt.subplots(3, n_frames, figsize=(n_frames, 3.3))
        for r in range(3):
            for f in range(n_frames):
                ax = axes[r, f]
                ax.imshow(rows[r][f, 0].numpy(), cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                if f == 0:
                    ax.set_ylabel(row_labels[r], fontsize=10)

        fig.suptitle(f"Sample {sample_idx}")
        fig.tight_layout()

        out_path = os.path.join(RESULTS_DIR, f"comparison_sample_{sample_idx}.png")
        fig.savefig(out_path)
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


@torch.no_grad()
def generate_prediction_gif(model, sequence, device="cpu", out_path=None, duration=200):
    """Save a side-by-side (real | predicted) animated GIF for a single (input, target) sequence."""
    model.eval()
    inputs, targets = sequence
    inputs_batch = inputs.unsqueeze(0).to(device)
    predictions = model(inputs_batch, future_frames=targets.shape[0]).clamp(0, 1).squeeze(0).cpu()

    frames = []
    for f in range(targets.shape[0]):
        real_img = _to_uint8(targets[f])
        pred_img = _to_uint8(predictions[f])
        separator = np.full((real_img.shape[0], 4), 255, dtype=np.uint8)
        combined = np.concatenate([real_img, separator, pred_img], axis=1)
        frames.append(Image.fromarray(combined).resize((combined.shape[1] * 4, combined.shape[0] * 4), Image.NEAREST))

    out_path = out_path or os.path.join(RESULTS_DIR, "prediction.gif")
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    return out_path


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = MovingMNISTDataset(DATA_PATH, split="val")

    model = PredictionModel(in_channels=1, hidden_channels=[128, 128, 128, 128]).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    grid_paths = generate_comparison_grid(model, val_dataset, n_samples=5, device=device)
    print("Grilles de comparaison sauvegardées :")
    for path in grid_paths:
        print(f"  - {path}")

    gif_sequence = val_dataset[random.randint(0, len(val_dataset) - 1)]
    gif_path = generate_prediction_gif(model, gif_sequence, device=device)
    print(f"GIF de prédiction sauvegardé : {gif_path}")


if __name__ == "__main__":
    main()
