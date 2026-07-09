import os

import torch
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm

from src.dataset import MovingMNISTDataset
from src.model import PredictionModel

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "mnist_test_seq.npy")
BEST_MODEL_PATH = os.path.join(ROOT_DIR, "results", "best_model.pth")


@torch.no_grad()
def evaluate(model, loader, device):
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    total_mse = 0.0
    total_ssim = 0.0
    num_batches = 0

    for inputs, targets in tqdm(loader, desc="Evaluating"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        predictions = model(inputs, future_frames=targets.shape[1]).clamp(0, 1)

        mse = torch.nn.functional.mse_loss(predictions, targets)

        b, t, c, h, w = predictions.shape
        ssim = ssim_metric(predictions.view(b * t, c, h, w), targets.view(b * t, c, h, w))

        total_mse += mse.item()
        total_ssim += ssim.item()
        num_batches += 1

    return total_mse / num_batches, total_ssim / num_batches


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = MovingMNISTDataset(DATA_PATH, split="val")
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = PredictionModel(in_channels=1, hidden_channels=[128, 128, 128, 128]).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()

    mse, ssim = evaluate(model, val_loader, device)

    print("\n=== Résultats de l'évaluation (set de validation) ===")
    print(f"{'Métrique':<10} | {'Valeur':>10}")
    print(f"{'-' * 10} | {'-' * 10}")
    print(f"{'MSE':<10} | {mse:>10.4f}")
    print(f"{'SSIM':<10} | {ssim:>10.4f}")


if __name__ == "__main__":
    main()
