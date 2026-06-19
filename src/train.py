import os

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from src.dataset import MovingMNISTDataset
from src.model import PredictionModel

config = {
    "batch_size": 16,
    "lr": 1e-3,
    "epochs": 20,
    "hidden_channels": [128, 128, 128, 128],
    "input_frames": 10,
    "target_frames": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "mnist_test_seq.npy")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pth")


def comparison_grid(targets, predictions, n_samples=4):
    """Stack target/prediction sequences into a single grid image (one pair of rows per sample)."""
    targets = targets[:n_samples].detach().cpu()
    predictions = predictions[:n_samples].detach().cpu()

    rows = []
    for i in range(targets.shape[0]):
        rows.append(make_grid(targets[i], nrow=targets.shape[1]))
        rows.append(make_grid(predictions[i], nrow=predictions.shape[1]))
    return torch.cat(rows, dim=1)


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    num_batches = 0

    with torch.set_grad_enabled(is_train):
        for inputs, targets in tqdm(loader, leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs, future_frames=targets.shape[1])
            loss = criterion(predictions, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def train():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device(config["device"])

    train_dataset = MovingMNISTDataset(DATA_PATH, split="train")
    val_dataset = MovingMNISTDataset(DATA_PATH, split="val")
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    model = PredictionModel(in_channels=1, hidden_channels=config["hidden_channels"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)
    criterion = nn.MSELoss()

    wandb.init(project="vibeworld", config=config)
    best_val_loss = float("inf")

    for epoch in range(1, config["epochs"] + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        log = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}

        if epoch % 5 == 0:
            inputs, targets = next(iter(val_loader))
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                predictions = model(inputs, future_frames=targets.shape[1])
            grid = comparison_grid(targets, predictions)
            log["predictions"] = wandb.Image(grid, caption=f"Epoch {epoch} — top: real, bottom: predicted")

        wandb.log(log)
        print(f"Epoch {epoch}/{config['epochs']} — train_loss: {train_loss:.4f} — val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> New best model saved (val_loss: {val_loss:.4f})")

    wandb.finish()


if __name__ == "__main__":
    train()
