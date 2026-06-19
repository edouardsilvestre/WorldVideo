#!/bin/bash
set -e

cd ~/WorldVideo

echo "=== Vérification GPU ==="
nvidia-smi --query-gpu=name --format=csv,noheader \
  || { echo "ERREUR : nvidia-smi introuvable. Vérifie que tu as bien redémarré après setup_drivers.sh"; exit 1; }

echo "=== Création de l'environnement Python ==="
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip --quiet

echo "=== Installation PyTorch (CUDA 12.1) ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet

echo "=== Installation des dépendances ==="
pip install -r requirements.txt --quiet

echo "=== Vérification CUDA dans PyTorch ==="
python3 -c "import torch; ok = torch.cuda.is_available(); print('CUDA:', ok, '—', torch.cuda.get_device_name(0) if ok else 'CPU seulement')"

echo "=== Lancement de l'entraînement ==="
WANDB_MODE=disabled python src/train.py
