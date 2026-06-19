# Guide — Lancer l'entraînement (PR3) sur la machine puissante (via clé USB)

Ce guide part de l'état actuel : `src/train.py` est écrit et testé (smoke test OK)
mais pas encore commité. Pas de git/SSH nécessaire entre les deux machines :
tout passe par une clé USB.

## Étape 0 — Préparer la clé USB (sur cette machine)

Copier le projet sur la clé en excluant le venv et les caches (inutiles et
volumineux), mais en gardant le dataset déjà téléchargé pour ne pas dépendre
du réseau de l'autre PC :

```bash
# Remplacer /media/.../USB par le point de montage de ta clé
rsync -av --exclude 'venv' --exclude '__pycache__' --exclude '.git' \
  /home/edouard/delivery/4-Semestre/Hub/WorldVideo/ /media/.../USB/WorldVideo/
```

Vérifier que `WorldVideo/data/raw/mnist_test_seq.npy` (~800 Mo) est bien présent
sur la clé — ça évite d'avoir à relancer `data/download.py`.

---

## Étape 1 — Copier le projet sur l'autre PC

Brancher la clé, copier le dossier `WorldVideo` vers un emplacement local sur
le disque de l'autre machine (ne pas travailler directement depuis la clé,
c'est lent et risqué pour l'écriture des checkpoints) :

```bash
cp -r /media/.../USB/WorldVideo ~/WorldVideo
cd ~/WorldVideo
```

---

## Étape 2 — Environnement Python

```bash
python3 -m venv venv
source venv/bin/activate   # Windows : venv\Scripts\activate
pip install --upgrade pip
```

Installer PyTorch **avec support CUDA** (vérifier d'abord la version CUDA
disponible avec `nvidia-smi`, puis installer la build correspondante, voir
https://pytorch.org/get-started/locally/). Exemple pour CUDA 12.x :

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Puis le reste des dépendances :

```bash
pip install -r requirements.txt
```

Vérifier que CUDA est bien détecté :

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Si `False`, ne pas continuer le run complet — corriger l'install PyTorch
d'abord (sinon ça tournera sur CPU et ce sera très long).

---

## Étape 3 — Vérifier le dataset

```bash
ls -lh data/raw/mnist_test_seq.npy   # doit faire ~800 Mo
```

S'il est absent (oublié sur la clé) et que l'autre PC a un accès internet :

```bash
python data/download.py
```

---

## Étape 4 — Configurer Weights & Biases

```bash
wandb login
```

Coller la clé API (disponible sur https://wandb.ai/authorize, connecté avec
le même compte que sur cette machine). Ça nécessite un accès internet sur
l'autre PC.

---

## Étape 5 — Lancer l'entraînement

```bash
python src/train.py
```

Ce que ça fait (config par défaut dans `src/train.py`) :
- 20 epochs, batch_size=16, lr=1e-3
- Optimiseur Adam + scheduler ReduceLROnPlateau (patience=3)
- Log `train_loss` / `val_loss` sur wandb à chaque epoch
- Log d'une grille d'images (réel vs prédit) toutes les 5 epochs
- Sauvegarde du meilleur modèle dans `results/best_model.pth` à chaque
  amélioration de la val loss

Le terminal affiche une barre de progression par epoch et une ligne
récap (`Epoch X/20 — train_loss: ... — val_loss: ...`).

---

## Étape 6 — Vérifier le résultat

- Dashboard wandb : les courbes `train_loss`/`val_loss` doivent diminuer
- Fichier `results/best_model.pth` doit exister

---

## Étape 7 — Rapatrier le modèle entraîné sur la clé USB

```bash
cp results/best_model.pth /media/.../USB/WorldVideo/results/
```

---

## Étape 8 — Récupérer le résultat sur cette machine

```bash
cp /media/.../USB/WorldVideo/results/best_model.pth \
   /home/edouard/delivery/4-Semestre/Hub/WorldVideo/results/
```

Une fois récupéré, penser à committer `src/train.py` (et éventuellement le
checkpoint si vous voulez le versionner) :

```bash
git add src/train.py
git commit -m "Implémente la boucle d'entraînement avec suivi wandb"
git push
```

`results/best_model.pth` servira ensuite pour la PR4 (évaluation et
visualisations).

---

## Dépannage rapide

- **CUDA out of memory** : réduire `batch_size` dans `config` (ex: 8) dans
  `src/train.py`.
- **`ModuleNotFoundError: No module named 'src'`** : lancer la commande
  depuis la racine du projet (`python src/train.py`, pas
  `cd src && python train.py`).
- **wandb demande de se logger en boucle** : relancer `wandb login`, ou
  exporter `WANDB_API_KEY=<ta clé>` avant de lancer le script.
- **Pas d'accès internet sur l'autre PC** : `wandb` peut tourner en mode
  hors-ligne (`wandb offline` avant de lancer le script, puis
  `wandb sync` plus tard pour uploader les runs une fois reconnecté).
