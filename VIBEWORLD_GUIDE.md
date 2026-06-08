# VibeWorld — Guide de développement complet

> Prédiction de frames vidéo par World Model visuel  
> Stack : Python 3.10+ · PyTorch · Gradio · Weights & Biases

---

## Structure du projet

```
vibeworld/
├── data/
│   └── download.py          # Script de téléchargement du dataset
├── src/
│   ├── dataset.py           # Dataset PyTorch
│   ├── model.py             # Architecture ConvLSTM
│   ├── train.py             # Boucle d'entraînement
│   ├── evaluate.py          # Évaluation et métriques
│   └── visualize.py         # Génération des visualisations
├── notebooks/
│   └── exploration.ipynb    # Exploration du dataset
├── demo/
│   └── app.py               # Interface Gradio
├── results/
│   └── .gitkeep
├── requirements.txt
├── README.md
└── report.md
```

---

## Pull Requests

Le projet est découpé en **6 pull requests** indépendantes à réaliser dans l'ordre.

---

## PR 1 — Setup & Dataset

### Objectif
Mettre en place le projet, installer les dépendances et charger le dataset **Moving MNIST**.

### Pourquoi Moving MNIST
Moving MNIST est le dataset standard pour la prédiction de frames vidéo. Il contient des séquences de 20 frames (64×64 px, niveaux de gris) de chiffres MNIST en mouvement. Il est léger (~800 MB), facile à charger et largement utilisé dans la littérature sur les world models.

### Étapes

**1. Initialiser le repo**
```bash
git init vibeworld
cd vibeworld
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate
```

**2. Créer `requirements.txt`**
```
torch>=2.0.0
torchvision>=0.15.0
numpy
matplotlib
tqdm
wandb
gradio
Pillow
```

**3. Créer `data/download.py`**

Script qui télécharge Moving MNIST depuis `http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy` et le sauvegarde dans `data/raw/`.

Le fichier `.npy` a la forme `(20, 10000, 64, 64)` :
- 20 = nombre de frames par séquence
- 10000 = nombre de séquences
- 64×64 = résolution

**4. Créer `src/dataset.py`**

Classe `MovingMNISTDataset(torch.utils.data.Dataset)` qui :
- Charge le fichier `.npy`
- Retourne des tuples `(input_frames, target_frames)` où :
  - `input_frames` = les 10 premières frames → shape `(10, 1, 64, 64)`, normalisées entre 0 et 1
  - `target_frames` = les 10 frames suivantes → shape `(10, 1, 64, 64)`, normalisées entre 0 et 1
- Implémente un split train/val (80/20)

**5. Créer `notebooks/exploration.ipynb`**

Notebook qui :
- Charge le dataset
- Affiche 3 séquences complètes sous forme de grille d'images (20 frames côte à côte)
- Affiche les statistiques basiques (min, max, moyenne, shape)

### Résultat attendu
- Le dataset se charge sans erreur
- On peut afficher des séquences dans le notebook
- `len(dataset)` retourne 10000

---

## PR 2 — Architecture ConvLSTM

### Objectif
Implémenter l'architecture **ConvLSTM** dans `src/model.py`.

### Pourquoi ConvLSTM
Le ConvLSTM (Convolutional LSTM) est l'architecture de référence pour la prédiction de séquences spatiales. Contrairement à un LSTM classique qui opère sur des vecteurs, le ConvLSTM applique des convolutions dans ses portes, ce qui lui permet de capturer à la fois les dépendances temporelles et les structures spatiales des frames. C'est le building block de nombreux world models visuels.

### Étapes

**1. Implémenter `ConvLSTMCell`**

Cellule LSTM dont toutes les opérations matricielles sont remplacées par des convolutions 2D.

Paramètres :
- `in_channels` : nombre de canaux en entrée
- `hidden_channels` : nombre de canaux cachés
- `kernel_size` : taille du noyau de convolution (default: 3)

La cellule reçoit `(x, h_prev, c_prev)` et retourne `(h, c)`.

Les 4 portes (input, forget, output, cell) sont calculées avec une seule convolution puis splitées.

**2. Implémenter `ConvLSTM`**

Module qui empile plusieurs `ConvLSTMCell` en séquence.

Paramètres :
- `in_channels` : 1 (niveaux de gris)
- `hidden_channels` : liste du nombre de canaux par couche, ex: `[64, 64, 64]`
- `kernel_size` : 3
- `num_layers` : longueur de `hidden_channels`

Prend en entrée un tenseur de shape `(batch, seq_len, channels, height, width)` et retourne les états cachés pour chaque timestep.

**3. Implémenter `PredictionModel`**

Module complet qui :
- Encode les frames d'entrée via le ConvLSTM
- Décode l'état caché final en frames de sortie via une tête de prédiction `Conv2d` + `Sigmoid`
- En mode autorégressif : utilise chaque frame prédite comme entrée pour prédire la suivante

Architecture de la tête de décodage :
```
Conv2d(hidden_channels[-1], 64, 3, padding=1) → ReLU
Conv2d(64, 1, 1) → Sigmoid
```

### Résultat attendu
```python
model = PredictionModel(in_channels=1, hidden_channels=[64, 64, 64])
x = torch.randn(2, 10, 1, 64, 64)  # batch=2, 10 frames d'entrée
out = model(x, future_frames=10)
print(out.shape)  # → (2, 10, 1, 64, 64)
```

---

## PR 3 — Entraînement

### Objectif
Implémenter la boucle d'entraînement dans `src/train.py` avec suivi via **Weights & Biases**.

### Étapes

**1. Configurer les hyperparamètres**

Définir un dictionnaire de config en haut du fichier :
```python
config = {
    "batch_size": 16,
    "lr": 1e-3,
    "epochs": 20,
    "hidden_channels": [64, 64, 64],
    "input_frames": 10,
    "target_frames": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
```

**2. Implémenter la boucle d'entraînement**

- Optimiseur : `Adam` avec lr=1e-3
- Loss : `MSELoss` entre les frames prédites et les frames cibles
- Scheduler : `ReduceLROnPlateau` sur la val loss (patience=3)
- Initialiser un run `wandb` et logger à chaque epoch :
  - `train_loss`
  - `val_loss`
  - Grille d'images comparant frames réelles vs prédites (toutes les 5 epochs)

**3. Implémenter la sauvegarde du modèle**

Sauvegarder le meilleur modèle (val loss minimale) dans `results/best_model.pth` à chaque fois qu'une amélioration est détectée.

**4. Point d'entrée**

Le script doit pouvoir être lancé directement :
```bash
python src/train.py
```

### Résultat attendu
- L'entraînement tourne sans erreur
- La loss diminue au fil des epochs
- Le modèle est sauvegardé dans `results/`
- Les courbes de loss sont visibles sur le dashboard wandb

---

## PR 4 — Évaluation & Visualisations

### Objectif
Évaluer le modèle entraîné et générer des visualisations dans `src/evaluate.py` et `src/visualize.py`.

### Étapes

**1. Implémenter `src/evaluate.py`**

Charger le meilleur modèle (`results/best_model.pth`) et calculer sur le set de validation :
- **MSE** (Mean Squared Error) : mesure l'erreur pixel par pixel
- **SSIM** (Structural Similarity Index) : mesure la similarité structurelle perçue, plus proche de la perception humaine. Utiliser `torchmetrics.image.StructuralSimilarityIndexMeasure` ou implémenter manuellement.

Afficher un tableau récapitulatif des métriques.

**2. Implémenter `src/visualize.py`**

Fonction `generate_comparison_grid(model, dataset, n_samples=5)` qui :
- Sélectionne `n_samples` séquences aléatoires du dataset
- Pour chaque séquence, génère une image avec 3 lignes :
  - Ligne 1 : frames d'entrée (10 frames)
  - Ligne 2 : frames réelles cibles (10 frames)
  - Ligne 3 : frames prédites par le modèle (10 frames)
- Sauvegarde chaque grille dans `results/comparison_sample_N.png`

Fonction `generate_prediction_gif(model, sequence)` qui :
- Génère un GIF animé comparant la prédiction et la réalité côte à côte
- Sauvegarde dans `results/prediction.gif`

### Résultat attendu
- Les fichiers `results/comparison_sample_*.png` sont générés
- Le fichier `results/prediction.gif` est généré
- Les métriques MSE et SSIM sont affichées dans le terminal

---

## PR 5 — Démo interactive

### Objectif
Créer une interface **Gradio** dans `demo/app.py` permettant de tester le modèle interactivement.

### Étapes

**1. Implémenter `demo/app.py`**

Interface avec :
- **Entrée** : upload d'un fichier `.npy` contenant une séquence, ou sélection d'un exemple aléatoire du dataset via un bouton
- **Slider** : choix du nombre de frames à prédire (1 à 10)
- **Sortie** : affichage côte à côte des frames réelles et prédites sous forme de grille d'images + le GIF animé

**2. Ajouter un bouton "Exemple aléatoire"**

Qui tire une séquence au hasard dans le dataset de validation et lance la prédiction automatiquement sans que l'utilisateur ait à uploader quoi que ce soit.

**3. Lancement**

```bash
python demo/app.py
```

L'interface doit être accessible sur `http://localhost:7860`.

### Résultat attendu
- L'interface se lance sans erreur
- On peut générer des prédictions en un clic via le bouton "Exemple aléatoire"
- Les résultats s'affichent clairement

---

## PR 6 — Rapport & README

### Objectif
Rédiger la documentation finale du projet.

### `README.md`

Doit contenir :
- Description du projet en 3-4 lignes
- Instructions d'installation (`pip install -r requirements.txt`)
- Instructions de téléchargement du dataset (`python data/download.py`)
- Instructions d'entraînement (`python src/train.py`)
- Instructions de lancement de la démo (`python demo/app.py`)
- 2-3 exemples visuels de prédictions (images depuis `results/`)

### `report.md`

Rapport technique structuré ainsi :

1. **Introduction** — Contexte des World Models, motivation du projet
2. **Dataset** — Description de Moving MNIST, preprocessing appliqué
3. **Architecture** — Explication du ConvLSTM, choix de design, schéma de l'architecture
4. **Entraînement** — Hyperparamètres, fonction de loss, stratégie d'optimisation
5. **Résultats** — Métriques obtenues (MSE, SSIM), visualisations, analyse qualitative
6. **Conclusion** — Limites du modèle, pistes d'amélioration possibles
7. **Références** — Articles cités (Shi et al. 2015 pour ConvLSTM, Ha & Schmidhuber 2018 pour World Models)

### Résultat attendu
- Le README permet à quelqu'un de relancer le projet de zéro
- Le rapport fait 2-3 pages et couvre toutes les sections listées

---

## Références clés

- **ConvLSTM** : Shi et al., *Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting*, NeurIPS 2015
- **World Models** : Ha & Schmidhuber, *World Models*, 2018 — https://worldmodels.github.io
- **Moving MNIST** : Srivastava et al., *Unsupervised Learning of Video Representations using LSTMs*, ICML 2015
