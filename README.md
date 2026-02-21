# Fashion-MNIST Image Classifier with PyTorch MLP Test Accuracy up to 92.02%

A PyTorch-based **Multi-Layer Perceptron (MLP)** that classifies images from the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset into 10 fashion categories.

**Test accuracy: 92.02%** on the held-out test set (5,000 samples).

## Overview

This project trains a fully connected neural network to classify 28×28 grayscale fashion images. The dataset has 70,000 images (60,000 for training, 10,000 for test). The official test set is split into validation and test sets of 5,000 samples each for training and final evaluation.

**Classes:** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

## Model Architecture

- **Input:** Flattened 784-dimensional vector (28×28 pixels).
- **Hidden layers:** Four fully connected blocks with BatchNorm and GELU activation:
  - fc1: 784 → 144  
  - fc2: 144 → 128  
  - fc3: 128 → 128 with a residual (skip) connection  
  - fc4: 128 → 128  
- **Output:** 10-way classification (logits).
- **Regularization:** Dropout (0.3), gradient clipping (max norm 1.0), Kaiming initialization.
- **Parameters:** ~167K trainable.

## Training Setup

| Component        | Choice |
|-----------------|--------|
| Loss            | Cross-entropy with label smoothing (0.01) |
| Optimizer       | AdamW (lr=5e-4, weight decay 0.19 for non-BN/bias params) |
| Scheduler       | CosineAnnealingLR over 120 epochs (min lr ≈ 2e-8) |
| Batch size      | 128    |
| Epochs          | 120    |

**Data augmentation (training only):** Random horizontal flip, random affine (translate + scale). Both train and test use the same normalization (Fashion-MNIST mean/std).

## Results

- **Test accuracy: 92.02%** on the held-out test set (5,000 samples).
- Best validation accuracy: ~91.56% (checkpoint at epoch 97).
- Training time is on the order of a few minutes on GPU (e.g. ~6+ min for 120 epochs).
- Per-epoch checkpoints are saved under `Models/MLP_model_{epoch}.pth`.

The notebook also includes:
- Training/validation loss and accuracy curves  
- Test-set evaluation  
- Confusion matrix and per-class metrics (precision, recall, F1)  
- Sample predictions and visualizations  

## Requirements

- Python 3.x  
- PyTorch (with CUDA optional)  
- torchvision  
- torchinfo  
- matplotlib, pandas, numpy, scikit-learn  

Install extra dependency:

```bash
pip install torchinfo
```

Fashion-MNIST is downloaded automatically via `torchvision.datasets.FashionMNIST` on first run.

## How to Run

1. Clone or download this repository.
2. Install dependencies (see above).
3. Open `MLP3_FashionMNIST.ipynb` in Jupyter or a compatible environment.
4. Run all cells. Training will create a `Models/` directory and save checkpoints there.
5. Use the “best” checkpoint (e.g. by validation accuracy) for test evaluation and plots.

## Repository Contents

- `MLP3_FashionMNIST.ipynb` — Main notebook: data loading, model definition, training, evaluation, and visualizations.
- `README.md` — This file.

## References

- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) — Dataset and description.
- [Image Classifier using PyTorch and Keras](https://medium.com/dataseries/image-classifier-using-pytorch-and-keras-555007a50c2e) — PyTorch/Keras comparison.
- [Basic Syntax of the Markdown elements](https://www.markdownguide.org/basic-syntax/) — Markdown reference.
