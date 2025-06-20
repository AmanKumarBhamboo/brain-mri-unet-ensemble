# Brain MRI U-Net Segmentation

This project implements a U-Net-based deep learning pipeline for brain tumor segmentation on MRI scans. It covers data preprocessing, augmentation, model training, and evaluation using PyTorch and Albumentations.

---

## Table of Contents
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Acknowledgements](#acknowledgements)

---

## Project Structure
```
BRAIN MRI U-Net/
├── archive/                # Raw MRI data (organized by patient/study)
│   └── kaggle_3m/          # Main dataset (TCGA, etc.)
├── processed_data/         # Preprocessed data (train/val/test splits)
│   ├── train/
│   ├── val/
│   └── test/
├── model.ipynb             # Main notebook for model training and evaluation
├── dataset.ipynb           # Notebook for data preprocessing
├── unet_model1.py          # U-Net model implementation (version 1)
├── unet_model2.py          # U-Net model implementation (version 2, with MC Dropout)
├── final_model1.pth        # Saved trained model (version 1)
├── final_model2.pth        # Saved trained model (version 2)
├── checkpoints/            # Model checkpoints
└── ...                     # Other supporting files and directories
```

---

## Dataset
- **Source:** TCGA-LGG (Low Grade Glioma) MRI dataset, organized by patient.
- **Raw Data:** Located in `archive/kaggle_3m/`, with each patient in a separate folder containing `.tif` images and corresponding `_mask.tif` files.
- **Preprocessed Data:** Saved as `.npy` files in `processed_data/`, split into `train`, `val`, and `test` sets.

---

## Preprocessing
- **Performed in:** `dataset.ipynb`
- **Steps:**
  - Patient-wise splitting into train/val/test
  - Image resizing (default: 128x128 or 256x256)
  - Min-max normalization of images
  - Binarization of masks
  - Saving as `.npy` files for efficient loading

---

## Model Architecture
- **U-Net**: Encoder-decoder architecture for semantic segmentation.
- **Variants**: 
  - `unet_model1.py`: Standard U-Net
  - `unet_model2.py`: U-Net with Monte Carlo Dropout for uncertainty estimation

---

## Training
- **Main notebook:** `model.ipynb`
- **Features:**
  - Data augmentation using Albumentations
  - Dice loss and BCE loss for segmentation
  - Early stopping and learning rate scheduling
  - Model checkpointing and saving

---

## Evaluation
- **Metrics:** Dice coefficient, loss curves, and qualitative mask predictions
- **Visualization:** Side-by-side display of MRI slices and predicted masks

---

## Usage
### 1. **Preprocess the Data**
Open and run `dataset.ipynb` to:
- Split the dataset
- Resize and normalize images
- Save processed `.npy` files

### 2. **Train the Model**
Open and run `model.ipynb` to:
- Load the processed data
- Train the U-Net model
- Save the best model weights

### 3. **Evaluate and Visualize**
- Use the evaluation cells in `model.ipynb` to assess model performance and visualize results.

---

## Requirements
- Python 3.8+
- PyTorch
- Albumentations
- scikit-learn
- numpy
- matplotlib
- tqdm
- pillow



---

## Acknowledgements
- [TCGA-LGG Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Albumentations](https://albumentations.ai/)

---

