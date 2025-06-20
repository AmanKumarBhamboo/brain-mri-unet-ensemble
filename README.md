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
├── archive/                # Raw MRI data (organized by patient/study)         # Main dataset (TCGA, etc.)
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

```

---

## Dataset
## 📁 Dataset Information

- **Source:**  
  This dataset is derived from the **TCGA-LGG (The Cancer Genome Atlas - Lower Grade Glioma)** project and includes MRI scans sourced from **The Cancer Imaging Archive (TCIA)**. The data is organized by patient.

- **Description:**  
  The dataset consists of brain MR images with corresponding **manual FLAIR abnormality segmentation masks**. These masks highlight tumor regions for medical image segmentation tasks.

- **Patients:**  
  Includes MRI data from **110 patients** diagnosed with **lower-grade gliomas**. Each patient has available **FLAIR sequences** and **genomic cluster** information.

- **Genomic Information:**  
  Tumor genomic cluster data and patient metadata are available in the `data.csv` file.  
  For more details on the genomic analysis, refer to the publication:  
  **“Comprehensive, Integrative Genomic Analysis of Diffuse Lower-Grade Gliomas”**  
  *(The New England Journal of Medicine, 2015)*  
  🔗 [Link to publication](https://www.nejm.org/doi/full/10.1056/NEJMoa1402121)

- **Raw Data Directory:**  
  Path: `archive/lgg-mri-segmentation/kaggle_3m/`  
  Each patient's folder contains:
  - `.tif` files — Brain MRI slices  
  - `_mask.tif` files — Corresponding binary segmentation masks

- **Applications:**  
  - Brain tumor segmentation  
  - Radiogenomic analysis  
  - Medical imaging model development

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

