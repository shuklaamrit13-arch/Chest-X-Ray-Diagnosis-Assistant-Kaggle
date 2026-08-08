# AI Chest X-Ray Diagnosis Assistant

A deep learning model that looks at a chest X-ray and predicts the probability of **14 thoracic pathologies** (Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia), plus a heatmap showing *where* in the image it looked.

Built on the **NIH ChestX-ray14** dataset, using the same backbone as the CheXNet paper (Rajpurkar et al., 2017).

> ⚠️ Research / educational demo only — **not** a diagnostic tool. Never use this to make real medical decisions.

---

## 🧠 ML Techniques Used (Explained Simply)

### 1. Transfer Learning
Instead of training a model from scratch (which needs millions of images), we start with **DenseNet-121**, a model already trained on 1.2 million everyday photos (ImageNet). It already knows how to detect edges, shapes, and textures — we just teach it to apply that skill to X-rays. This makes training much faster and more accurate with limited data.

### 2. Convolutional Neural Network (CNN) — DenseNet-121
A CNN is a type of neural network built to "see" images. It scans the image in small patches, learning to detect simple patterns first (edges, curves) and combining them into complex ones (organs, abnormalities) in deeper layers. DenseNet specifically connects every layer to every other layer, which helps information and gradients flow better through a deep network.

### 3. Multi-Label Classification
A normal classifier picks **one** answer (e.g., "cat" or "dog"). Here, a single X-ray can show **several** conditions at once (e.g., both Effusion and Infiltration), so the model outputs 14 independent yes/no probabilities — one per pathology — instead of a single label.

### 4. Sigmoid Activation
For multi-label problems, each output uses a **sigmoid function**, which squashes a raw score into a probability between 0 and 1 for that specific condition, independently of the others. This is different from "softmax," which is used when only one class can be true at a time.

### 5. Class-Weighted Loss (Handling Imbalanced Data)
Some conditions (like Hernia) appear in less than 1% of images, while others (like Infiltration) appear in ~18%. If trained normally, the model would just learn to always predict "no disease" and still look accurate. To fix this, rare conditions are given **more weight** in the loss function (`BCEWithLogitsLoss` with `pos_weight`), forcing the model to pay attention to them too.

### 6. Data Augmentation
During training, images are randomly flipped and slightly rotated. This artificially creates more variety in the training data, helping the model generalize better instead of memorizing exact images.

### 7. Optimizer: AdamW + Learning Rate Scheduling
**AdamW** is the algorithm that adjusts the model's internal parameters step-by-step to reduce errors. **ReduceLROnPlateau** automatically shrinks the learning rate when progress stalls, helping the model fine-tune itself as training goes on.

### 8. Patient-Level Data Splitting
The dataset is split into train/validation/test sets **by patient**, not by image. Since one patient can have multiple X-rays, splitting by image alone could let the model "see" a patient during training and be tested on another scan of that same patient — artificially inflating performance. Splitting by patient prevents this leakage.

### 9. Evaluation Metrics
- **AUC-ROC**: Measures how well the model ranks sick vs. healthy cases across all possible decision thresholds. A score of 0.5 = random guessing, 1.0 = perfect.
- **Precision, Recall, F1-score**: Measured at a fixed decision threshold (50%) to show real-world usefulness — e.g., "of all the times it flagged Pneumonia, how often was it right?" (precision) vs. "of all the real Pneumonia cases, how many did it catch?" (recall).
- **Exact-match accuracy**: The strict measure of getting *all 14* labels correct simultaneously on one image.

### 10. Grad-CAM (Explainable AI)
A probability score alone isn't trustworthy without knowing *why* the model made that call. **Grad-CAM** (Gradient-weighted Class Activation Mapping) produces a heatmap over the X-ray showing which regions most influenced a given prediction — so you can visually check the model is looking at, say, the heart border for a Cardiomegaly call, rather than an unrelated artifact like text or a pacemaker.

---

## 🏗️ Pipeline Overview

1. **Download data** — NIH ChestX-ray14 dataset via `kagglehub`.
2. **Prepare data** — Parse labels, build multi-hot vectors, apply official patient-level split.
3. **Explore data** — Visualize label frequency and class imbalance.
4. **Build Dataset/DataLoader** — Resize, normalize, and augment images for PyTorch.
5. **Define model** — DenseNet-121 backbone + 14-way sigmoid output head.
6. **Train** — Weighted BCE loss, AdamW optimizer, LR scheduler, save best checkpoint by validation AUC.
7. **Evaluate** — Compute AUC, precision, recall, F1 on the held-out test set.
8. **Explain predictions** — Generate Grad-CAM heatmaps.
9. **Run inference** — Try the model on sample or custom X-ray images.

## ⚙️ Requirements

- GPU (recommended: Kaggle Notebooks or Google Colab)
- Python packages: `torch`, `torchvision`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `kagglehub`, `Pillow`

## 🚀 Quick Start

1. Open the notebook in **Kaggle Notebooks** or **Google Colab** (GPU + internet required).
2. Run the cells top to bottom.
3. For a fast first run, use the dataset's `sample/` subfolder (~1GB, ~5,600 images) instead of the full ~42GB dataset.
4. Increase `EPOCHS` once you've confirmed the pipeline works end-to-end.

## 📈 Results

These are the actual results from a 10-epoch training run on the full dataset in this notebook.

**Training progress** — validation mean AUC improved steadily, peaking at epoch 9:

| Epoch | Train Loss | Val Mean AUC |
|-------|-----------|--------------|
| 1     | 1.1259    | 0.7856       |
| 3     | 0.9805    | 0.8094       |
| 5     | 0.9222    | 0.8272       |
| 9     | 0.8156    | **0.8308** (best) |
| 10    | 0.7806    | 0.8178       |

**Final test set: mean AUC = 0.8023**

| Pathology | AUC | Pathology | AUC |
|---|---|---|---|
| Hernia | 0.925 | Effusion | 0.817 |
| Emphysema | 0.902 | Fibrosis | 0.814 |
| Cardiomegaly | 0.884 | Mass | 0.814 |
| Pneumothorax | 0.852 | Pleural Thickening | 0.761 |
| Edema | 0.841 | Atelectasis | 0.757 |
| Nodule | 0.745 | Consolidation | 0.742 |
| Pneumonia | 0.697 | Infiltration | 0.680 |

**Accuracy at 0.5 threshold:** mean per-label accuracy **69.3%**, exact-match (all 14 correct at once) **5.3%** — expected to be low since it requires every single label to match simultaneously.

This run lands close to the published CheXNet benchmark (mean AUC ≈ 0.84) despite only 10 epochs, with the best individual scores on Hernia, Emphysema, and Cardiomegaly, and the toughest findings being Infiltration and Pneumonia (both notoriously subtle/ambiguous even for radiologists).

### Label distribution in the dataset
![Label frequency](label_frequency.png)

### Grad-CAM examples (model prediction + where it looked)
![Grad-CAM examples](gradcam_examples.png)

Sample outputs from the trained model on held-out test images:
- Top finding: **Mass (96.6%)**, also flagging Nodule (82.3%), Hernia (70.6%), Pneumothorax (53.0%)
- Top finding: **Consolidation (77.6%)**, also flagging Effusion (72.3%), Mass (72.0%), and four others above threshold

## 🔮 Possible Improvements

- Try EfficientNet or a Vision Transformer (ViT) backbone and compare results.
- Add test-time augmentation for more stable predictions.
- Tune per-class thresholds instead of one global 0.5 cutoff.
- Validate against an external, radiologist-labeled test set before trusting any output clinically.

## ⚠️ Disclaimer

This project is for **research and educational purposes only**. It is not a certified medical device and must not be used for real clinical diagnosis. Any real-world use should involve review by a qualified radiologist.
