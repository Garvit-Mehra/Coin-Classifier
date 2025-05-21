# 🪙 Coin Classifier using EfficientNetB0

A deep learning-based image classifier for recognizing the **currency**, **denomination**, and **country** of coins using transfer learning on coin images.

---

## 🧠 Project Overview

This project aims to solve a fine-grained image classification problem where each coin belongs to a specific category like:
```“1 Cent, Australian Dollar, Australia”```
It uses a pre-trained **EfficientNetB0** model fine-tuned on the dataset, achieving strong performance with high training efficiency.

---

## 📂 Dataset Structure
```
├── train/                  # Folder with training coin images
├── test/                   # Folder with test coin images
├── train.csv               # Contains: Id, Class
├── test.csv                # Contains: Id only
├── sample_submission.csv   # Placeholder for predictions
```
- Images may be `.jpg`, `.jpeg`, `.png`, or `.webp`
- Each class label is a combination of denomination, currency, and country

---

## 📊 Performance Summary

| Model         | Training Time | Validation Accuracy |
|---------------|---------------|---------------------|
| Custom CNN    | 5–6 hrs       | ~10–15%             |
| ResNet50      | 2–3 hrs       | ~25–30%             |
| **EfficientNetB0** | **~30 min**     | **85–90%**           |

> ✅ EfficientNetB0 offered the best tradeoff in terms of training time and performance.

---

## ⚙️ Training Pipeline

- **Model**: EfficientNetB0 (via `timm`)
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam (`lr = 1e-4`)
- **Batch Size**: 32
- **Epochs**: 10
- **Transforms**:
  - Resize to (224, 224)
  - Random horizontal flip
  - Normalization

---

## 🛠️ Features

- Robust image loader handles **corrupted/missing files**
- Custom `CoinDataset` class using `torch.utils.data.Dataset`
- Data augmentation improves generalization by **15–20%**
- Model saving and inference pipeline included
- Outputs predictions in `submission.csv`

---

## 🔁 Inference

To run predictions:
1. Load your trained model (`model.pth`)
2. Run the inference script on `test.csv`
3. Outputs saved in `submission.csv`

---

## 📈 Evaluation

- **Confusion Matrix** for class-wise accuracy
- **Classification Report** with precision, recall, F1-score

---

## 🧪 Improvements

With more time/data:
- Implement **OCR + CNN** fusion for better recognition of text-heavy coins
- Use **Test-Time Augmentation (TTA)** or model ensembling
- Apply **label smoothing**, **learning rate scheduling**
- Experiment with **semi-supervised training** or **larger EfficientNet variants**

---

## 💻 Requirements

- Python 3.8+
- PyTorch
- `timm`, `pandas`, `seaborn`, `matplotlib`, `Pillow`

Install via:

```bash
pip install torch torchvision timm pandas matplotlib seaborn pillow
```
🚀 Getting Started

# Train the model
train_model()

# Evaluate performance
validate()

# Run inference
predict_and_generate_submission()
