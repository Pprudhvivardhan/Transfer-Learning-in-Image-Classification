# 🧠 Image Classification with Transfer Learning (VGG16 vs. Custom CNN)

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Run in Colab](https://img.shields.io/badge/Notebook-Colab-yellow.svg)](https://colab.research.google.com/drive/1or_iiRKke1SBUwxUokgN96OZVXXMnlrT#scrollTo=81445993)

This project explores the power of **transfer learning** for multi-class image classification. We compare a **custom CNN model** trained from scratch against a **VGG16-based transfer learning approach**, including both **frozen** and **fine-tuned** versions. The dataset comprises over 14,000 images of natural and urban scenes across six categories.

---

## 📑 Table of Contents
- [📖 Background and Motivation](#-background-and-motivation)
- [🎯 Project Objectives](#-project-objectives)
- [📊 Dataset Description](#-dataset-description)
- [⚙️ Methodology Overview](#️-methodology-overview)
- [📈 Experimental Results](#-experimental-results)
- [🔍 Model Comparison](#-model-comparison)
- [💡 Limitations & Ethical Considerations](#-limitations--ethical-considerations)
- [✅ Conclusion](#-conclusion)
- [🛠️ How to Run](#️-how-to-run)
- [📂 Project Structure](#-project-structure)
- [👤 Author](#-author)
- [📚 References](#-references)

---

## 📖 Background and Motivation

Image classification is a critical computer vision task with widespread applications. Deep CNNs have revolutionized this domain, but training models from scratch can be time-consuming and require large datasets. Transfer learning addresses this by reusing knowledge from pre-trained models.

This project investigates:
- How a basic CNN compares with a pre-trained VGG16 model
- How freezing and fine-tuning VGG16 layers affects performance

---

## 🎯 Project Objectives

- Build a baseline CNN for image classification.
- Implement transfer learning using VGG16.
- Compare model performance on accuracy, loss curves, and confusion matrix.
- Analyze training dynamics, overfitting, and generalization.
- Fine-tune VGG16 to maximize performance.

---

## 📊 Dataset Description

**Natural Scenes Dataset**
- **Images:** ~14,000
- **Resolution:** 150x150 pixels
- **Categories:** Buildings, Forests, Glaciers, Mountains, Seas, Streets
- **Split:**
  - Training set
  - Test set
  - Prediction set (optional/real-world testing)

---

## ⚙️ Methodology Overview

### 1. Custom CNN Model
- Architecture: Convolution + MaxPooling + Dense layers
- Training Accuracy: ~86.09%
- Validation Accuracy: ~85.31%
- Test Accuracy: ~17.13% → Severe **overfitting** observed.

### 2. Transfer Learning with VGG16
- Used pre-trained VGG16 from ImageNet
- Initial phase: All layers **frozen**, only classifier trained
- Second phase: Selective **fine-tuning** (unfroze deeper layers)

---

## 📈 Experimental Results

| Model         | Train Acc | Val Acc | Test Acc | Val Loss | Overfitting |
|---------------|-----------|---------|----------|----------|-------------|
| Base CNN      | 86.09%    | 85.31%  | 17.13%   | ↑ After 12th epoch | High |
| VGG16 Frozen  | 86.23%    | 85.77%  | -        | ~0.3776  | Moderate |
| VGG16 Tuned   | 99.15%    | 92.04%  | 92.00%   | ~0.3283  | Low |

**Key Observations:**
- Validation loss diverges early in CNN → overfitting
- VGG16 generalizes much better
- Fine-tuning VGG16 yields highest performance

[CHART: Training vs Validation Loss vs Training vs Validation Accuracy]
<img width="1978" height="777" alt="image" src="https://github.com/user-attachments/assets/3b39373f-45de-43f3-bab9-b089ae64db5d" />

---

## 🔍 Model Comparison

- **Base CNN**:
  - Trained from scratch
  - Suffers from poor test generalization
  - Poor class separation (confusion matrix)

- **VGG16 Frozen**:
  - Utilizes powerful ImageNet feature extractors
  - Stable accuracy and validation loss

- **VGG16 Fine-Tuned**:
  - Best performance overall
  - Closer training/validation accuracy
  - Smooth learning curves

---

## 💡 Limitations & Ethical Considerations

- **Domain Mismatch:** Pre-trained models may not always suit specific datasets.
- **Computational Demand:** Fine-tuning VGG16 requires significant GPU resources.
- **Interpretability:** Deep models lack transparency.
- **Bias & Fairness:** Pre-trained models may carry over dataset bias.
- **Data Ethics:** Ensure ethical use of data, respect privacy, and obtain proper consent.

---

## ✅ Conclusion

- Transfer learning **vastly outperforms** training CNNs from scratch on this dataset.
- Selective fine-tuning of VGG16 boosts performance to **92% accuracy**.
- Proper layer freezing/unfreezing strategies and early stopping are critical.
- Transfer learning is ideal for **resource-efficient**, high-performing image classification.

---

## 🛠️ How to Run

1. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/transfer-learning-image-classification.git
   cd transfer-learning-image-classification
