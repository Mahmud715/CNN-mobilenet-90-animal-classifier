# Animal90 CNN Classifier

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.16%2B-orange)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

**A lightweight, single‑file TensorFlow implementation to classify 90 animal species** from the [Kaggle Animal Image Dataset – 90 Different Animals](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals).

The model achieves:

* **97%** training accuracy
* **≥85%** test accuracy on a held-out 10% split

---

## 📋 Table of Contents

* [Introduction](#introduction)
* [Features](#features)
* [Dataset](#dataset)
* [Dependencies](#dependencies)
* [Installation](#installation)
* [Usage](#usage)
* [Architecture](#architecture)
* [Results](#results)
* [Configuration](#configuration)
* [License](#license)

---

## 📝 Introduction

This project demonstrates how to leverage a **MobileNetV2** backbone (pretrained on ImageNet) and a simple classification head to build a high‑accuracy animal species classifier with minimal code. It’s designed for rapid experimentation on consumer hardware (CPU or GPU).

## 🌟 Features

* **Transfer Learning**: Frozen MobileNetV2 base for fast convergence
* **Data Augmentation**: Random flips, rotations, and zooms to improve generalisability
* **Regularisation**: Dropout and Early Stopping to prevent over‑fitting
* **Dynamic Splitting**: On‑the‑fly train/validation/test set creation with `tf.data`
* **Batch Inference**: Utility to classify entire folders of new images
* **Single‑File**: All logic in `animal90_cnn.py`, ready to run

## 📂 Dataset

* **Source**: [Kaggle – Animal Image Dataset (90 Classes)](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)
* **Structure**: One subfolder per class (e.g. `datasets/animals90/lion/`, `datasets/animals90/tiger/`, …)

## 🛠 Dependencies

* Python 3.10 or higher
* TensorFlow 2.16 or higher

Install with:

```bash
pip install tensorflow
```

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/Mahmud715/animal90-cnn-classifier.git
cd animal90-cnn-classifier

# (Optional) Create a virtual environment
env="venv"
python -m venv "$env"
source "$env/bin/activate"  # macOS/Linux
"$env\Scripts\activate"    # Windows

# Install dependencies
pip install --upgrade tensorflow
```

## ⚙️ Configuration

By default, the script looks for your dataset in the environment variable `ANIMALS90_DIR`. If unset, it falls back to:

```bash
# Example for Windows PowerShell
env:ANIMALS90_DIR = 'C:\datasets\animals90'
```

You can also hard‑code the path in `animal90_cnn.py` at the top:

```python
DATASET_DIR = Path(r'C:\path\to\animals90')
```

## ▶️ Usage

**Train & evaluate**:

```bash
python animal90_cnn.py
```

**Batch inference**:

```bash
python animal90_cnn.py --predict /path/to/new_images
```

Results are printed to the console, and the model is saved as `animal90_cnn_savedmodel.keras`.

## 🏗 Architecture

1. **Input**: 224×224 RGB images
2. **Preprocessing**: `tf.keras.applications.mobilenet_v2.preprocess_input`
3. **Base**: MobileNetV2 (frozen)
4. **Head**:

   * Global Average Pooling
   * Dropout (rate=0.3)
   * Dense softmax (90 classes)

## 📈 Results

| Metric              | Value |
| ------------------- | ----- |
| Training Accuracy   | 97%   |
| Validation Accuracy | \~88% |
| Test Accuracy       | ≥85%  |

> These results were obtained on an NVIDIA RTX‑series GPU in \~10 minutes. CPU‑only runs may take longer.

## ⚙️ Configuration Options

Inside `animal90_cnn.py`, you can adjust:

* `IMG_SIZE` (default: 224×224)
* `BATCH_SIZE` (default: 32)
* `EPOCHS` (default: 30)
* `VAL_SPLIT` / `TEST_SPLIT`

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
