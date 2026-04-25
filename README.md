# MNIST Handwritten Digit Classifier — Deep Learning with TensorFlow

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-CNN-D00000?logo=keras&logoColor=white)](https://keras.io)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-brightgreen)](LICENSE)

> **~99.4% test accuracy on MNIST using a production-grade CNN pipeline — with data augmentation, callbacks, REST API, and full evaluation suite.**

---

## What This Project Demonstrates

| Skill | Implementation |
|---|---|
| Deep Learning | CNN with Conv2D, BatchNorm, Dropout, MaxPooling |
| TensorFlow/Keras | Full model build, training, evaluation, export |
| Data Preprocessing | Normalisation, reshape, one-hot encoding |
| Data Augmentation | Random rotation, translation, zoom via Keras layers |
| Training Best Practices | Early stopping, LR decay, model checkpointing |
| Model Evaluation | Accuracy, confusion matrix, per-class analysis |
| Production Deployment | FastAPI REST API with pixel + base64 endpoints |
| Visualisation | Training curves, sample predictions, heatmaps |

---

## Results

| Metric | Score |
|---|---|
| **Test Accuracy** | **~99.4%** |
| Test Loss | ~0.021 |
| Training Epochs | 12–15 (early stopping) |
| Parameters | ~420K |
| Training Time | ~3–5 mins (CPU) / ~1 min (GPU) |

---

## CNN Architecture

```
Input: (28 × 28 × 1)
    ↓
Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(64) → BatchNorm → Dropout(0.25)
    ↓
Flatten → Dense(128) → BatchNorm → Dropout(0.5)
    ↓
Dense(10, softmax) → Predicted Digit (0–9)
```

**Design Decisions:**
- **Batch Normalisation** — stabilises training, faster convergence, acts as regularisation
- **Progressive filters (32→64)** — captures low-level edges first, then complex patterns
- **Dropout (0.25 + 0.5)** — prevents overfitting on the 60K training set
- **Cosine Decay LR** — smooth learning rate reduction for stable final convergence
- **Data Augmentation** — rotation, translation, zoom for better generalisation

---

## Project Structure

```
tensorflow-mnist-classifier/
│
├── src/
│   ├── train.py          # Full training pipeline — data → model → evaluation → export
│   └── inference.py      # Load model + predict on new images
│
├── api/
│   └── app.py            # FastAPI REST API (pixel array + base64 image endpoints)
│
├── notebooks/
│   └── 01_Training_Demo.ipynb  # EDA, training, results analysis
│
├── models/               # Saved model weights (after training)
├── reports/              # Generated plots and metrics
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone & install
```bash
git clone https://github.com/anam-aleena/tensorflow-mnist-classifier.git
cd tensorflow-mnist-classifier
pip install -r requirements.txt
```

### 2. Train the model
```bash
python src/train.py
# Trains CNN, evaluates on test set, saves model + reports
```

### 3. Run inference demo
```bash
python src/inference.py
# Predicts 10 random test samples with confidence scores
```

### 4. Predict in Python
```python
from src.inference import load_model, predict
import numpy as np

model = load_model()

# Random 28x28 image (replace with your image)
image = np.random.rand(28, 28)
result = predict(image, model)

print(result["predicted_digit"])   # e.g. 7
print(result["confidence_pct"])    # e.g. "99.83%"
print(result["top_3"])             # Top 3 predictions with probabilities
```

### 5. Launch REST API
```bash
uvicorn api.app:app --reload
# → http://localhost:8000/docs   (interactive Swagger UI)
```

**API Request Example:**
```json
POST /predict/pixels
{
  "pixels": [0.0, 0.1, 0.9, ...]   // 784 float values (28x28 flattened)
}
```

**API Response:**
```json
{
  "predicted_digit": 7,
  "confidence": 99.83,
  "confidence_pct": "99.83%",
  "top_3": [
    {"digit": 7, "probability": 99.83},
    {"digit": 1, "probability": 0.12},
    {"digit": 2, "probability": 0.04}
  ]
}
```

---

## Generated Reports

After training, the following are saved to `/reports/`:

| File | Description |
|---|---|
| `training_history.png` | Accuracy & loss curves over epochs |
| `confusion_matrix.png` | 10×10 confusion matrix heatmap |
| `per_class_accuracy.png` | Accuracy per digit class |
| `sample_predictions.png` | 20 random predictions with confidence |
| `inference_demo.png` | Live inference demo output |
| `metrics.json` | Final test accuracy and loss |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | TensorFlow 2.15, Keras |
| Data Processing | NumPy, Scikit-learn |
| Visualisation | Matplotlib, Seaborn |
| API | FastAPI, Pydantic |
| Image Processing | Pillow |
| Environment | Jupyter Notebook |

---

## Author

**Aleena Anam** — AI/ML Engineer & Data Scientist  
📧 anamaleena0@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/aleena-anam-2056a4368) | [GitHub](https://github.com/anam-aleena)

---

## License

MIT License — free to use, modify, and distribute.
