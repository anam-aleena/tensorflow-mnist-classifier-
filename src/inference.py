"""
MNIST Classifier — Inference Module
Author: Aleena Anam | github.com/anam-aleena

Load saved model and predict on new images.
Supports: single image, batch images, numpy arrays.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path


MODEL_PATH = "models/mnist_cnn_final.keras"
IMG_SIZE   = 28


def load_model(model_path: str = MODEL_PATH):
    """Load the saved CNN model."""
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Run src/train.py first to train and save the model."
        )
    model = keras.models.load_model(model_path)
    print(f"[MODEL] Loaded from {model_path}")
    return model


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess a raw image for inference.
    Accepts: (28,28), (28,28,1), or (N,28,28) arrays.
    Returns: (N, 28, 28, 1) float32 normalised array.
    """
    image = image.astype("float32")

    # Normalise if not already in [0, 1]
    if image.max() > 1.0:
        image = image / 255.0

    # Ensure shape is (N, 28, 28, 1)
    if image.ndim == 2:
        image = image.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    elif image.ndim == 3 and image.shape[-1] != 1:
        image = image.reshape(image.shape[0], IMG_SIZE, IMG_SIZE, 1)
    elif image.ndim == 3:
        image = image.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    return image


def predict(image: np.ndarray, model=None) -> dict:
    """
    Predict digit from image.

    Args:
        image: numpy array (28x28 pixel values)
        model: loaded Keras model (loads from disk if None)

    Returns:
        dict with prediction, confidence, and all class probabilities
    """
    if model is None:
        model = load_model()

    processed = preprocess_image(image)
    proba     = model.predict(processed, verbose=0)[0]
    predicted = int(np.argmax(proba))
    confidence = float(proba[predicted])

    return {
        "predicted_digit": predicted,
        "confidence":      round(confidence * 100, 2),
        "confidence_pct":  f"{confidence * 100:.2f}%",
        "all_probabilities": {str(i): round(float(p), 4) for i, p in enumerate(proba)},
        "top_3": sorted(
            [{"digit": i, "probability": round(float(p) * 100, 2)}
             for i, p in enumerate(proba)],
            key=lambda x: x["probability"], reverse=True
        )[:3]
    }


def predict_batch(images: list, model=None) -> list:
    """Predict digits for a batch of images."""
    if model is None:
        model = load_model()
    return [predict(img, model) for img in images]


def demo_with_mnist():
    """
    Demo: load 10 random MNIST test samples and predict.
    Shows predictions vs ground truth with confidence scores.
    """
    print("\n[DEMO] Running inference on 10 random MNIST test samples...")
    model = load_model()

    (_, _), (X_test, y_test) = keras.datasets.mnist.load_data()
    indices = np.random.choice(len(X_test), 10, replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle("Inference Demo — MNIST CNN", fontsize=13, fontweight="bold")

    print(f"\n{'Idx':<6} {'True':>5} {'Pred':>5} {'Conf':>8} {'Result':>8}")
    print("-" * 40)

    for i, (ax, idx) in enumerate(zip(axes.flatten(), indices)):
        result = predict(X_test[idx], model)
        pred   = result["predicted_digit"]
        conf   = result["confidence"]
        true   = y_test[idx]
        status = "✓" if pred == true else "✗"
        color  = "#4CAF50" if pred == true else "#F44336"

        print(f"{idx:<6} {true:>5} {pred:>5} {conf:>7.2f}%  {status}")

        ax.imshow(X_test[idx], cmap="gray")
        ax.set_title(f"True:{true}  Pred:{pred}\n{conf:.1f}% {status}",
                     color=color, fontsize=9, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("reports/inference_demo.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n[DEMO] Demo complete. Plot saved → reports/inference_demo.png")


if __name__ == "__main__":
    demo_with_mnist()
