"""
MNIST Handwritten Digit Classifier — TensorFlow/Keras Pipeline
Author: Aleena Anam | github.com/anam-aleena

End-to-end deep learning pipeline:
Data loading → Preprocessing → CNN training → Evaluation → Model export
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ──────────────────────────────────────────────────────────────────

RANDOM_SEED   = 42
EPOCHS        = 15
BATCH_SIZE    = 128
LEARNING_RATE = 0.001
NUM_CLASSES   = 10
IMG_SIZE      = 28
MODEL_DIR     = "models"
REPORT_DIR    = "reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─── 1. DATA LOADING & PREPROCESSING ────────────────────────────────────────

def load_and_preprocess():
    """
    Load MNIST dataset. Normalise pixel values to [0, 1].
    Reshape for CNN input: (N, 28, 28, 1).
    One-hot encode labels.
    """
    print("[DATA] Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Normalise pixel values
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32")  / 255.0

    # Reshape: add channel dimension for CNN
    X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_test  = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # One-hot encode labels
    y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_cat  = keras.utils.to_categorical(y_test,  NUM_CLASSES)

    print(f"[DATA] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[DATA] Label range: {y_train.min()} – {y_train.max()}")
    return X_train, X_test, y_train, y_test, y_train_cat, y_test_cat


# ─── 2. MODEL ARCHITECTURE ───────────────────────────────────────────────────

def build_cnn_model():
    """
    Convolutional Neural Network for MNIST classification.

    Architecture:
        Input (28x28x1)
        → Conv2D(32) + BatchNorm + MaxPool + Dropout
        → Conv2D(64) + BatchNorm + MaxPool + Dropout
        → Conv2D(64) + BatchNorm
        → Flatten
        → Dense(128) + BatchNorm + Dropout
        → Dense(10, softmax)

    Design choices:
    - Batch normalisation: stabilises training, faster convergence
    - Dropout: prevents overfitting on small image dataset
    - Progressive filter increase (32→64): captures increasingly complex features
    """
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        # Fully connected
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        # Output
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="mnist_cnn")

    return model


# ─── 3. TRAINING ─────────────────────────────────────────────────────────────

def train_model(model, X_train, y_train_cat, X_test, y_test_cat):
    """
    Compile and train model with:
    - Adam optimiser with cosine decay learning rate
    - Early stopping to prevent overfitting
    - Model checkpoint to save best weights
    - Data augmentation for robustness
    """
    # Learning rate schedule
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=EPOCHS * (len(X_train) // BATCH_SIZE)
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"\n[MODEL] Architecture Summary:")
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=f"{MODEL_DIR}/best_model.keras",
            monitor="val_accuracy", save_best_only=True, verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2,
            min_lr=1e-6, verbose=1
        )
    ]

    # Data augmentation
    data_augmentation = keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomZoom(0.05),
    ])

    # Augment only training data
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat))
    train_dataset = (train_dataset
                     .shuffle(10000)
                     .batch(BATCH_SIZE)
                     .map(lambda x, y: (data_augmentation(x, training=True), y),
                          num_parallel_calls=tf.data.AUTOTUNE)
                     .prefetch(tf.data.AUTOTUNE))

    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test_cat))
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print(f"\n[TRAIN] Training for up to {EPOCHS} epochs...")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )

    return history


# ─── 4. EVALUATION ───────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, y_test_cat):
    """
    Full evaluation: accuracy, per-class metrics,
    confusion matrix, training curves.
    """
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    print(f"\n[EVAL] Test Loss    : {loss:.4f}")
    print(f"[EVAL] Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\n[EVAL] Per-class Report:")
    print(classification_report(y_test, y_pred,
          target_names=[str(i) for i in range(10)]))

    metrics = {
        "test_accuracy": round(float(accuracy), 4),
        "test_loss":     round(float(loss), 4),
    }
    with open(f"{REPORT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return y_pred, y_pred_proba, metrics


# ─── 5. VISUALISATIONS ───────────────────────────────────────────────────────

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("MNIST CNN — Training History", fontsize=14, fontweight="bold")

    ax1.plot(history.history["accuracy"],     label="Train", color="#2196F3", lw=2)
    ax1.plot(history.history["val_accuracy"], label="Val",   color="#4CAF50", lw=2)
    ax1.set_title("Accuracy", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax1.set_ylim([0.95, 1.0])

    ax2.plot(history.history["loss"],     label="Train", color="#2196F3", lw=2)
    ax2.plot(history.history["val_loss"], label="Val",   color="#4CAF50", lw=2)
    ax2.set_title("Loss", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/training_history.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Training history saved.")


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10), ax=ax)
    ax.set_title("Confusion Matrix — MNIST CNN", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Digit"); ax.set_ylabel("Actual Digit")
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Confusion matrix saved.")


def plot_sample_predictions(X_test, y_test, y_pred, y_pred_proba, n=20):
    fig, axes = plt.subplots(4, 5, figsize=(14, 11))
    fig.suptitle("Sample Predictions — MNIST CNN", fontsize=14, fontweight="bold")
    indices = np.random.choice(len(X_test), n, replace=False)

    for i, (ax, idx) in enumerate(zip(axes.flatten(), indices)):
        img = X_test[idx].reshape(IMG_SIZE, IMG_SIZE)
        pred, true = y_pred[idx], y_test[idx]
        conf = y_pred_proba[idx][pred] * 100
        color = "#4CAF50" if pred == true else "#F44336"
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Pred: {pred}  True: {true}\n{conf:.1f}%",
                     color=color, fontsize=9, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/sample_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Sample predictions saved.")


def plot_per_class_accuracy(y_test, y_pred):
    per_class = []
    for digit in range(10):
        mask = y_test == digit
        acc  = (y_pred[mask] == y_test[mask]).mean()
        per_class.append(acc * 100)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(10), per_class,
                  color=["#4CAF50" if a >= 99 else "#2196F3" if a >= 98 else "#FF9800"
                         for a in per_class],
                  edgecolor="white")
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit Class", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Per-Class Accuracy — MNIST CNN", fontsize=13, fontweight="bold")
    ax.set_ylim([95, 100.5])
    for bar, acc in zip(bars, per_class):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{acc:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/per_class_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Per-class accuracy saved.")


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  MNIST HANDWRITTEN DIGIT CLASSIFIER — CNN Pipeline")
    print("  Author: Aleena Anam | github.com/anam-aleena")
    print("=" * 62)

    # Load data
    X_train, X_test, y_train, y_test, y_train_cat, y_test_cat = load_and_preprocess()

    # Build model
    model = build_cnn_model()

    # Train
    history = train_model(model, X_train, y_train_cat, X_test, y_test_cat)

    # Evaluate
    y_pred, y_pred_proba, metrics = evaluate_model(model, X_test, y_test, y_test_cat)

    # Visualise
    print("\n[PLOTS] Generating visualisations...")
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred)
    plot_sample_predictions(X_test, y_test, y_pred, y_pred_proba)
    plot_per_class_accuracy(y_test, y_pred)

    # Save model
    model.save(f"{MODEL_DIR}/mnist_cnn_final.keras")
    print(f"[SAVE] Final model saved → {MODEL_DIR}/mnist_cnn_final.keras")

    print("\n" + "=" * 62)
    print(f"  FINAL RESULT")
    print(f"  Test Accuracy : {metrics['test_accuracy']*100:.2f}%")
    print(f"  Test Loss     : {metrics['test_loss']:.4f}")
    print(f"  Reports saved → /reports/")
    print("=" * 62)
