"""
utils.py — Shared utilities for the WikiArt artist classification project.

Only includes functions reused across multiple notebooks (data loading,
training curves, evaluation, model comparison, Grad-CAM, history I/O).
EDA and one-off preprocessing live in their respective notebooks.

Usage:
    from utils import *
"""

import os
import json
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)


# ============================================================
# Reproducibility
# ============================================================

def set_seeds(seed: int = 42):
    """Set Python, NumPy, and TensorFlow random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ============================================================
# Data Pipeline
# ============================================================

def load_datasets(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    img_size: tuple = (224, 224),
    batch_size: int = 32,
    seed: int = 42,
    shuffle_buffer: int = 1000,
):
    """
    Load train/val/test as tf.data.Dataset with categorical labels,
    .cache(), .shuffle() (train only), and .prefetch().
    Returns (train_ds, val_ds, test_ds, class_names).
    """
    common = dict(image_size=img_size, batch_size=batch_size,
                  label_mode="categorical", seed=seed)

    train_ds = keras.utils.image_dataset_from_directory(train_dir, **common)
    val_ds = keras.utils.image_dataset_from_directory(val_dir, **common)
    test_ds = keras.utils.image_dataset_from_directory(test_dir, **common)

    class_names = train_ds.class_names
    print(f"Classes ({len(class_names)}): {class_names}")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(shuffle_buffer, seed=seed).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)

    return train_ds, val_ds, test_ds, class_names


# ============================================================
# Training Curves
# ============================================================

def plot_learning_curves(history, title: str = ""):
    """
    Plot loss, accuracy, and macro F1 from a Keras History object.
    Accepts a single History, a list of Histories (concatenated),
    or a raw dict (from load_history).
    """
    if isinstance(history, (list, tuple)):
        combined = {}
        for h in history:
            for k, v in h.history.items():
                combined.setdefault(k, []).extend(v)
    elif isinstance(history, dict):
        combined = history
    else:
        combined = history.history

    epochs = range(1, len(combined["loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(epochs, combined["loss"], label="Train")
    axes[0].plot(epochs, combined["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, combined["accuracy"], label="Train")
    axes[1].plot(epochs, combined["val_accuracy"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    f1_key, val_f1_key = "f1_macro", "val_f1_macro"
    if f1_key in combined:
        axes[2].plot(epochs, combined[f1_key], label="Train")
        axes[2].plot(epochs, combined[val_f1_key], label="Val")
        axes[2].set_title("F1 Score (macro)")
        axes[2].set_xlabel("Epoch")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "F1 not tracked", ha="center", va="center",
                     transform=axes[2].transAxes)

    if title:
        plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    best_epoch = int(np.argmin(combined["val_loss"]))
    print(f"Best epoch (lowest val_loss): {best_epoch + 1}")
    print(f"  Val Loss: {combined['val_loss'][best_epoch]:.4f}")
    print(f"  Val Acc:  {combined['val_accuracy'][best_epoch]:.4f}")
    if val_f1_key in combined:
        print(f"  Val F1:   {combined[val_f1_key][best_epoch]:.4f}")


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, test_ds, class_names: list[str], model_name: str = "Model"):
    """
    Full test evaluation: model.evaluate, classification report,
    confusion matrix, and per-class F1 chart.
    Returns a metrics dict for use with compare_models().
    """
    results = model.evaluate(test_ds, verbose=0)
    metric_names = model.metrics_names

    print(f"\n{'='*50}")
    print(f"  {model_name} — Test Evaluation")
    print(f"{'='*50}")
    for name, val in zip(metric_names, results):
        print(f"  {name:>12s}: {val:.4f}")

    y_true, y_pred_probs = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred_probs.extend(preds)

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    plot_confusion_matrix(y_true, y_pred, class_names,
                          title=f"Confusion Matrix — {model_name}")
    plot_per_class_f1(y_true, y_pred, class_names,
                      title=f"Per-Class F1 — {model_name}")

    return {
        "model_name": model_name,
        "test_loss": results[metric_names.index("loss")],
        "test_accuracy": results[metric_names.index("accuracy")],
        "test_f1_macro": f1_score(y_true, y_pred, average="macro"),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_probs": y_pred_probs,
    }


def plot_confusion_matrix(
    y_true, y_pred, class_names: list[str],
    title: str = "Confusion Matrix",
    normalize: bool = False,
    figsize: tuple = (14, 12),
):
    """Confusion matrix heatmap. Set normalize=True for row-percentages."""
    cm = confusion_matrix(y_true, y_pred)
    fmt = "d"
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_per_class_f1(y_true, y_pred, class_names: list[str], title: str = ""):
    """Horizontal bar chart of per-class F1 scores, sorted ascending."""
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True)
    f1_scores = {cls: report[cls]["f1-score"] for cls in class_names}
    sorted_items = sorted(f1_scores.items(), key=lambda x: x[1])
    names = [item[0].replace("_", " ") for item in sorted_items]
    scores = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(8, max(6, len(class_names) * 0.35)))
    colors = ["#e74c3c" if s < 0.5 else "#f39c12" if s < 0.7 else "#2ecc71"
              for s in scores]
    ax.barh(names, scores, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 Score")
    ax.set_title(title or "Per-Class F1 Score")
    ax.axvline(np.mean(scores), color="black", ls="--", lw=1,
               label=f"Macro avg: {np.mean(scores):.3f}")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Model Comparison
# ============================================================

def compare_models(metrics_list: list[dict]):
    """
    Print comparison table + bar chart from a list of metrics dicts
    (each returned by evaluate_model).
    """
    print(f"\n{'='*65}")
    print(f"  Model Comparison")
    print(f"{'='*65}")
    header = f"{'Model':<30s} {'Accuracy':>10s} {'F1 (macro)':>12s} {'Loss':>10s}"
    print(header)
    print("-" * 65)
    for m in metrics_list:
        print(f"{m['model_name']:<30s} "
              f"{m['test_accuracy']:>10.4f} "
              f"{m['test_f1_macro']:>12.4f} "
              f"{m['test_loss']:>10.4f}")
    print("-" * 65)

    names = [m["model_name"] for m in metrics_list]
    accs = [m["test_accuracy"] for m in metrics_list]
    f1s = [m["test_f1_macro"] for m in metrics_list]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(names)), 5))
    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, f1s, width, label="F1 (macro)", color="#55A868")

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.show()


# ============================================================
# Grad-CAM
# ============================================================

def make_gradcam_heatmap(model, img_array, last_conv_layer_name: str, pred_index=None):
    """
    Compute Grad-CAM heatmap. img_array shape: (1, H, W, 3), already preprocessed.
    """
    grad_model = keras.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def plot_gradcam(
    model,
    img_path: str,
    last_conv_layer_name: str,
    img_size: tuple = (224, 224),
    preprocess_fn=None,
    class_names: list[str] | None = None,
    alpha: float = 0.4,
):
    """
    End-to-end Grad-CAM: load image, compute heatmap, show overlay.

    preprocess_fn: e.g. tf.keras.applications.resnet50.preprocess_input.
                   If None, divides by 255.
    """
    img = tf.keras.utils.load_img(img_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    display_img = img_array.astype("uint8")

    img_input = np.expand_dims(img_array, axis=0)
    if preprocess_fn is not None:
        img_input = preprocess_fn(img_input.copy())
    else:
        img_input = img_input / 255.0

    preds = model.predict(img_input, verbose=0)
    pred_idx = np.argmax(preds[0])
    pred_conf = preds[0][pred_idx]
    pred_label = class_names[pred_idx] if class_names else str(pred_idx)

    heatmap = make_gradcam_heatmap(model, img_input, last_conv_layer_name,
                                   pred_index=pred_idx)
    heatmap_resized = np.uint8(255 * heatmap)
    heatmap_resized = tf.image.resize(
        heatmap_resized[..., np.newaxis], img_size
    ).numpy().squeeze()

    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = (jet_colors[heatmap_resized.astype(int)] * 255).astype("uint8")
    superimposed = (jet_heatmap * alpha + display_img * (1 - alpha)).astype("uint8")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(display_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(superimposed)
    axes[2].set_title(f"Pred: {pred_label.replace('_', ' ')} ({pred_conf:.2%})")
    axes[2].axis("off")

    plt.suptitle("Grad-CAM Visualization", fontsize=13)
    plt.tight_layout()
    plt.show()


# ============================================================
# History Serialization
# ============================================================

def save_history(history, filepath: str):
    """Save Keras History (or list of Histories) to JSON."""
    if isinstance(history, (list, tuple)):
        combined = {}
        for h in history:
            for k, v in h.history.items():
                combined.setdefault(k, []).extend(v)
    elif isinstance(history, dict):
        combined = history
    else:
        combined = history.history

    serializable = {k: [float(x) for x in v] for k, v in combined.items()}
    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"History saved to {filepath}")


def load_history(filepath: str) -> dict:
    """Load a previously saved history dict from JSON."""
    with open(filepath, "r") as f:
        return json.load(f)