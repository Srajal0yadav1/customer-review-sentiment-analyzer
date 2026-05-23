import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)


SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]


def print_metrics(y_true, y_pred) -> None:
    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {acc:.4f} ({acc * 100:.2f}%)\n")
    print("Per-Class Metrics:")
    print("-" * 60)
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=SENTIMENT_LABELS,
            zero_division=0,
        )
    )


def plot_confusion_matrix(y_true, y_pred, save_path: str = None):
    cm = confusion_matrix(y_true, y_pred, labels=SENTIMENT_LABELS)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=SENTIMENT_LABELS,
        yticklabels=SENTIMENT_LABELS,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("Actual Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_class_distribution(y, save_path: str = None):
    series = pd.Series(y).value_counts().reindex(SENTIMENT_LABELS, fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#d73027", "#fee090", "#1a9641"]
    bars = ax.bar(series.index, series.values, color=colors, edgecolor="white", width=0.5)

    for bar, val in zip(bars, series.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("Sentiment Class", fontsize=12)
    ax.set_ylabel("Number of Reviews", fontsize=12)
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_top_features(pipeline, n: int = 20, save_path: str = None):
    try:
        vectorizer = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["clf"]
        feature_names = vectorizer.get_feature_names_out()
        classes = clf.classes_
        coef = clf.coef_

        fig, axes = plt.subplots(1, len(classes), figsize=(18, 6))

        class_colors = {"Negative": "#d73027", "Neutral": "#fee090", "Positive": "#1a9641"}

        for ax, class_name, coef_row in zip(axes, classes, coef):
            top_idx = np.argsort(coef_row)[-n:][::-1]
            top_features = feature_names[top_idx]
            top_coefs = coef_row[top_idx]

            ax.barh(
                top_features[::-1],
                top_coefs[::-1],
                color=class_colors.get(class_name, "steelblue"),
                alpha=0.85,
            )
            ax.set_title(f"Top {n} Features: {class_name}", fontweight="bold")
            ax.set_xlabel("Coefficient Weight")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.suptitle("Most Influential Words per Sentiment Class", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    except AttributeError:
        print("Feature importance visualization is only available for Logistic Regression.")
        return None


def compare_algorithms(results_dict: dict, save_path: str = None):
    names = list(results_dict.keys())
    scores = list(results_dict.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, scores, color=["#4472C4", "#ED7D31", "#A9D18E"], edgecolor="white", width=0.4)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Weighted F1-Score", fontsize=12)
    ax.set_title("Algorithm Comparison (Weighted F1-Score)", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig