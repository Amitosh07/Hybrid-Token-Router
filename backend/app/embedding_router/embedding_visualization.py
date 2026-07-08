"""Visualization module for embedding projections and model performance comparison.

Generates t-SNE, PCA plots of embeddings, and model comparison bar charts.
"""

from __future__ import annotations

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from app.ml.model_utils import PLOTS_DIR
from app.embedding_router.embedding_dataset import load_dataset_and_extract_embeddings

logger = logging.getLogger(__name__)

# Style settings
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "figure.facecolor": "#121212",
    "axes.facecolor": "#1e1e1e",
    "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#b0b0b0",
    "ytick.color": "#b0b0b0",
    "grid.color": "#333333",
    "axes.edgecolor": "#444444"
})


def generate_embedding_projections() -> None:
    """Generates PCA and t-SNE plots of prompt embeddings, colored by ground-truth labels."""
    logger.info("Loading data for embedding projections...")
    df, X_emb, _, _ = load_dataset_and_extract_embeddings()
    y = df["label"].values
    
    # 1. PCA Projection
    logger.info("Computing 2D PCA projection...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_emb)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1], hue=y,
        palette={"LOCAL": "#00C853", "REMOTE": "#2979FF"},
        alpha=0.6, s=40, edgecolor="none"
    )
    plt.title("PCA Projection of Prompt Embeddings", fontsize=16, pad=15)
    plt.xlabel(f"PC1 (Variance Explained: {pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
    plt.ylabel(f"PC2 (Variance Explained: {pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
    plt.legend(title="Ground Truth Route", facecolor="#1e1e1e", edgecolor="#444444")
    plt.tight_layout()
    
    pca_path = PLOTS_DIR / "embedding_pca.png"
    plt.savefig(pca_path, dpi=150, facecolor="#121212")
    plt.close()
    logger.info("Saved PCA projection plot to %s", pca_path)
    
    # 2. t-SNE Projection (using a subset of 1000 prompts for speed and clarity)
    logger.info("Computing 2D t-SNE projection (subset of 1000 prompts for clarity)...")
    np.random.seed(42)
    indices = np.random.choice(len(X_emb), min(1000, len(X_emb)), replace=False)
    X_subset = X_emb[indices]
    y_subset = y[indices]
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_subset)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_subset,
        palette={"LOCAL": "#00C853", "REMOTE": "#2979FF"},
        alpha=0.7, s=50, edgecolor="none"
    )
    plt.title("t-SNE Projection of Prompt Embeddings", fontsize=16, pad=15)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(title="Ground Truth Route", facecolor="#1e1e1e", edgecolor="#444444")
    plt.tight_layout()
    
    tsne_path = PLOTS_DIR / "embedding_tsne.png"
    plt.savefig(tsne_path, dpi=150, facecolor="#121212")
    plt.close()
    logger.info("Saved t-SNE projection plot to %s", tsne_path)


def generate_router_comparison_charts(comparison_df: pd.DataFrame) -> None:
    """Generates bar charts comparing model metrics across routing systems.
    
    Args:
        comparison_df: DataFrame containing columns: ['Router', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
    """
    logger.info("Generating router comparison charts...")
    
    # Melt comparison dataframe for plotting
    melted_df = pd.melt(
        comparison_df, 
        id_vars=["Router"], 
        value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC AUC"],
        var_name="Metric", 
        value_name="Value"
    )
    
    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=melted_df, x="Metric", y="Value", hue="Router",
        palette="viridis", edgecolor="none"
    )
    plt.title("Routing Performance Metrics Comparison", fontsize=18, pad=20)
    plt.ylim(0, 1.05)
    plt.ylabel("Score", fontsize=14)
    plt.xlabel("Metric", fontsize=14)
    plt.legend(title="Routing Method", facecolor="#1e1e1e", edgecolor="#444444", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    
    metrics_chart_path = PLOTS_DIR / "router_metrics_comparison.png"
    plt.savefig(metrics_chart_path, dpi=150, facecolor="#121212")
    plt.close()
    logger.info("Saved metrics comparison chart to %s", metrics_chart_path)
    
    # Latency comparison chart
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=comparison_df, x="Router", y="Prediction Latency (ms)",
        palette="rocket", edgecolor="none"
    )
    plt.title("Average Router Decision Latency", fontsize=18, pad=20)
    plt.ylabel("Latency (ms / sample)", fontsize=14)
    plt.xlabel("Routing Method", fontsize=14)
    plt.tight_layout()
    
    latency_chart_path = PLOTS_DIR / "router_latency_comparison.png"
    plt.savefig(latency_chart_path, dpi=150, facecolor="#121212")
    plt.close()
    logger.info("Saved latency comparison chart to %s", latency_chart_path)


if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    generate_embedding_projections()
