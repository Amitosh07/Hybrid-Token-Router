"""EDA and visualization utilities for Phase 3 router ML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from app.ml.model_utils import DOCS_DIR, PLOTS_DIR, POST_INFERENCE_COLUMNS
except ModuleNotFoundError:  # pragma: no cover
    from model_utils import DOCS_DIR, PLOTS_DIR, POST_INFERENCE_COLUMNS


def generate_eda_report(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    feature_analysis: dict[str, Any],
    report_path: Path | str = DOCS_DIR / "eda_report.md",
    plots_dir: Path | str = PLOTS_DIR,
) -> dict[str, str]:
    """Create EDA plots and a Markdown report."""
    plots = create_visualizations(df, X, y, feature_analysis, Path(plots_dir))
    write_eda_report(df, X, y, feature_analysis, plots, Path(report_path))
    return plots


def create_visualizations(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    feature_analysis: dict[str, Any],
    plots_dir: Path,
) -> dict[str, str]:
    """Generate EDA and feature-importance figures."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    plots: dict[str, str] = {}

    plt.figure(figsize=(6, 4))
    sns.countplot(x=y, order=sorted(y.unique()))
    plt.title("Class Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plots["class_distribution"] = _save_plot(plots_dir / "class_distribution.png")

    if numeric_cols:
        hist_cols = numeric_cols[:12]
        axes = X[hist_cols].hist(figsize=(14, 10), bins=20)
        for ax in axes.ravel():
            ax.set_ylabel("Count")
        plt.suptitle("Feature Distributions", y=1.02)
        plots["histograms"] = _save_plot(plots_dir / "feature_histograms.png")

        plt.figure(figsize=(12, 9))
        corr = X[numeric_cols].corr().fillna(0.0)
        sns.heatmap(corr, cmap="vlag", center=0, linewidths=0.25)
        plt.title("Correlation Matrix")
        plots["correlation_heatmap"] = _save_plot(plots_dir / "correlation_heatmap.png")

        box_cols = numeric_cols[:10]
        melted = pd.concat([X[box_cols], y.rename("label")], axis=1).melt(id_vars="label")
        plt.figure(figsize=(14, 7))
        sns.boxplot(data=melted, x="variable", y="value", hue="label")
        plt.xticks(rotation=45, ha="right")
        plt.title("Feature Boxplots by Class")
        plots["boxplots"] = _save_plot(plots_dir / "feature_boxplots.png")

    importance = feature_analysis.get("feature_importance", {})
    if importance:
        top = pd.Series(importance).head(15).sort_values()
        plt.figure(figsize=(9, 6))
        top.plot(kind="barh")
        plt.title("Top 15 Feature Importances")
        plt.xlabel("Importance")
        plots["feature_importance"] = _save_plot(plots_dir / "feature_importance.png")

    pair_cols = [col for col in list(feature_analysis.get("feature_importance", {}).keys())[:4] if col in X.columns]
    if len(pair_cols) >= 2:
        pair_df = pd.concat([X[pair_cols], y.rename("label")], axis=1)
        grid = sns.pairplot(pair_df, hue="label", diag_kind="hist", corner=True)
        grid.fig.suptitle("Pairwise Relationships", y=1.02)
        pair_path = plots_dir / "pairwise_relationships.png"
        grid.savefig(pair_path, bbox_inches="tight", dpi=140)
        plt.close(grid.fig)
        plots["pairwise_relationships"] = str(pair_path)

    return plots


def write_eda_report(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    feature_analysis: dict[str, Any],
    plots: dict[str, str],
    report_path: Path,
) -> None:
    """Write EDA report Markdown."""
    numeric_X = X.select_dtypes(include=["number", "bool"])
    missing = df.isna().sum()
    outlier_lines = []
    for col in numeric_X.columns:
        q1 = numeric_X[col].quantile(0.25)
        q3 = numeric_X[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            count = 0
        else:
            count = int(((numeric_X[col] < q1 - 1.5 * iqr) | (numeric_X[col] > q3 + 1.5 * iqr)).sum())
        outlier_lines.append((col, count))

    lines = [
        "# Exploratory Data Analysis Report",
        "",
        "## Dataset Overview",
        "",
        f"- Rows: {len(df)}",
        f"- Columns: {len(df.columns)}",
        f"- Trainable pre-routing features: {len(X.columns)}",
        f"- Post-inference columns reserved for offline evaluation: {', '.join(sorted(POST_INFERENCE_COLUMNS & set(df.columns)))}",
        "",
        "## Class Distribution",
        "",
    ]
    lines.extend([f"- `{label}`: {count}" for label, count in y.value_counts().items()])
    lines.extend(["", "## Feature Statistics", "", _markdown_table(numeric_X.describe().round(4)), ""])
    lines.extend(["## Missing Value Analysis", ""])
    nonzero_missing = missing[missing > 0]
    if nonzero_missing.empty:
        lines.append("- No missing values detected.")
    else:
        lines.extend([f"- `{col}`: {int(count)}" for col, count in nonzero_missing.items()])

    lines.extend(["", "## Outlier Analysis", ""])
    for col, count in outlier_lines:
        lines.append(f"- `{col}`: {count} IQR outliers")

    lines.extend(["", "## Correlation Matrix", ""])
    lines.append("- See `backend/app/ml/plots/correlation_heatmap.png`.")
    if feature_analysis.get("redundant_pairs"):
        for pair in feature_analysis["redundant_pairs"]:
            lines.append(f"- `{pair['feature_a']}` vs `{pair['feature_b']}`: {pair['correlation']}")
    else:
        lines.append("- No feature pairs exceeded the high-correlation removal threshold.")

    lines.extend(["", "## Feature Importance", ""])
    for feature, score in list(feature_analysis.get("feature_importance", {}).items())[:15]:
        lines.append(f"- `{feature}`: {score:.6f}")

    lines.extend(["", "## Feature Distributions and Pairwise Relationships", ""])
    for name, path in plots.items():
        lines.append(f"- `{name}`: `{Path(path).as_posix()}`")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_plot(path: Path) -> str:
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=140)
    plt.close()
    return str(path)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame as Markdown without optional dependencies."""
    table = frame.reset_index().rename(columns={"index": "stat"})
    headers = [str(col) for col in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table.iterrows():
        values = [str(row[col]) for col in table.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
