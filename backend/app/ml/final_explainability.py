"""Generate final feature importance and explainability reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from app.ml.model_utils import (
    DOCS_DIR,
    FEATURE_COLUMNS_PATH,
    MODEL_PATH,
    PLOTS_DIR,
    load_artifact,
    load_json,
    provider_to_numeric,
)
from app.ml.predict import predict_prompt
from app.ml.preprocess import prepare_training_data


IMPORTANCE_REPORT = DOCS_DIR / "final_feature_importance.md"
EXPLAINABILITY_REPORT = DOCS_DIR / "model_explainability.md"


def generate_explainability_reports() -> dict[str, str]:
    """Generate final feature importance docs and plots from saved artifacts."""
    model = load_artifact(MODEL_PATH)
    feature_columns = load_json(FEATURE_COLUMNS_PATH)
    prepared = prepare_training_data(scale_numeric=False)
    X = prepared.X[feature_columns]
    y = prepared.y.map(provider_to_numeric)
    importances = _feature_importances(model, X, y)
    plots = _write_plots(importances)
    _write_feature_importance_report(importances, plots)
    _write_model_explainability_report(importances)
    return {
        "feature_importance": str(IMPORTANCE_REPORT),
        "model_explainability": str(EXPLAINABILITY_REPORT),
        **plots,
    }


def _feature_importances(model: Any, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    names = [str(name) for name in preprocessor.get_feature_names_out()]

    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
        method = "native feature importance"
    elif hasattr(classifier, "coef_"):
        values = abs(classifier.coef_[0])
        method = "native absolute coefficient"
    else:
        result = permutation_importance(model, X, y, n_repeats=20, random_state=42, scoring="f1")
        names = X.columns.tolist()
        values = result.importances_mean
        method = "permutation importance"

    frame = pd.DataFrame({"feature": names, "importance": values})
    grouped = frame.groupby("feature", as_index=False)["importance"].sum()
    grouped["method"] = method
    total = grouped["importance"].sum()
    grouped["normalized_importance"] = grouped["importance"] / total if total else grouped["importance"]
    return grouped.sort_values("importance", ascending=False).reset_index(drop=True)


def _write_plots(importances: pd.DataFrame) -> dict[str, str]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    top = importances.head(20).sort_values("importance")

    bar_path = PLOTS_DIR / "final_feature_importance_bar.png"
    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["importance"])
    plt.title("Top 20 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=140, bbox_inches="tight")
    plt.close()

    sorted_path = PLOTS_DIR / "final_feature_importance_sorted.png"
    plt.figure(figsize=(10, 7))
    plt.plot(range(1, len(importances) + 1), importances["importance"], marker="o")
    plt.title("Sorted Feature Importance")
    plt.xlabel("Feature Rank")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(sorted_path, dpi=140, bbox_inches="tight")
    plt.close()

    return {"bar_chart": str(bar_path), "sorted_chart": str(sorted_path)}


def _write_feature_importance_report(importances: pd.DataFrame, plots: dict[str, str]) -> None:
    lines = [
        "# Final Feature Importance Report",
        "",
        f"- Importance method: {importances['method'].iloc[0]}",
        f"- Bar chart: `{Path(plots['bar_chart']).as_posix()}`",
        f"- Sorted chart: `{Path(plots['sorted_chart']).as_posix()}`",
        "",
        "## Top 20 Features",
        "",
        "| Rank | Feature | Importance | Normalized Share | Interpretation |",
        "|---:|---|---:|---:|---|",
    ]
    for idx, row in importances.head(20).iterrows():
        lines.append(
            f"| {idx + 1} | `{row['feature']}` | {row['importance']:.6f} | "
            f"{row['normalized_importance']:.4f} | {_feature_interpretation(row['feature'])} |"
        )
    lines.extend(["", "## Examples", ""])
    for feature in importances.head(10)["feature"]:
        lines.append(f"- `{feature}`: {_feature_example(feature)}")
    IMPORTANCE_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_model_explainability_report(importances: pd.DataFrame) -> None:
    examples = [
        "Write a Go service that validates signed payment webhooks and retries failed database writes.",
        "Translate this short delivery update into Spanish using a friendly tone.",
        "Analyze the legal and ethical risks of deploying facial recognition in public schools.",
    ]
    lines = [
        "# Model Explainability",
        "",
        "## How Routing Decisions Are Made",
        "",
        "The live ML router extracts pre-routing prompt features, aligns them to the saved training feature list, applies the saved preprocessing pipeline inside `router_model.pkl`, and returns the provider with the highest predicted probability.",
        "",
        "The model never receives post-inference latency, cost, or quality columns at prediction time.",
        "",
        "## Probability and Confidence",
        "",
        "- `prediction_probability` is the probability assigned to the selected provider.",
        "- `prediction_confidence` is the larger of LOCAL and REMOTE probabilities.",
        "- High confidence means the model is far from its learned decision boundary; it does not guarantee the provider response will be better for every individual prompt.",
        "",
        "## Example Predictions",
        "",
    ]
    for prompt in examples:
        pred = predict_prompt(prompt)
        lines.append(
            f"- Prompt: {prompt} Selected `{pred['predicted_provider']}` with confidence "
            f"{pred['confidence']:.4f}. Top signal: {_top_signal(pred)}"
        )
    lines.extend([
        "",
        "## Feature Ranking Summary",
        "",
    ])
    for idx, row in importances.head(10).iterrows():
        lines.append(f"- {idx + 1}. `{row['feature']}` ({row['importance']:.6f}): {_feature_interpretation(row['feature'])}")
    lines.extend([
        "",
        "## Limitations",
        "",
        "- The model learns from Decision Engine labels, so it approximates that policy rather than independent human preference.",
        "- It cannot observe live provider failures until after routing; provider-level fallback still protects local runtime failures.",
        "- Feature importance explains the fitted model globally, while individual predictions can depend on feature interactions.",
    ])
    EXPLAINABILITY_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _top_signal(prediction: dict[str, Any]) -> str:
    contributions = prediction.get("feature_contributions") or []
    if not contributions:
        return "feature contribution unavailable"
    top = contributions[0]
    return top.get("feature", "unknown")


def _feature_interpretation(feature: str) -> str:
    if "prompt_length" in feature:
        return "Longer prompts often imply more context, constraints, or scope, shifting routing utility."
    if "context_complexity" in feature:
        return "Dense terminology and context load change whether remote quality is worth the trade-off."
    if "task_complexity" in feature:
        return "Implementation, synthesis, planning, and multi-deliverable wording influences provider choice."
    if "reasoning_depth" in feature or "reasoning_score" in feature:
        return "Deeper analytical or multi-step reasoning can favor stronger remote responses."
    if "technical_complexity" in feature:
        return "Specialized technical domains alter expected response quality requirements."
    if "constraint_complexity" in feature:
        return "Explicit formatting, safety, or consistency constraints affect routing confidence."
    if "contains_code" in feature:
        return "Code-like prompts follow different routing patterns than general questions."
    if "contains_math" in feature or "contains_numbers" in feature:
        return "Numerical or mathematical content can indicate precision-sensitive work."
    return "This pre-routing signal contributes to the learned Decision Engine approximation."


def _feature_example(feature: str) -> str:
    if "prompt_length" in feature:
        return "A detailed cloud migration request carries more operational context than a one-line definition."
    if "context_complexity" in feature:
        return "A healthcare consent translation with clinical terminology has more context pressure than a casual reminder."
    if "task_complexity" in feature:
        return "A rollout plan with milestones, risks, and owners is more demanding than a simple checklist."
    if "technical_complexity" in feature:
        return "Distributed consensus, cryptography, and Kubernetes prompts trigger specialized technical signals."
    if "constraint_complexity" in feature:
        return "Requests with exact output formats or compliance constraints raise the cost of a weak answer."
    return "Prompts where this feature is active change the learned probability assigned to LOCAL or REMOTE."


if __name__ == "__main__":
    print(generate_explainability_reports())
