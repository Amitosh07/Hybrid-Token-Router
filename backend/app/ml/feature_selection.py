"""Feature selection analysis and reporting for router ML training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

try:
    from app.ml.model_utils import DOCS_DIR, provider_to_numeric
except ModuleNotFoundError:  # pragma: no cover
    from model_utils import DOCS_DIR, provider_to_numeric


def analyze_features(
    X: pd.DataFrame,
    y: pd.Series,
    report_path: Path | str = DOCS_DIR / "feature_importance_report.md",
) -> dict[str, Any]:
    """Analyze constant, low variance, correlated, redundant, and important features."""
    numeric_X = X.select_dtypes(include=["number", "bool"]).copy()
    y_num = y.map(provider_to_numeric)

    variances = numeric_X.var(numeric_only=True).sort_values()
    constant_features = variances[variances <= 0.0].index.tolist()
    low_variance_features = variances[(variances > 0.0) & (variances < 0.001)].index.tolist()

    corr = numeric_X.corr().fillna(0.0)
    redundant_pairs: list[dict[str, Any]] = []
    redundant_features: set[str] = set()
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1:]:
            value = float(corr.loc[left, right])
            if abs(value) >= 0.95:
                redundant_pairs.append({"feature_a": left, "feature_b": right, "correlation": round(value, 4)})
                redundant_features.add(right)

    retained_features = [
        col for col in X.columns
        if col not in set(constant_features) and col not in redundant_features
    ]
    removed_features = sorted(set(constant_features) | redundant_features)

    importance = _compute_feature_importance(numeric_X, y_num)
    mutual_info = _compute_mutual_info(numeric_X, y_num)

    analysis = {
        "retained_features": retained_features,
        "removed_features": removed_features,
        "constant_features": constant_features,
        "low_variance_features": low_variance_features,
        "redundant_features": sorted(redundant_features),
        "redundant_pairs": redundant_pairs,
        "variance": variances.round(6).to_dict(),
        "correlation_with_target": _target_correlations(numeric_X, y_num),
        "feature_importance": importance,
        "mutual_information": mutual_info,
    }
    write_feature_report(analysis, Path(report_path))
    return analysis


def _compute_feature_importance(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    if X.empty or y.nunique() < 2:
        return {}
    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    model.fit(X.fillna(X.median(numeric_only=True)), y)
    return {
        feature: round(float(score), 6)
        for feature, score in sorted(
            zip(X.columns, model.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
    }


def _compute_mutual_info(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    if X.empty or y.nunique() < 2:
        return {}
    filled = X.fillna(X.median(numeric_only=True))
    scores = mutual_info_classif(filled, y, discrete_features="auto", random_state=42)
    return {
        feature: round(float(score), 6)
        for feature, score in sorted(zip(X.columns, scores), key=lambda item: item[1], reverse=True)
    }


def _target_correlations(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    values: dict[str, float] = {}
    for col in X.columns:
        if X[col].nunique(dropna=True) <= 1:
            values[col] = 0.0
            continue
        corr = np.corrcoef(X[col].fillna(X[col].median()), y)[0, 1]
        values[col] = round(float(0.0 if np.isnan(corr) else corr), 6)
    return dict(sorted(values.items(), key=lambda item: abs(item[1]), reverse=True))


def write_feature_report(analysis: dict[str, Any], report_path: Path) -> None:
    """Write the feature selection report as Markdown."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Feature Importance and Selection Report",
        "",
        "## Leakage Policy",
        "",
        "Training excludes all post-inference latency, cost, and quality metrics. These fields are reserved for offline evaluation only.",
        "",
        "## Retained Features",
        "",
    ]
    lines.extend([f"- `{feature}`" for feature in analysis["retained_features"]])
    lines.extend(["", "## Removed Features", ""])
    if analysis["removed_features"]:
        for feature in analysis["removed_features"]:
            reason = "constant" if feature in analysis["constant_features"] else "redundant with a highly correlated feature"
            lines.append(f"- `{feature}`: removed because it is {reason}.")
    else:
        lines.append("- No model input features were removed by automatic feature selection.")

    lines.extend(["", "## Low Variance Features", ""])
    if analysis["low_variance_features"]:
        lines.extend([f"- `{feature}`" for feature in analysis["low_variance_features"]])
    else:
        lines.append("- None detected above the constant-feature threshold.")

    lines.extend(["", "## Highly Correlated Pairs", ""])
    if analysis["redundant_pairs"]:
        for pair in analysis["redundant_pairs"]:
            lines.append(f"- `{pair['feature_a']}` and `{pair['feature_b']}`: correlation {pair['correlation']}")
    else:
        lines.append("- No feature pairs exceeded the 0.95 absolute-correlation threshold.")

    lines.extend(["", "## Top Feature Importance", ""])
    for feature, score in list(analysis["feature_importance"].items())[:15]:
        lines.append(f"- `{feature}`: {score:.6f}")

    lines.extend(["", "## Target Correlation", ""])
    for feature, score in list(analysis["correlation_with_target"].items())[:15]:
        lines.append(f"- `{feature}`: {score:.6f}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
