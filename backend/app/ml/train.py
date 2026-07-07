"""End-to-end supervised ML training pipeline for the Hybrid Token Router."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from app.ml.evaluate import cross_validate_model, evaluate_model, prediction_frame
    from app.ml.feature_selection import analyze_features
    from app.ml.model_utils import (
        DOCS_DIR,
        FEATURE_COLUMNS_PATH,
        METADATA_PATH,
        MODEL_PATH,
        PREPROCESSOR_PATH,
        ensure_output_dirs,
        numeric_to_provider,
        provider_to_numeric,
        save_artifact,
        save_json,
    )
    from app.ml.preprocess import build_preprocessor, prepare_training_data
    from app.ml.visualization import generate_eda_report
    from app.services.feature_extractor import extract_features
    from app.services.router import route
except ModuleNotFoundError:  # pragma: no cover
    from evaluate import cross_validate_model, evaluate_model, prediction_frame
    from feature_selection import analyze_features
    from model_utils import (
        DOCS_DIR,
        FEATURE_COLUMNS_PATH,
        METADATA_PATH,
        MODEL_PATH,
        PREPROCESSOR_PATH,
        ensure_output_dirs,
        numeric_to_provider,
        provider_to_numeric,
        save_artifact,
        save_json,
    )
    from preprocess import build_preprocessor, prepare_training_data
    from visualization import generate_eda_report
    from app.services.feature_extractor import extract_features
    from app.services.router import route


RANDOM_STATE = 42


def run_training_pipeline() -> dict[str, Any]:
    """Run preprocessing, EDA, feature selection, training, reports, and persistence."""
    ensure_output_dirs()

    prepared = prepare_training_data(scale_numeric=False)
    feature_analysis = analyze_features(prepared.X, prepared.y)
    generate_eda_report(prepared.dataframe, prepared.X, prepared.y, feature_analysis)

    selected_columns = feature_analysis["retained_features"]
    X = prepared.X[selected_columns].copy()
    y = prepared.y.map(provider_to_numeric)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = build_candidate_models(X_train)
    results: dict[str, Any] = {}
    fitted_models: dict[str, Pipeline] = {}

    for name, model in models.items():
        started = time.perf_counter()
        model.fit(X_train, y_train)
        training_seconds = time.perf_counter() - started
        train_metrics = evaluate_model(model, X_train, y_train.map(numeric_to_provider))
        test_metrics = evaluate_model(model, X_test, y_test.map(numeric_to_provider))
        cv_metrics = cross_validate_model(model, X, prepared.y.loc[X.index])
        results[name] = {
            "training_seconds": round(float(training_seconds), 6),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "cross_validation": cv_metrics,
            "estimator": model.named_steps["classifier"].__class__.__name__,
        }
        fitted_models[name] = model

    best_name = select_best_model(results)
    best_model = fitted_models[best_name]
    best_result = results[best_name]

    save_artifact(MODEL_PATH, best_model)
    save_artifact(PREPROCESSOR_PATH, best_model.named_steps["preprocessor"])
    save_json(FEATURE_COLUMNS_PATH, selected_columns)

    predictions = prediction_frame(best_model, X_test, y_test.map(numeric_to_provider))
    misclassified = build_misclassified_examples(prepared.dataframe, X_test, predictions)
    metadata = build_metadata(
        prepared=prepared,
        feature_analysis=feature_analysis,
        selected_columns=selected_columns,
        results=results,
        best_name=best_name,
        misclassified=misclassified,
    )
    save_json(METADATA_PATH, metadata)
    write_model_report(metadata)
    write_ml_vs_heuristic_report(prepared.dataframe, prepared.X[selected_columns], prepared.y, best_model)

    return metadata


def build_candidate_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    """Build baseline model pipelines."""
    candidates: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    if importlib.util.find_spec("xgboost"):
        from xgboost import XGBClassifier

        candidates["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
    else:
        candidates["Gradient Boosting"] = GradientBoostingClassifier(random_state=RANDOM_STATE)

    pipelines: dict[str, Pipeline] = {}
    for name, estimator in candidates.items():
        scale_numeric = name == "Logistic Regression"
        pipelines[name] = Pipeline([
            ("preprocessor", build_preprocessor(X, scale_numeric=scale_numeric)),
            ("classifier", estimator),
        ])
    return pipelines


def select_best_model(results: dict[str, Any]) -> str:
    """Select the best model using test F1, then ROC AUC, then CV F1."""
    def key(name: str) -> tuple[float, float, float, float]:
        result = results[name]
        return (
            result["test_metrics"]["f1"],
            result["test_metrics"]["roc_auc"],
            result["cross_validation"]["f1"]["mean"],
            result["test_metrics"]["accuracy"],
        )

    return max(results, key=key)


def build_misclassified_examples(
    df: pd.DataFrame,
    X_test: pd.DataFrame,
    predictions: pd.DataFrame,
    limit: int = 15,
) -> list[dict[str, Any]]:
    """Collect representative holdout misclassifications."""
    rows: list[dict[str, Any]] = []
    missed = predictions[predictions["is_correct"] == False].head(limit)  # noqa: E712
    for idx, pred in missed.iterrows():
        source = df.loc[idx]
        rows.append({
            "prompt_id": source.get("prompt_id", str(idx)),
            "prompt": str(source.get("prompt", ""))[:240],
            "actual_label": pred["actual_label"],
            "predicted_label": pred["predicted_label"],
            "confidence": round(float(pred["confidence"]), 6),
            "features": X_test.loc[idx].to_dict(),
        })
    return rows


def build_metadata(
    prepared: Any,
    feature_analysis: dict[str, Any],
    selected_columns: list[str],
    results: dict[str, Any],
    best_name: str,
    misclassified: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build model metadata for reproducibility and reporting."""
    return {
        "pipeline_version": "phase_3_supervised_router_v1",
        "dataset_path": str(Path(prepared.dataframe.attrs.get("path", "")) if prepared.dataframe.attrs.get("path") else ""),
        "dataset_rows": int(len(prepared.dataframe)),
        "class_distribution": prepared.y.value_counts().to_dict(),
        "removed_columns": prepared.removed_columns,
        "selected_feature_columns": selected_columns,
        "removed_model_features": feature_analysis["removed_features"],
        "model_results": results,
        "best_model": best_name,
        "best_model_result": results[best_name],
        "misclassified_examples": misclassified,
        "artifact_paths": {
            "model": str(MODEL_PATH),
            "preprocessor": str(PREPROCESSOR_PATH),
            "feature_columns": str(FEATURE_COLUMNS_PATH),
        },
    }


def write_model_report(metadata: dict[str, Any], report_path: Path = DOCS_DIR / "model_report.md") -> None:
    """Write supervised model training report."""
    best_name = metadata["best_model"]
    best = metadata["best_model_result"]
    lines = [
        "# Supervised Router Model Report",
        "",
        "## Best Model",
        "",
        f"- Best model: `{best_name}`",
        f"- Estimator: `{best['estimator']}`",
        f"- Training accuracy: {best['train_metrics']['accuracy']:.4f}",
        f"- Validation accuracy: {best['test_metrics']['accuracy']:.4f}",
        f"- Validation precision: {best['test_metrics']['precision']:.4f}",
        f"- Validation recall: {best['test_metrics']['recall']:.4f}",
        f"- Validation F1: {best['test_metrics']['f1']:.4f}",
        f"- Validation ROC AUC: {best['test_metrics']['roc_auc']:.4f}",
        f"- Prediction latency: {best['test_metrics']['prediction_latency_ms_per_sample']:.6f} ms/sample",
        "",
        "## Model Comparison",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC AUC | CV F1 Mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in metadata["model_results"].items():
        test = result["test_metrics"]
        cv_f1 = result["cross_validation"]["f1"]["mean"]
        lines.append(
            f"| {name} | {test['accuracy']:.4f} | {test['precision']:.4f} | {test['recall']:.4f} | "
            f"{test['f1']:.4f} | {test['roc_auc']:.4f} | {cv_f1:.4f} |"
        )

    lines.extend([
        "",
        "## Cross Validation",
        "",
    ])
    for metric, values in best["cross_validation"].items():
        lines.append(f"- `{metric}`: mean {values['mean']:.4f}, std {values['std']:.4f}, folds {values['folds']}")

    lines.extend([
        "",
        "## Confusion Matrix",
        "",
        "Rows are actual `[LOCAL, REMOTE]`; columns are predicted `[LOCAL, REMOTE]`.",
        "",
        f"`{best['test_metrics']['confusion_matrix']}`",
        "",
        "## Top 15 Most Influential Features",
        "",
    ])
    feature_importance = _extract_best_feature_importance(metadata)
    if feature_importance:
        for feature, value in feature_importance[:15]:
            lines.append(f"- `{feature}`: {value:.6f}")
    else:
        lines.append("- Feature importance is unavailable for this estimator.")

    lines.extend(["", "## Misclassified Examples", ""])
    if metadata["misclassified_examples"]:
        for item in metadata["misclassified_examples"]:
            lines.append(
                f"- `{item['prompt_id']}`: actual `{item['actual_label']}`, predicted `{item['predicted_label']}`, "
                f"confidence {item['confidence']:.4f}. Prompt: {item['prompt']}"
            )
    else:
        lines.append("- No holdout misclassifications.")

    lines.extend([
        "",
        "## Interpretation",
        "",
        model_interpretation(metadata),
        "",
        "## Limitations",
        "",
        "- The target is generated by the Phase 2 Decision Engine, so the model learns that policy rather than independent human preferences.",
        "- The model intentionally does not use post-inference latency, cost, or quality metrics at prediction time.",
        "- Dataset metadata columns are excluded because live prompts only provide extracted pre-routing features.",
        "",
        "## Recommendations",
        "",
        "- Keep this model behind a feature flag until it is shadow-tested against live router traffic.",
        "- Retrain whenever benchmark composition, local model quality, or remote pricing changes materially.",
        "- Add more LOCAL-positive examples if live traffic shows over-routing to REMOTE.",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extract_best_feature_importance(metadata: dict[str, Any]) -> list[tuple[str, float]]:
    model = None
    try:
        model = pd.read_pickle(MODEL_PATH)
    except Exception:
        try:
            from joblib import load

            model = load(MODEL_PATH)
        except Exception:
            return []
    classifier = model.named_steps.get("classifier")
    names = model.named_steps["preprocessor"].get_feature_names_out()
    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = np.abs(classifier.coef_[0])
    else:
        return []
    return sorted(zip([str(name) for name in names], [float(v) for v in values]), key=lambda item: item[1], reverse=True)


def model_interpretation(metadata: dict[str, Any]) -> str:
    """Generate a concise model quality interpretation."""
    results = metadata["model_results"]
    lr = results.get("Logistic Regression")
    rf = results.get("Random Forest")
    best_name = metadata["best_model"]
    parts: list[str] = []
    if lr and rf:
        lr_f1 = lr["test_metrics"]["f1"]
        rf_f1 = rf["test_metrics"]["f1"]
        if lr_f1 + 0.03 < rf_f1:
            parts.append(
                "Logistic Regression underperforms Random Forest, which suggests the routing boundary is not purely linear. "
                "Interactions among complexity, structural flags, and token pressure appear important."
            )
        elif lr_f1 >= rf_f1:
            parts.append(
                "Logistic Regression is competitive, which suggests the engineered features already encode much of the routing policy linearly."
            )
        else:
            parts.append(
                "Logistic Regression is close to Random Forest, but tree models still capture some nonlinear feature interactions."
            )
    if best_name == "Random Forest":
        parts.append("Random Forest was selected because it produced the strongest holdout ranking while remaining robust on cross validation.")
    elif best_name == "Gradient Boosting":
        parts.append("Gradient Boosting was selected as the XGBoost fallback and performed best among the available baselines.")
    elif best_name == "XGBoost":
        parts.append("XGBoost was selected because it best captured nonlinear routing policy interactions among the available models.")
    else:
        parts.append("The selected linear model is simple, fast, and easier to inspect.")
    return " ".join(parts)


def write_ml_vs_heuristic_report(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    model: Pipeline,
    report_path: Path = DOCS_DIR / "ml_vs_heuristic.md",
) -> None:
    """Compare heuristic router predictions with supervised ML predictions."""
    records: list[dict[str, Any]] = []
    heuristic_latencies: list[float] = []
    ml_latencies: list[float] = []

    for idx, row in df.iterrows():
        prompt = str(row.get("prompt", ""))
        actual = str(row["label"]).upper()

        start = time.perf_counter()
        features = extract_features(prompt)
        heuristic = route(features)
        heuristic_latencies.append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        ml_pred = int(model.predict(X.loc[[idx]])[0])
        ml_probs = model.predict_proba(X.loc[[idx]])[0] if hasattr(model, "predict_proba") else np.array([1 - ml_pred, ml_pred])
        ml_latencies.append((time.perf_counter() - start) * 1000.0)

        records.append({
            "actual": actual,
            "heuristic": str(heuristic["provider"]).upper(),
            "heuristic_confidence": float(heuristic.get("confidence", 0.0)),
            "ml": numeric_to_provider(ml_pred),
            "ml_confidence": float(max(ml_probs)),
            "prompt_id": row.get("prompt_id", str(idx)),
            "prompt": prompt[:200],
        })

    comp = pd.DataFrame(records)
    heuristic_accuracy = float((comp["heuristic"] == comp["actual"]).mean())
    ml_accuracy = float((comp["ml"] == comp["actual"]).mean())
    agreement = float((comp["heuristic"] == comp["ml"]).mean())
    heuristic_misses = comp[comp["heuristic"] != comp["actual"]].head(10)
    ml_misses = comp[comp["ml"] != comp["actual"]].head(10)

    lines = [
        "# ML vs Heuristic Router Comparison",
        "",
        "The Decision Engine labels in `training_dataset.csv` are treated as the offline ground truth for this comparison.",
        "",
        "## Summary Metrics",
        "",
        f"- Heuristic Router accuracy vs Decision Engine: {heuristic_accuracy:.4f}",
        f"- ML Router accuracy vs Decision Engine: {ml_accuracy:.4f}",
        f"- Heuristic and ML agreement: {agreement:.4f}",
        f"- Heuristic average prediction latency: {np.mean(heuristic_latencies):.6f} ms",
        f"- ML average prediction latency: {np.mean(ml_latencies):.6f} ms",
        f"- Heuristic average confidence: {comp['heuristic_confidence'].mean():.4f}",
        f"- ML average confidence: {comp['ml_confidence'].mean():.4f}",
        "",
        "## Strengths",
        "",
        "- Heuristic Router: transparent, deterministic, and independent of training data drift.",
        "- Decision Engine: uses offline benchmark outcomes to create utility-aware labels.",
        "- ML Router: learns the Decision Engine policy from pre-routing features and can capture nonlinear feature interactions.",
        "",
        "## Weaknesses",
        "",
        "- Heuristic Router: hand-tuned thresholds may not match the validated Decision Engine labels.",
        "- Decision Engine: depends on post-inference benchmark signals and cannot run before routing in production.",
        "- ML Router: only generalizes as well as the Phase 2 dataset covers real traffic.",
        "",
        "## Heuristic Misclassifications",
        "",
    ]
    lines.extend(_format_misses(heuristic_misses, "heuristic"))
    lines.extend(["", "## ML Misclassifications", ""])
    lines.extend(_format_misses(ml_misses, "ml"))
    lines.extend([
        "",
        "## Recommendation",
        "",
        "Use the ML router as a shadow-mode replacement candidate. It is trained only on pre-routing features, while the Decision Engine remains the offline labeling and audit mechanism.",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_misses(frame: pd.DataFrame, column: str) -> list[str]:
    if frame.empty:
        return ["- None."]
    return [
        f"- `{row['prompt_id']}`: actual `{row['actual']}`, predicted `{row[column]}`. Prompt: {row['prompt']}"
        for _, row in frame.iterrows()
    ]


if __name__ == "__main__":
    metadata = run_training_pipeline()
    print(f"Best model: {metadata['best_model']}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved preprocessor: {PREPROCESSOR_PATH}")
    print(f"Saved feature columns: {FEATURE_COLUMNS_PATH}")
