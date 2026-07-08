"""Supervised ML training pipeline for the Traditional ML Router (Phase 6).

Implements:
1. Dataset split partitioning (excluding locked evaluation set)
2. Hyperparameter optimization using Optuna (Bayesian Optimization) with fallback
3. SMOTE for class imbalance handling
4. Probability calibration
5. Decision threshold optimization
6. Experiment tracking and metrics reporting
"""

from __future__ import annotations

import logging
import time
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.calibration import FrozenEstimator
    HAS_FROZEN = True
except ImportError:
    HAS_FROZEN = False

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

try:
    import optuna
    HAS_OPTUNA = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    HAS_OPTUNA = False

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
from app.ml.locked_eval import get_locked_evaluation_splits, TRAIN_SPLIT_PATH
from app.ml.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

RANDOM_STATE = 42


def run_training_pipeline() -> dict[str, Any]:
    """Run Phase 6 supervised ML router training pipeline."""
    ensure_output_dirs()
    logger.info("Starting Phase 6 ML training pipeline...")

    # 1. Initialize locked evaluation and training splits
    train_df, locked_df = get_locked_evaluation_splits()
    
    # Load prepared training data using the training subset
    prepared = prepare_training_data(dataset_path=TRAIN_SPLIT_PATH, scale_numeric=False)
    feature_analysis = analyze_features(prepared.X, prepared.y)
    
    # Generate EDA visualization reports
    try:
        generate_eda_report(prepared.dataframe, prepared.X, prepared.y, feature_analysis)
    except Exception as e:
        logger.warning("EDA visualization failed: %s", e)

    selected_columns = feature_analysis["retained_features"]
    X = prepared.X[selected_columns].copy()
    y = prepared.y.map(provider_to_numeric)

    # Split train/validation (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 2. Build and preprocess features
    # Scale only numeric features for linear models/SVM
    preprocessor = build_preprocessor(X, scale_numeric=True)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    # Convert sparse matrices to dense if needed
    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()
        X_val_proc = X_val_proc.toarray()

    # 3. Handle class imbalance using SMOTE
    if HAS_SMOTE:
        logger.info("Applying SMOTE oversampling to balance training classes...")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)
        logger.info("Balanced class shapes: X=%s, y=%s", X_train_res.shape, y_train_res.shape)
    else:
        logger.info("SMOTE unavailable. Defaulting to class weighted optimization.")
        X_train_res, y_train_res = X_train_proc, y_train

    # 4. Search and tune candidate classifiers
    candidate_names = ["Logistic Regression", "Random Forest", "Linear SVM", "Gradient Boosting"]
    if HAS_XGB:
        candidate_names.append("XGBoost")
    if HAS_LGB:
        candidate_names.append("LightGBM")
    if HAS_CAT:
        candidate_names.append("CatBoost")

    results: dict[str, Any] = {}
    fitted_pipelines: dict[str, Pipeline] = {}

    for name in candidate_names:
        logger.info("Optimizing candidate model: %s", name)
        best_clf, best_params = optimize_classifier(name, X_train_res, y_train_res, X_val_proc, y_val)
        
        # Calibration of probabilities
        logger.info("Calibrating probabilities for: %s", name)
        if HAS_FROZEN:
            calibrated_clf = CalibratedClassifierCV(estimator=FrozenEstimator(best_clf), method="sigmoid")
        else:
            calibrated_clf = CalibratedClassifierCV(estimator=best_clf, method="sigmoid", cv="prefit")
        calibrated_clf.fit(X_val_proc, y_val)

        # Threshold optimization on the validation set
        val_probs = calibrated_clf.predict_proba(X_val_proc)[:, 1]
        best_threshold = optimize_threshold(y_val, val_probs)
        logger.info("%s - Optimized decision threshold: %.3f", name, best_threshold)

        # Evaluate model on training and validation sets at optimized threshold
        train_metrics = evaluate_at_threshold(calibrated_clf, X_train_proc, y_train, best_threshold)
        val_metrics = evaluate_at_threshold(calibrated_clf, X_val_proc, y_val, best_threshold)

        # Cross Validation (using Stratified 5-Fold)
        cv_scores = perform_cross_validation(best_clf, X_train_proc, y_train)

        # Calculate final weighted score
        # F1: 40%, ROC AUC: 30%, Precision: 15%, Recall: 15%
        weighted_score = (
            0.40 * val_metrics["f1"]
            + 0.30 * val_metrics["roc_auc"]
            + 0.15 * val_metrics["precision"]
            + 0.15 * val_metrics["recall"]
        )

        results[name] = {
            "hyperparameters": best_params,
            "train_metrics": train_metrics,
            "test_metrics": val_metrics,
            "cross_validation": cv_scores,
            "optimized_threshold": round(best_threshold, 3),
            "weighted_ranking_score": round(weighted_score, 4),
            "estimator": best_clf.__class__.__name__,
        }

        # Build pipeline artifact wrapper (preprocessor + calibrated classifier + threshold information)
        fitted_pipelines[name] = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", calibrated_clf),
        ])
        # Save threshold directly inside the pipeline object for use during inference
        fitted_pipelines[name].threshold = best_threshold

        logger.info(
            "%s - Val Accuracy: %.4f, F1: %.4f, ROC AUC: %.4f, Weighted Score: %.4f",
            name, val_metrics["accuracy"], val_metrics["f1"], val_metrics["roc_auc"], weighted_score
        )

    # 5. Select and persist the best model
    best_name = max(results, key=lambda k: results[k]["weighted_ranking_score"])
    best_pipeline = fitted_pipelines[best_name]
    best_result = results[best_name]

    logger.info("Selecting best model overall: %s (score: %.4f)", best_name, best_result["weighted_ranking_score"])

    save_artifact(MODEL_PATH, best_pipeline)
    save_artifact(PREPROCESSOR_PATH, best_pipeline.named_steps["preprocessor"])
    save_json(FEATURE_COLUMNS_PATH, selected_columns)

    # 6. Log the experiment for tracking
    tracker = ExperimentTracker()
    tracker.log_experiment(
        experiment_name="phase_6_model_search",
        dataset_version="large_dataset_balanced_v1",
        feature_version="V3_expanded_features",
        embedding_model="None (Traditional ML)",
        classifier=best_name,
        hyperparameters=best_result["hyperparameters"],
        metrics=best_result["test_metrics"],
    )

    # 7. Write reports
    metadata = {
        "pipeline_version": "phase_6_supervised_router_v1",
        "dataset_path": str(TRAIN_SPLIT_PATH.resolve()),
        "dataset_rows": int(len(train_df)),
        "class_distribution": y.value_counts().to_dict(),
        "removed_columns": prepared.removed_columns,
        "selected_feature_columns": selected_columns,
        "removed_model_features": feature_analysis["removed_features"],
        "model_results": results,
        "best_model": best_name,
        "best_model_result": best_result,
        "misclassified_examples": [],
        "artifact_paths": {
            "model": str(MODEL_PATH),
            "preprocessor": str(PREPROCESSOR_PATH),
            "feature_columns": str(FEATURE_COLUMNS_PATH),
        },
    }
    save_json(METADATA_PATH, metadata)
    write_model_report(metadata)

    return metadata


def optimize_classifier(
    name: str,
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: np.ndarray,
    y_val: pd.Series,
) -> tuple[Any, dict[str, Any]]:
    """Performs Optuna hyperparameter optimization with RandomizedSearch fallback."""
    if HAS_OPTUNA:
        try:
            return optimize_with_optuna(name, X_train, y_train, X_val, y_val)
        except Exception as exc:
            logger.warning("Optuna optimization failed for %s (%s). Falling back to RandomizedSearch.", name, exc)
            
    return optimize_with_random_search(name, X_train, y_train)


def optimize_with_optuna(
    name: str,
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: np.ndarray,
    y_val: pd.Series,
) -> tuple[Any, dict[str, Any]]:
    """Bayesian Hyperparameter Search using Optuna."""
    def objective(trial: optuna.Trial) -> float:
        if name == "Logistic Regression":
            C = trial.suggest_float("C", 1e-4, 1e2, log=True)
            clf = LogisticRegression(C=C, max_iter=3000, random_state=RANDOM_STATE)
        elif name == "Random Forest":
            n_est = trial.suggest_int("n_estimators", 50, 400)
            depth = trial.suggest_int("max_depth", 3, 15)
            clf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=RANDOM_STATE, n_jobs=-1)
        elif name == "Linear SVM":
            C = trial.suggest_float("C", 1e-4, 1e2, log=True)
            clf = SVC(C=C, kernel="linear", probability=True, random_state=RANDOM_STATE)
        elif name == "Gradient Boosting":
            n_est = trial.suggest_int("n_estimators", 50, 300)
            lr = trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True)
            clf = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr, random_state=RANDOM_STATE)
        elif name == "XGBoost" and HAS_XGB:
            n_est = trial.suggest_int("n_estimators", 50, 300)
            lr = trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True)
            depth = trial.suggest_int("max_depth", 2, 8)
            clf = XGBClassifier(n_estimators=n_est, learning_rate=lr, max_depth=depth, random_state=RANDOM_STATE, n_jobs=-1)
        elif name == "LightGBM" and HAS_LGB:
            n_est = trial.suggest_int("n_estimators", 50, 300)
            lr = trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True)
            clf = LGBMClassifier(n_estimators=n_est, learning_rate=lr, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        elif name == "CatBoost" and HAS_CAT:
            n_est = trial.suggest_int("iterations", 50, 300)
            lr = trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True)
            clf = CatBoostClassifier(iterations=n_est, learning_rate=lr, random_state=RANDOM_STATE, verbose=0)
        else:
            clf = LogisticRegression(random_state=RANDOM_STATE)

        clf.fit(X_train, y_train)
        
        # Optimize primarily for F1 score
        from sklearn.metrics import f1_score
        preds = clf.predict(X_val)
        return float(f1_score(y_val, preds, zero_division=0))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)
    best_params = study.best_params

    # Refit best estimator
    if name == "Logistic Regression":
        clf = LogisticRegression(**best_params, max_iter=3000, random_state=RANDOM_STATE)
    elif name == "Random Forest":
        clf = RandomForestClassifier(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
    elif name == "Linear SVM":
        clf = SVC(**best_params, kernel="linear", probability=True, random_state=RANDOM_STATE)
    elif name == "Gradient Boosting":
        clf = GradientBoostingClassifier(**best_params, random_state=RANDOM_STATE)
    elif name == "XGBoost" and HAS_XGB:
        clf = XGBClassifier(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
    elif name == "LightGBM" and HAS_LGB:
        clf = LGBMClassifier(**best_params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    elif name == "CatBoost" and HAS_CAT:
        clf = CatBoostClassifier(**best_params, random_state=RANDOM_STATE, verbose=0)
    else:
        clf = LogisticRegression(random_state=RANDOM_STATE)

    clf.fit(X_train, y_train)
    return clf, best_params


def optimize_with_random_search(name: str, X_train: np.ndarray, y_train: pd.Series) -> tuple[Any, dict[str, Any]]:
    """Fallback hyperparameter optimization using RandomizedSearchCV."""
    if name == "Logistic Regression":
        clf = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
        param_dist = {"C": np.logspace(-3, 2, 20)}
    elif name == "Random Forest":
        clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        param_dist = {"n_estimators": [100, 200, 300], "max_depth": [5, 10, 15, None]}
    elif name == "Linear SVM":
        clf = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)
        param_dist = {"C": np.logspace(-3, 2, 20)}
    elif name == "Gradient Boosting":
        clf = GradientBoostingClassifier(random_state=RANDOM_STATE)
        param_dist = {"n_estimators": [100, 200], "learning_rate": [0.01, 0.05, 0.1]}
    elif name == "XGBoost" and HAS_XGB:
        clf = XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        param_dist = {"n_estimators": [100, 200], "learning_rate": [0.01, 0.05, 0.1]}
    elif name == "LightGBM" and HAS_LGB:
        clf = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        param_dist = {"n_estimators": [100, 200], "learning_rate": [0.01, 0.05, 0.1]}
    elif name == "CatBoost" and HAS_CAT:
        clf = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0)
        param_dist = {"iterations": [100, 200], "learning_rate": [0.01, 0.05, 0.1]}
    else:
        clf = LogisticRegression(random_state=RANDOM_STATE)
        param_dist = {}

    search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=8, cv=3, scoring="f1", random_state=RANDOM_STATE, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def optimize_threshold(y_true: pd.Series, probs: np.ndarray) -> float:
    """Find the probability threshold that maximizes balanced accuracy."""
    from sklearn.metrics import balanced_accuracy_score
    best_thresh = 0.5
    best_score = 0.0
    for thresh in np.arange(0.01, 0.99, 0.01):
        preds = (probs >= thresh).astype(int)
        score = balanced_accuracy_score(y_true, preds)
        if score > best_score:
            best_score = score
            best_thresh = thresh
    return float(best_thresh)


def evaluate_at_threshold(clf: Any, X: np.ndarray, y: pd.Series, threshold: float) -> dict[str, Any]:
    """Evaluate model performance using custom threshold."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    probs = clf.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else 0.5,
        "confusion_matrix": confusion_matrix(y, preds, labels=[0, 1]).tolist(),
        "prediction_latency_ms_per_sample": 0.05, # Placeholder
    }


def perform_cross_validation(clf: Any, X: np.ndarray, y: pd.Series) -> dict[str, Any]:
    """Run stratified 5-fold cross validation."""
    from sklearn.metrics import f1_score, roc_auc_score
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    f1s = []
    aucs = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
        
        clf.fit(X_tr, y_tr)
        if HAS_FROZEN:
            fold_clf = CalibratedClassifierCV(estimator=FrozenEstimator(clf), method="sigmoid")
        else:
            fold_clf = CalibratedClassifierCV(estimator=clf, method="sigmoid", cv="prefit")
        fold_clf.fit(X_va, y_va)
        
        probs = fold_clf.predict_proba(X_va)[:, 1]
        preds = fold_clf.predict(X_va)
        f1s.append(f1_score(y_va, preds, zero_division=0))
        aucs.append(roc_auc_score(y_va, probs) if len(np.unique(y_va)) > 1 else 0.5)
        
    return {
        "f1": {"mean": float(np.mean(f1s)), "std": float(np.std(f1s)), "folds": 5},
        "roc_auc": {"mean": float(np.mean(aucs)), "std": float(np.std(aucs)), "folds": 5},
    }


def write_model_report(metadata: dict[str, Any], report_path: Path = DOCS_DIR / "model_report.md") -> None:
    """Write model report markdown file."""
    best_name = metadata["best_model"]
    best = metadata["best_model_result"]
    lines = [
        "# Supervised Router Model Report (Phase 6)",
        "",
        "## Best Model",
        "",
        f"- Best model: `{best_name}`",
        f"- Estimator: `{best['estimator']}`",
        f"- Validation accuracy: {best['test_metrics']['accuracy']:.4f}",
        f"- Validation precision: {best['test_metrics']['precision']:.4f}",
        f"- Validation recall: {best['test_metrics']['recall']:.4f}",
        f"- Validation F1: {best['test_metrics']['f1']:.4f}",
        f"- Validation ROC AUC: {best['test_metrics']['roc_auc']:.4f}",
        f"- Optimized threshold: `{best['optimized_threshold']}`",
        "",
        "## Model Comparison",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC AUC | CV F1 Mean | Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in metadata["model_results"].items():
        test = result["test_metrics"]
        cv_f1 = result["cross_validation"]["f1"]["mean"]
        lines.append(
            f"| {name} | {test['accuracy']:.4f} | {test['precision']:.4f} | {test['recall']:.4f} | "
            f"{test['f1']:.4f} | {test['roc_auc']:.4f} | {cv_f1:.4f} | {result['optimized_threshold']} |"
        )
        
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Successfully generated model report at: %s", report_path)


if __name__ == "__main__":
    metadata = run_training_pipeline()
    print(f"Best model: {metadata['best_model']}")
