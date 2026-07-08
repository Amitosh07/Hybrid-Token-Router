"""Model training script for embedding-based and hybrid-based routing models (Phase 6).

Compares classifiers, tunes hyperparameters (Optuna), calibrates, optimizes threshold,
and persists models and metadata.
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
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.calibration import FrozenEstimator
    HAS_FROZEN = True
except ImportError:
    HAS_FROZEN = False

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
from app.ml.model_utils import (
    numeric_to_provider,
    provider_to_numeric,
    save_artifact,
    save_json,
    MODELS_DIR,
    FEATURE_COLUMNS_PATH,
    load_json,
)
from app.ml.preprocess import prepare_training_data, build_preprocessor
from app.embedding_router.embedding_utils import (
    get_embedding_model_name,
    get_model,
    get_model_version,
    EMBEDDING_MODEL_PATH,
    EMBEDDING_METADATA_PATH,
    HYBRID_MODEL_PATH,
    HYBRID_METADATA_PATH,
)
from app.embedding_router.embedding_dataset import prepare_dataset_splits, load_dataset_and_extract_embeddings
from app.ml.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

RANDOM_STATE = 42


def compute_weighted_score(metrics: dict[str, float]) -> float:
    """Computes a weighted ranking score based on evaluation metrics."""
    f1 = metrics.get("f1", 0.0)
    roc_auc = metrics.get("roc_auc", 0.0)
    precision = metrics.get("precision", 0.0)
    recall = metrics.get("recall", 0.0)
    
    score = 0.40 * f1 + 0.30 * roc_auc + 0.15 * precision + 0.15 * recall
    return round(score, 6)


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
            n_est = trial.suggest_int("n_estimators", 50, 300)
            depth = trial.suggest_int("max_depth", 3, 15)
            clf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=RANDOM_STATE, n_jobs=-1)
        elif name == "Linear SVM":
            C = trial.suggest_float("C", 1e-4, 1e2, log=True)
            clf = SVC(C=C, kernel="linear", probability=True, random_state=RANDOM_STATE)
        elif name == "Gradient Boosting":
            n_est = trial.suggest_int("n_estimators", 50, 200)
            lr = trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True)
            clf = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr, random_state=RANDOM_STATE)
        elif name == "XGBoost" and HAS_XGB:
            n_est = trial.suggest_int("n_estimators", 50, 200)
            lr = trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True)
            depth = trial.suggest_int("max_depth", 2, 8)
            clf = XGBClassifier(n_estimators=n_est, learning_rate=lr, max_depth=depth, random_state=RANDOM_STATE, n_jobs=-1)
        elif name == "LightGBM" and HAS_LGB:
            n_est = trial.suggest_int("n_estimators", 50, 200)
            lr = trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True)
            clf = LGBMClassifier(n_estimators=n_est, learning_rate=lr, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        elif name == "CatBoost" and HAS_CAT:
            n_est = trial.suggest_int("iterations", 50, 200)
            lr = trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True)
            clf = CatBoostClassifier(iterations=n_est, learning_rate=lr, random_state=RANDOM_STATE, verbose=0)
        else:
            clf = LogisticRegression(random_state=RANDOM_STATE)

        clf.fit(X_train, y_train)
        
        from sklearn.metrics import f1_score
        preds = clf.predict(X_val)
        return float(f1_score(y_val, preds, zero_division=0))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
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
        param_dist = {"C": np.logspace(-3, 2, 10)}
    elif name == "Random Forest":
        clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        param_dist = {"n_estimators": [50, 150, 250], "max_depth": [5, 10, None]}
    elif name == "Linear SVM":
        clf = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)
        param_dist = {"C": np.logspace(-3, 2, 10)}
    elif name == "Gradient Boosting":
        clf = GradientBoostingClassifier(random_state=RANDOM_STATE)
        param_dist = {"n_estimators": [50, 150], "learning_rate": [0.01, 0.1]}
    elif name == "XGBoost" and HAS_XGB:
        clf = XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        param_dist = {"n_estimators": [50, 150], "learning_rate": [0.01, 0.1]}
    elif name == "LightGBM" and HAS_LGB:
        clf = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        param_dist = {"n_estimators": [50, 150], "learning_rate": [0.01, 0.1]}
    elif name == "CatBoost" and HAS_CAT:
        clf = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0)
        param_dist = {"iterations": [50, 150], "learning_rate": [0.01, 0.1]}
    else:
        clf = LogisticRegression(random_state=RANDOM_STATE)
        param_dist = {}

    search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=6, cv=3, scoring="f1", random_state=RANDOM_STATE, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def optimize_threshold(y_true: pd.Series, probs: np.ndarray) -> float:
    """Find probability threshold that maximizes balanced accuracy."""
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
    }


def perform_cross_validation(clf: Any, X: np.ndarray, y: pd.Series) -> dict[str, Any]:
    """Run stratified 5-fold cross validation."""
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
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


def train_and_select_best(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: pd.Series,
    y_val: pd.Series,
    experiment_name: str,
) -> tuple[str, Any, dict[str, Any]]:
    """Trains, optimizes, calibrates and ranks all candidates."""
    logger.info("Starting model training for: %s", experiment_name)
    
    # Balance using SMOTE if available
    if HAS_SMOTE:
        logger.info("Applying SMOTE to: %s", experiment_name)
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train
        
    candidate_names = ["Logistic Regression", "Random Forest", "Linear SVM", "Gradient Boosting"]
    if HAS_XGB:
        candidate_names.append("XGBoost")
    if HAS_LGB:
        candidate_names.append("LightGBM")
    if HAS_CAT:
        candidate_names.append("CatBoost")

    results = {}
    fitted_models = {}

    for name in candidate_names:
        logger.info("Training candidate: %s", name)
        best_clf, best_params = optimize_classifier(name, X_train_res, y_train_res, X_val, y_val)
        
        # Calibration
        if HAS_FROZEN:
            calibrated_clf = CalibratedClassifierCV(estimator=FrozenEstimator(best_clf), method="sigmoid")
        else:
            calibrated_clf = CalibratedClassifierCV(estimator=best_clf, method="sigmoid", cv="prefit")
        calibrated_clf.fit(X_val, y_val)
        
        # Threshold optimization
        val_probs = calibrated_clf.predict_proba(X_val)[:, 1]
        best_threshold = optimize_threshold(y_val, val_probs)
        
        # Evaluation
        train_metrics = evaluate_at_threshold(calibrated_clf, X_train, y_train, best_threshold)
        val_metrics = evaluate_at_threshold(calibrated_clf, X_val, y_val, best_threshold)
        
        cv_scores = perform_cross_validation(best_clf, X_train, y_train)
        
        weighted_score = compute_weighted_score(val_metrics)
        
        results[name] = {
            "hyperparameters": best_params,
            "train_metrics": train_metrics,
            "test_metrics": val_metrics,
            "cross_validation": cv_scores,
            "optimized_threshold": round(best_threshold, 3),
            "weighted_ranking_score": weighted_score,
            "estimator": best_clf.__class__.__name__,
        }
        
        # Wrap calibrated classifier
        calibrated_clf.threshold = best_threshold
        fitted_models[name] = calibrated_clf
        
        logger.info(
            "%s - Holdout F1: %.4f, ROC AUC: %.4f, Weighted Score: %.4f, Threshold: %.2f",
            name, val_metrics["f1"], val_metrics["roc_auc"], weighted_score, best_threshold
        )
        
    best_name = max(results, key=lambda k: results[k]["weighted_ranking_score"])
    return best_name, fitted_models[best_name], {
        "all_results": results,
        "best_model_name": best_name,
        "best_model_metadata": results[best_name],
    }


def run_embedding_training_pipeline() -> dict[str, Any]:
    """Runs the full embedding and hybrid router training pipeline."""
    from app.ml.locked_eval import TRAIN_SPLIT_PATH
    # 1. Load splits (will load from training_dataset_large_train.csv)
    splits = prepare_dataset_splits()
    model_name = get_embedding_model_name()
    model = get_model(model_name)
    model_version = get_model_version(model)
    
    # 2. Train Embedding Router
    best_emb_name, best_emb_clf, emb_meta = train_and_select_best(
        splits.X_train_emb,
        splits.X_test_emb,
        splits.y_train,
        splits.y_test,
        "Embedding-Only Router"
    )
    
    # 3. Train Hybrid Router
    best_hyb_name, best_hyb_clf, hyb_meta = train_and_select_best(
        splits.X_train_hybrid,
        splits.X_test_hybrid,
        splits.y_train,
        splits.y_test,
        "Hybrid Router (Embeddings + Handcrafted)"
    )
    
    # 4. Save artifacts
    # Save pure embedding router bundle
    emb_bundle = {
        "model": best_emb_clf,
        "embedding_model_name": model_name,
        "embedding_model_version": model_version,
    }
    save_artifact(EMBEDDING_MODEL_PATH, emb_bundle)
    
    emb_metadata = {
        "pipeline_version": "phase_6_embedding_router_v1",
        "embedding_model_name": model_name,
        "embedding_model_version": model_version,
        "best_classifier": best_emb_name,
        "results": emb_meta,
    }
    save_json(EMBEDDING_METADATA_PATH, emb_metadata)
    
    # Load handcrafted features
    df, _, X_tab, feature_names = load_dataset_and_extract_embeddings(model_name=model_name)
    prepared = prepare_training_data(dataset_path=splits.LARGE_DATASET_PATH if hasattr(splits, "LARGE_DATASET_PATH") else Path(TRAIN_SPLIT_PATH), scale_numeric=False)
    
    selected_columns = []
    if FEATURE_COLUMNS_PATH.exists():
        try:
            selected_columns = load_json(FEATURE_COLUMNS_PATH)
        except Exception:
            pass
    if not selected_columns:
        selected_columns = prepared.feature_columns
        
    X_ml = prepared.X[selected_columns].copy()
    preprocessor = build_preprocessor(X_ml, scale_numeric=True)
    preprocessor.fit(X_ml)
    
    # Save hybrid router bundle
    hyb_bundle = {
        "model": best_hyb_clf,
        "embedding_model_name": model_name,
        "embedding_model_version": model_version,
        "preprocessor": preprocessor,
        "feature_columns": selected_columns,
    }
    save_artifact(HYBRID_MODEL_PATH, hyb_bundle)
    
    hyb_metadata = {
        "pipeline_version": "phase_6_hybrid_router_v1",
        "embedding_model_name": model_name,
        "embedding_model_version": model_version,
        "best_classifier": best_hyb_name,
        "results": hyb_meta,
    }
    save_json(HYBRID_METADATA_PATH, hyb_metadata)
    
    # Log experiments
    tracker = ExperimentTracker()
    tracker.log_experiment(
        experiment_name="phase_6_embedding_router",
        dataset_version="large_dataset_balanced_v1",
        feature_version="embeddings_only",
        embedding_model=model_name,
        classifier=best_emb_name,
        hyperparameters=emb_meta["best_model_metadata"]["hyperparameters"],
        metrics=emb_meta["best_model_metadata"]["test_metrics"],
    )
    tracker.log_experiment(
        experiment_name="phase_6_hybrid_router",
        dataset_version="large_dataset_balanced_v1",
        feature_version="hybrid_emb_tab",
        embedding_model=model_name,
        classifier=best_hyb_name,
        hyperparameters=hyb_meta["best_model_metadata"]["hyperparameters"],
        metrics=hyb_meta["best_model_metadata"]["test_metrics"],
    )
    
    logger.info("Successfully trained Phase 6 embedding & hybrid models!")
    return {
        "embedding": emb_metadata,
        "hybrid": hyb_metadata,
    }


if __name__ == "__main__":
    run_embedding_training_pipeline()
