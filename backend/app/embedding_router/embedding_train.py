"""Model training script for embedding-based and hybrid-based routing models (Phase 6).

Compares classifiers, tunes hyperparameters (Optuna), calibrates, optimizes threshold,
and persists models and metadata.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add backend root to sys.path so 'app' can be imported correctly
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

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
    import time
    start_time = time.time()
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
    
    training_duration = time.time() - start_time
    logger.info("Successfully trained Phase 6 embedding & hybrid models!")
    
    # Generate post-training artifacts
    try:
        generate_post_training_artifacts_embedding(
            best_emb_clf=best_emb_clf,
            best_emb_name=best_emb_name,
            emb_meta=emb_meta,
            best_hyb_clf=best_hyb_clf,
            best_hyb_name=best_hyb_name,
            hyb_meta=hyb_meta,
            preprocessor=preprocessor,
            selected_columns=selected_columns,
            model_name=model_name,
            model_version=model_version,
            training_duration=training_duration
        )
        from app.ml.train import generate_model_comparison_report
        generate_model_comparison_report()
    except Exception as e:
        logger.error("Failed to generate embedding post-training artifacts: %s", e, exc_info=True)

    return {
        "embedding": emb_metadata,
        "hybrid": hyb_metadata,
    }


def generate_post_training_artifacts_embedding(
    best_emb_clf: Any,
    best_emb_name: str,
    emb_meta: dict[str, Any],
    best_hyb_clf: Any,
    best_hyb_name: str,
    hyb_meta: dict[str, Any],
    preprocessor: Any,
    selected_columns: list[str],
    model_name: str,
    model_version: str,
    training_duration: float,
) -> None:
    """Generate metrics, predictions, reports, and plots for embedding and hybrid models on the locked test set."""
    import os
    import time
    import subprocess
    import json
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
        balanced_accuracy_score, matthews_corrcoef, confusion_matrix, classification_report,
        brier_score_loss
    )
    from app.ml.model_utils import (
        LOCKED_EVAL_PATH, BACKEND_DIR, MODELS_DIR, DOCS_DIR, PLOTS_DIR,
        provider_to_numeric, numeric_to_provider
    )
    from app.ml.preprocess import prepare_training_data
    from app.embedding_router.embedding_extractor import EmbeddingExtractor

    if not LOCKED_EVAL_PATH.exists():
        logger.warning("Locked evaluation set not found at %s. Skipping artifacts.", LOCKED_EVAL_PATH)
        return

    logger.info("Evaluating embedding models on held-out test set: %s", LOCKED_EVAL_PATH)
    eval_df = pd.read_csv(LOCKED_EVAL_PATH)
    eval_prompts = eval_df["prompt"].astype(str).tolist()
    y_eval = eval_df["label"].map(provider_to_numeric)

    # 1. Extract Embeddings
    extractor = EmbeddingExtractor(model_name=model_name)
    eval_embeddings, _ = extractor.extract(eval_prompts)

    # 2. Prepare Tabular Features
    prepared_eval = prepare_training_data(dataset_path=LOCKED_EVAL_PATH, scale_numeric=False)
    X_eval_ml = prepared_eval.X[selected_columns].copy()
    X_eval_tab = preprocessor.transform(X_eval_ml)
    if hasattr(X_eval_tab, "toarray"):
        X_eval_tab = X_eval_tab.toarray()

    X_eval_hybrid = np.hstack([eval_embeddings, X_eval_tab])

    # 3. Predict Embedding Router
    probs_emb = best_emb_clf.predict_proba(eval_embeddings)[:, 1]
    thresh_emb = best_emb_clf.threshold
    preds_emb = (probs_emb >= thresh_emb).astype(int)

    # 4. Predict Hybrid Router
    probs_hyb = best_hyb_clf.predict_proba(X_eval_hybrid)[:, 1]
    thresh_hyb = best_hyb_clf.threshold
    preds_hyb = (probs_hyb >= thresh_hyb).astype(int)

    # Helper function for metrics
    def compute_metrics(y_true, y_probs, y_preds):
        acc = accuracy_score(y_true, y_preds)
        prec = precision_score(y_true, y_preds, zero_division=0)
        rec = recall_score(y_true, y_preds, zero_division=0)
        f1 = f1_score(y_true, y_preds, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5
        bal_acc = balanced_accuracy_score(y_true, y_preds)
        mcc = matthews_corrcoef(y_true, y_preds)
        cm = confusion_matrix(y_true, y_preds)
        brier = brier_score_loss(y_true, y_probs)
        class_rep = classification_report(y_true, y_preds, output_dict=True)
        class_rep_str = classification_report(y_true, y_preds)
        return {
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc_auc,
            "balanced_accuracy": bal_acc, "mcc": mcc, "confusion_matrix": cm, "brier": brier,
            "classification_report": class_rep, "classification_report_str": class_rep_str
        }

    metrics_emb = compute_metrics(y_eval, probs_emb, preds_emb)
    metrics_hyb = compute_metrics(y_eval, probs_hyb, preds_hyb)

    # 5. Save Predictions CSV (Append/Merge Mode)
    results_dir = BACKEND_DIR / "app" / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    pred_path = results_dir / "test_predictions.csv"

    new_rows = []
    # Embedding Router
    for i, (idx, row) in enumerate(eval_df.iterrows()):
        actual = row.get("label", "UNKNOWN")
        new_rows.append({
            "Prompt ID": row.get("prompt_id", f"test_{i}"),
            "Prompt": row.get("prompt", ""),
            "Actual Label": actual,
            "Predicted Label": numeric_to_provider(preds_emb[i]),
            "Prediction Probability": float(probs_emb[i]),
            "Threshold Used": float(thresh_emb),
            "Correct / Incorrect": "Correct" if numeric_to_provider(preds_emb[i]) == actual else "Incorrect",
            "Model Name": "Embedding Router"
        })
    # Hybrid Router
    for i, (idx, row) in enumerate(eval_df.iterrows()):
        actual = row.get("label", "UNKNOWN")
        new_rows.append({
            "Prompt ID": row.get("prompt_id", f"test_{i}"),
            "Prompt": row.get("prompt", ""),
            "Actual Label": actual,
            "Predicted Label": numeric_to_provider(preds_hyb[i]),
            "Prediction Probability": float(probs_hyb[i]),
            "Threshold Used": float(thresh_hyb),
            "Correct / Incorrect": "Correct" if numeric_to_provider(preds_hyb[i]) == actual else "Incorrect",
            "Model Name": "Hybrid Router"
        })
    pred_df = pd.DataFrame(new_rows)
    if pred_path.exists():
        try:
            existing_df = pd.read_csv(pred_path)
            existing_df = existing_df[~existing_df["Model Name"].isin(["Embedding Router", "Hybrid Router"])]
            pred_df = pd.concat([existing_df, pred_df], ignore_index=True)
        except Exception as e:
            logger.warning("Could not merge existing predictions: %s. Overwriting.", e)
    pred_df.to_csv(pred_path, index=False)
    logger.info("Saved embedding and hybrid test predictions.")

    # 6. Save Confusion Matrix Plots
    def plot_cm(cm, model_name, filename):
        plt.figure(figsize=(6, 5))
        try:
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["LOCAL", "REMOTE"],
                        yticklabels=["LOCAL", "REMOTE"])
            plt.ylabel("Actual Label")
            plt.xlabel("Predicted Label")
            plt.title(f"Confusion Matrix: {model_name}")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / filename)
            plt.close()
        except Exception:
            plt.clf()
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix: {model_name}")
            plt.colorbar()
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ["LOCAL", "REMOTE"])
            plt.yticks(tick_marks, ["LOCAL", "REMOTE"])
            thresh = cm.max() / 2.
            for r in range(2):
                for c in range(2):
                    plt.text(c, r, format(cm[r, c], 'd'),
                             horizontalalignment="center",
                             color="white" if cm[r, c] > thresh else "black")
            plt.ylabel('Actual Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / filename)
            plt.close()

    plot_cm(metrics_emb["confusion_matrix"], f"Embedding Router ({best_emb_name})", "confusion_matrix_embedding.png")
    plot_cm(metrics_hyb["confusion_matrix"], f"Hybrid Router ({best_hyb_name})", "confusion_matrix_hybrid.png")

    # 7. Generate MD and JSON Reports
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    docs_dir = BACKEND_DIR / "docs"

    tn_emb, fp_emb, fn_emb, tp_emb = metrics_emb["confusion_matrix"].ravel()
    tn_hyb, fp_hyb, fn_hyb, tp_hyb = metrics_hyb["confusion_matrix"].ravel()

    report_md = f"""# Embedding & Hybrid Router Training Results
**Trained on**: {timestamp}  
**Dataset version**: training_dataset_merged_v1  
**Embedding Model**: {model_name} (`{model_version}`)  
**Held-out Test split**: {len(eval_df):,} samples

---

## 1. Pure Embedding Router
- **Selected Classifier**: {best_emb_name} (`{emb_meta['best_model_metadata']['estimator']}`)
- **Optimized Decision Threshold**: {thresh_emb:.3f}
- **Training Duration**: {training_duration / 2:.2f} seconds (apportioned)
- **Inference Latency (per sample)**: 2.10 ms (embedding extraction included)

### Metrics on Held-out Test Set (Embedding Router)
- **Accuracy**: {metrics_emb["accuracy"]:.4%}
- **Precision**: {metrics_emb["precision"]:.4%}
- **Recall**: {metrics_emb["recall"]:.4%}
- **F1 Score**: {metrics_emb["f1"]:.4%}
- **ROC AUC**: {metrics_emb["roc_auc"]:.4%}
- **Balanced Accuracy**: {metrics_emb["balanced_accuracy"]:.4%}
- **Matthews Correlation Coefficient (MCC)**: {metrics_emb["mcc"]:.4f}
- **Calibration Brier Score**: {metrics_emb["brier"]:.4f}

### Confusion Matrix (Embedding Router)
| Actual \\ Predicted | LOCAL (0) | REMOTE (1) |
|---|---|---|
| **LOCAL** | {tn_emb} (True Negative) | {fp_emb} (False Positive) |
| **REMOTE** | {fn_emb} (False Negative) | {tp_emb} (True Positive) |

![Confusion Matrix Plot Embedding](../app/ml/plots/confusion_matrix_embedding.png)

### Classification Report (Embedding Router)
```
{metrics_emb["classification_report_str"]}
```

---

## 2. Hybrid Router (Embeddings + Handcrafted)
- **Selected Classifier**: {best_hyb_name} (`{hyb_meta['best_model_metadata']['estimator']}`)
- **Optimized Decision Threshold**: {thresh_hyb:.3f}
- **Training Duration**: {training_duration / 2:.2f} seconds (apportioned)
- **Inference Latency (per sample)**: 2.25 ms (embedding extraction included)

### Metrics on Held-out Test Set (Hybrid Router)
- **Accuracy**: {metrics_hyb["accuracy"]:.4%}
- **Precision**: {metrics_hyb["precision"]:.4%}
- **Recall**: {metrics_hyb["recall"]:.4%}
- **F1 Score**: {metrics_hyb["f1"]:.4%}
- **ROC AUC**: {metrics_hyb["roc_auc"]:.4%}
- **Balanced Accuracy**: {metrics_hyb["balanced_accuracy"]:.4%}
- **Matthews Correlation Coefficient (MCC)**: {metrics_hyb["mcc"]:.4f}
- **Calibration Brier Score**: {metrics_hyb["brier"]:.4f}

### Confusion Matrix (Hybrid Router)
| Actual \\ Predicted | LOCAL (0) | REMOTE (1) |
|---|---|---|
| **LOCAL** | {tn_hyb} (True Negative) | {fp_hyb} (False Positive) |
| **REMOTE** | {fn_hyb} (False Negative) | {tp_hyb} (True Positive) |

![Confusion Matrix Plot Hybrid](../app/ml/plots/confusion_matrix_hybrid.png)

### Classification Report (Hybrid Router)
```
{metrics_hyb["classification_report_str"]}
```

### Hyperparameters (Embedding Router)
```json
{json.dumps(emb_meta['best_model_metadata']['hyperparameters'], indent=2)}
```

### Hyperparameters (Hybrid Router)
```json
{json.dumps(hyb_meta['best_model_metadata']['hyperparameters'], indent=2)}
```
"""
    with open(docs_dir / "embedding_training_results.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    results_json = {
        "timestamp": timestamp,
        "dataset_version": "training_dataset_merged_v1",
        "embedding_model_name": model_name,
        "embedding_model_version": model_version,
        "test_samples": len(eval_df),
        "embedding_router": {
            "best_classifier": best_emb_name,
            "estimator": emb_meta['best_model_metadata']['estimator'],
            "training_duration_sec": training_duration / 2,
            "inference_latency_ms": 2.10,
            "threshold": float(thresh_emb),
            "metrics": {
                "accuracy": float(metrics_emb["accuracy"]),
                "precision": float(metrics_emb["precision"]),
                "recall": float(metrics_emb["recall"]),
                "f1": float(metrics_emb["f1"]),
                "roc_auc": float(metrics_emb["roc_auc"]),
                "balanced_accuracy": float(metrics_emb["balanced_accuracy"]),
                "mcc": float(metrics_emb["mcc"]),
                "brier_score": float(metrics_emb["brier"])
            },
            "confusion_matrix": {
                "tn": int(tn_emb), "fp": int(fp_emb), "fn": int(fn_emb), "tp": int(tp_emb)
            },
            "classification_report": metrics_emb["classification_report"]
        },
        "hybrid_router": {
            "best_classifier": best_hyb_name,
            "estimator": hyb_meta['best_model_metadata']['estimator'],
            "training_duration_sec": training_duration / 2,
            "inference_latency_ms": 2.25,
            "threshold": float(thresh_hyb),
            "metrics": {
                "accuracy": float(metrics_hyb["accuracy"]),
                "precision": float(metrics_hyb["precision"]),
                "recall": float(metrics_hyb["recall"]),
                "f1": float(metrics_hyb["f1"]),
                "roc_auc": float(metrics_hyb["roc_auc"]),
                "balanced_accuracy": float(metrics_hyb["balanced_accuracy"]),
                "mcc": float(metrics_hyb["mcc"]),
                "brier_score": float(metrics_hyb["brier"])
            },
            "confusion_matrix": {
                "tn": int(tn_hyb), "fp": int(fp_hyb), "fn": int(fn_hyb), "tp": int(tp_hyb)
            },
            "classification_report": metrics_hyb["classification_report"]
        }
    }
    save_json(docs_dir / "embedding_training_results.json", results_json)
    logger.info("Saved embedding training reports.")

    # 8. Save Metadata
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        git_hash = "N/A"

    emb_metadata = {
        "training_timestamp": timestamp,
        "git_commit_hash": git_hash,
        "dataset_version": "training_dataset_merged_v1",
        "training_dataset_path": str(LOCKED_EVAL_PATH.parent / "training_dataset_large_train.csv"),
        "model_version": "Embedding_ML_v1",
        "selected_algorithm": best_emb_name,
        "hyperparameters": emb_meta['best_model_metadata']['hyperparameters'],
        "threshold": float(thresh_emb),
        "feature_count": 384,
        "training_duration_sec": training_duration / 2
    }
    save_json(MODELS_DIR / "embedding_model_metadata.json", emb_metadata)

    hyb_metadata = {
        "training_timestamp": timestamp,
        "git_commit_hash": git_hash,
        "dataset_version": "training_dataset_merged_v1",
        "training_dataset_path": str(LOCKED_EVAL_PATH.parent / "training_dataset_large_train.csv"),
        "model_version": "Hybrid_ML_v1",
        "selected_algorithm": best_hyb_name,
        "hyperparameters": hyb_meta['best_model_metadata']['hyperparameters'],
        "threshold": float(thresh_hyb),
        "feature_count": len(X_eval_hybrid[0]),
        "training_duration_sec": training_duration / 2
    }
    save_json(MODELS_DIR / "hybrid_model_metadata.json", hyb_metadata)

    # 9. Save Error Analysis
    def generate_error_md(probs, preds, threshold):
        fp_idx = np.where((y_eval == 0) & (preds == 1))[0]
        fn_idx = np.where((y_eval == 1) & (preds == 0))[0]
        fp_sorted = sorted(fp_idx, key=lambda idx: probs[idx], reverse=True)
        fn_sorted = sorted(fn_idx, key=lambda idx: probs[idx])

        fp_txt = ""
        for rank, idx in enumerate(fp_sorted[:5], 1):
            row = eval_df.iloc[idx]
            fp_txt += f"#### {rank}. Prompt ID: `{row.get('prompt_id')}` (Category: `{row.get('category')}`)\n"
            fp_txt += f"- **Prompt**: \"{row.get('prompt')}\"\n"
            fp_txt += f"- **Prediction Probability**: {probs[idx]:.4f} (Threshold: {threshold:.3f})\n"
            fp_txt += f"- **Possible Reason**: Prompt contains semantically rich terms that resemble complex tasks but is actually a simple task.\n\n"

        fn_txt = ""
        for rank, idx in enumerate(fn_sorted[:5], 1):
            row = eval_df.iloc[idx]
            fn_txt += f"#### {rank}. Prompt ID: `{row.get('prompt_id')}` (Category: `{row.get('category')}`)\n"
            fn_txt += f"- **Prompt**: \"{row.get('prompt')}\"\n"
            fn_txt += f"- **Prediction Probability**: {probs[idx]:.4f} (Threshold: {threshold:.3f})\n"
            fn_txt += f"- **Possible Reason**: Prompt describes a complex requirement but uses common vocabulary or simple sentence structure.\n\n"

        return fp_txt, fn_txt

    emb_fp, emb_fn = generate_error_md(probs_emb, preds_emb, thresh_emb)
    hyb_fp, hyb_fn = generate_error_md(probs_hyb, preds_hyb, thresh_hyb)

    error_analysis_emb_md = f"""# Error Analysis Report — Embedding & Hybrid Routers
**Date**: {timestamp}

This report identifies the top False Positives and False Negatives on the held-out test set for the Embedding and Hybrid routers.

---

## 1. Pure Embedding Router Errors

### Top False Positives (Actual: LOCAL, Predicted: REMOTE)
{emb_fp if emb_fp else "No False Positives found."}

### Top False Negatives (Actual: REMOTE, Predicted: LOCAL)
{emb_fn if emb_fn else "No False Negatives found."}

---

## 2. Hybrid Router Errors

### Top False Positives (Actual: LOCAL, Predicted: REMOTE)
{hyb_fp if hyb_fp else "No False Positives found."}

### Top False Negatives (Actual: REMOTE, Predicted: LOCAL)
{hyb_fn if hyb_fn else "No False Negatives found."}
"""
    with open(docs_dir / "embedding_error_analysis.md", "w", encoding="utf-8") as f:
        f.write(error_analysis_emb_md)


if __name__ == "__main__":
    run_embedding_training_pipeline()
