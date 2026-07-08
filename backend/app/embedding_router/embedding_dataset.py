"""Data loader and dataset builder for embedding-based and hybrid-based routing models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.config import get_settings
from app.ml.preprocess import prepare_training_data, build_preprocessor
from app.ml.model_utils import provider_to_numeric, FEATURE_COLUMNS_PATH, load_json
from app.embedding_router.embedding_extractor import EmbeddingExtractor
from app.embedding_router.embedding_utils import DATA_DIR

logger = logging.getLogger(__name__)

LARGE_DATASET_PATH = DATA_DIR / "training" / "training_dataset_large_train.csv"


@dataclass
class DatasetSplits:
    """Container for split data arrays."""
    # Pure embeddings
    X_train_emb: np.ndarray
    X_test_emb: np.ndarray
    
    # Hybrid features (embeddings + handcrafted)
    X_train_hybrid: np.ndarray
    X_test_hybrid: np.ndarray
    
    # Tabular features (handcrafted only, scaled)
    X_train_tab: np.ndarray
    X_test_tab: np.ndarray
    
    # Targets
    y_train: pd.Series
    y_test: pd.Series
    
    # Original Dataframe index mapping
    train_indices: np.ndarray
    test_indices: np.ndarray


def load_dataset_and_extract_embeddings(
    dataset_path: Path = LARGE_DATASET_PATH,
    model_name: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Loads the dataset, extracts prompt embeddings, and prepares scaled handcrafted features.
    
    Returns:
        Tuple of (df, embeddings, scaled_handcrafted, handcrafted_feature_names)
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Large dataset not found: {dataset_path}")
        
    df = pd.read_csv(dataset_path)
    logger.info("Loaded dataset from %s with %d rows", dataset_path, len(df))
    
    # 1. Extract embeddings for every prompt
    extractor = EmbeddingExtractor(model_name=model_name)
    prompts = df["prompt"].astype(str).tolist()
    logger.info("Extracting embeddings for %d prompts (caching enabled)...", len(prompts))
    embeddings, stats = extractor.extract(prompts)
    logger.info("Embeddings extraction stats: %s", stats)
    
    # 2. Extract and scale handcrafted features
    prepared = prepare_training_data(dataset_path=dataset_path, scale_numeric=False)
    
    # Load feature columns selected in Phase 3
    selected_columns = []
    if FEATURE_COLUMNS_PATH.exists():
        try:
            selected_columns = load_json(FEATURE_COLUMNS_PATH)
            logger.info("Loaded %d selected handcrafted feature columns", len(selected_columns))
        except Exception as e:
            logger.warning("Could not load FEATURE_COLUMNS_PATH: %s. Using all features.", e)
    
    if not selected_columns:
        selected_columns = prepared.feature_columns
        
    X_ml = prepared.X[selected_columns].copy()
    
    # Scale tabular features
    preprocessor = build_preprocessor(X_ml, scale_numeric=True)
    X_ml_scaled = preprocessor.fit_transform(X_ml)
    
    # Convert sparse array to dense if returned by preprocessor (e.g. if categorical variables are one-hot encoded)
    if hasattr(X_ml_scaled, "toarray"):
        X_ml_scaled = X_ml_scaled.toarray()
    elif not isinstance(X_ml_scaled, np.ndarray):
        X_ml_scaled = np.array(X_ml_scaled)
        
    # Get feature names from preprocessor
    feature_names = list(preprocessor.get_feature_names_out())
    
    return df, embeddings, X_ml_scaled, feature_names


def prepare_dataset_splits(
    dataset_path: Path = LARGE_DATASET_PATH,
    model_name: str | None = None,
    random_state: int = 42,
) -> DatasetSplits:
    """Prepares and splits the embedding and hybrid datasets into 80/20 train/test splits."""
    df, X_emb, X_tab, tab_feature_names = load_dataset_and_extract_embeddings(
        dataset_path=dataset_path,
        model_name=model_name,
    )
    
    # Target values
    y = df["label"].map(provider_to_numeric)
    
    # Concat features for Hybrid router
    X_hybrid = np.hstack([X_emb, X_tab])
    
    # Indices for alignment
    indices = np.arange(len(df))
    
    # Split
    (
        train_idx,
        test_idx,
        y_train,
        y_test,
    ) = train_test_split(
        indices,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y,
    )
    
    return DatasetSplits(
        X_train_emb=X_emb[train_idx],
        X_test_emb=X_emb[test_idx],
        X_train_hybrid=X_hybrid[train_idx],
        X_test_hybrid=X_hybrid[test_idx],
        X_train_tab=X_tab[train_idx],
        X_test_tab=X_tab[test_idx],
        y_train=y_train,
        y_test=y_test,
        train_indices=train_idx,
        test_indices=test_idx,
    )
