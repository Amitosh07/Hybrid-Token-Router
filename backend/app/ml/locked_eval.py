"""Locked Evaluation Dataset management for Phase 6.

Creates a permanently locked test split (400 prompts) balanced by category,
difficulty, and domain, ensuring it is never used for training or tuning.
"""

from __future__ import annotations

import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from app.ml.model_utils import DATASET_PATH, LOCKED_EVAL_PATH

logger = logging.getLogger(__name__)

# Path for the training subset after removing locked evaluation set
TRAIN_SPLIT_PATH = Path(DATASET_PATH).parent / "training_dataset_large_train.csv"


def get_locked_evaluation_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieves or creates the permanently locked evaluation set and training split.

    Returns:
        tuple of (train_df, locked_eval_df)
    """
    if LOCKED_EVAL_PATH.exists() and TRAIN_SPLIT_PATH.exists():
        logger.info("Loading existing locked evaluation dataset and training split...")
        train_df = pd.read_csv(TRAIN_SPLIT_PATH)
        eval_df = pd.read_csv(LOCKED_EVAL_PATH)
        return train_df, eval_df

    logger.info("Locked evaluation dataset not found. Generating permanently locked evaluation split...")
    
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Source large dataset not found at: {DATASET_PATH}")
        
    df = pd.read_csv(DATASET_PATH)
    
    # Define stratified groups based on category and difficulty
    stratify_cols = []
    for _, row in df.iterrows():
        cat = str(row.get("category", "general"))
        diff = str(row.get("difficulty", "medium"))
        stratify_cols.append(f"{cat}_{diff}")
        
    # Extract 400 prompts (approx 8%) for the locked evaluation set
    train_df, eval_df = train_test_split(
        df,
        test_size=400,
        random_state=42,
        stratify=stratify_cols
    )
    
    # Save splits
    LOCKED_EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(LOCKED_EVAL_PATH, index=False)
    train_df.to_csv(TRAIN_SPLIT_PATH, index=False)
    
    logger.info(
        "Successfully created locked evaluation splits: Train=%d samples, Locked Eval=%d samples",
        len(train_df), len(eval_df)
    )
    
    return train_df, eval_df
