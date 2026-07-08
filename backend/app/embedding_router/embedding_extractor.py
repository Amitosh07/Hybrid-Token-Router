"""Embedding extractor with SHA-256 prompt hashing, model name/version-based cache invalidation, and batch processing."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any
import numpy as np
import joblib

from app.embedding_router.embedding_utils import (
    get_model,
    get_embedding_model_name,
    get_model_version,
    DATA_DIR,
)

logger = logging.getLogger(__name__)

CACHE_PATH = DATA_DIR / "processed" / "embedding_cache.pkl"


def get_prompt_hash(prompt: str) -> str:
    """Compute the SHA-256 hash of a prompt string."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class EmbeddingExtractor:
    """Extracts embeddings using SentenceTransformers with model-versioned caching."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or get_embedding_model_name()
        self.model = get_model(self.model_name)
        self.model_version = get_model_version(self.model)
        self.cache: dict[str, np.ndarray] = {}
        self.load_cache()

    def load_cache(self) -> None:
        """Load embedding cache from disk."""
        if CACHE_PATH.exists():
            try:
                self.cache = joblib.load(CACHE_PATH)
                logger.info("Loaded embedding cache with %d entries", len(self.cache))
            except Exception as e:
                logger.warning("Could not load embedding cache: %s. Starting fresh.", e)
                self.cache = {}
        else:
            self.cache = {}

    def save_cache(self) -> None:
        """Persist embedding cache to disk."""
        try:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.cache, CACHE_PATH)
            logger.debug("Saved embedding cache with %d entries", len(self.cache))
        except Exception as e:
            logger.error("Failed to save embedding cache: %s", e)

    def _get_cache_key(self, prompt: str) -> str:
        """Generate a versioned cache key for a prompt."""
        p_hash = get_prompt_hash(prompt)
        return f"{self.model_name}:{self.model_version}:{p_hash}"

    def extract(self, prompts: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
        """Extract embeddings for a list of prompts.
        
        Uses cache to avoid recomputing, and batches missing prompts.
        
        Returns:
            Tuple of (embeddings_array, stats_dict)
        """
        embeddings = [None] * len(prompts)
        missing_indices = []
        missing_prompts = []

        # Check cache
        for idx, prompt in enumerate(prompts):
            key = self._get_cache_key(prompt)
            if key in self.cache:
                embeddings[idx] = self.cache[key]
            else:
                missing_indices.append(idx)
                missing_prompts.append(prompt)

        extraction_latency = 0.0
        if missing_prompts:
            start_time = time.perf_counter()
            # Batch encode the missing prompts
            encoded = self.model.encode(
                missing_prompts,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            extraction_latency = time.perf_counter() - start_time
            logger.info(
                "Extracted %d new embeddings in %.4f seconds (batch size: %d)",
                len(missing_prompts), extraction_latency, len(missing_prompts)
            )

            # Store in cache and result list
            for idx, emb in zip(missing_indices, encoded):
                key = self._get_cache_key(prompts[idx])
                self.cache[key] = emb
                embeddings[idx] = emb

            # Save the cache back to disk
            self.save_cache()

        # Convert list of arrays to a single 2D numpy array
        embeddings_array = np.vstack(embeddings)
        
        stats = {
            "batch_size": len(prompts),
            "cached_hits": len(prompts) - len(missing_prompts),
            "uncached_misses": len(missing_prompts),
            "extraction_latency_seconds": extraction_latency,
            "avg_extraction_latency_ms": (extraction_latency * 1000.0 / len(missing_prompts)) if missing_prompts else 0.0,
        }

        return embeddings_array, stats
