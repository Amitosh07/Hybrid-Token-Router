"""Embedding utilities for dynamic model loading, hardware acceleration, and benchmarking."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any
import torch
from sentence_transformers import SentenceTransformer
from app.config import get_settings

logger = logging.getLogger(__name__)

# Module-level model cache to avoid reloading
_MODEL_CACHE: dict[str, SentenceTransformer] = {}
_MODEL_LOAD_TIMES: dict[str, float] = {}

# Output paths
EMBEDDING_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EMBEDDING_DIR.parents[1]
MODELS_DIR = BACKEND_DIR / "models"
DATA_DIR = BACKEND_DIR / "app" / "data"

# Model paths
EMBEDDING_MODEL_PATH = MODELS_DIR / "embedding_classifier.pkl"
EMBEDDING_METADATA_PATH = MODELS_DIR / "embedding_metadata.json"

HYBRID_MODEL_PATH = MODELS_DIR / "hybrid_classifier.pkl"
HYBRID_METADATA_PATH = MODELS_DIR / "hybrid_metadata.json"


def get_device() -> str:
    """Determine hardware device. Supports CUDA/ROCm and falls back to CPU."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_embedding_model_name() -> str:
    """Retrieve embedding model name from config."""
    try:
        return get_settings().EMBEDDING_MODEL_NAME
    except Exception:
        return "BAAI/bge-small-en-v1.5"


class MockLayer:
    def __init__(self) -> None:
        class Encoder:
            def __init__(self) -> None:
                self.layer = [1] * 12
        self.auto_model = type("AutoModel", (), {"encoder": Encoder()})()


class MockSentenceTransformer:
    """Deterministic simulated SentenceTransformer for offline execution."""

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        if "small" in model_name.lower():
            self.dim = 384
        elif "large" in model_name.lower() or "jina" in model_name.lower():
            self.dim = 1024
        else:  # base
            self.dim = 768

    def encode(self, sentences: list[str] | str, **kwargs: Any) -> np.ndarray:
        import hashlib
        import numpy as np
        if isinstance(sentences, str):
            sentences = [sentences]
        
        embeddings = []
        for s in sentences:
            h = hashlib.sha256(s.encode("utf-8")).digest()
            # Deterministic embedding generation using SHA-256 seed
            seed = int.from_bytes(h[:4], byteorder="big") % (2**32 - 1)
            rng = np.random.default_rng(seed)
            embeddings.append(rng.standard_normal(self.dim))
        
        return np.array(embeddings, dtype=np.float32)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def get_embedding_dimension(self) -> int:
        return self.dim

    def __getitem__(self, index: int) -> Any:
        return MockLayer()


def get_model(model_name: str | None = None) -> Any:
    """Get or load a SentenceTransformer model dynamically and cache it."""
    name = model_name or get_embedding_model_name()
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]

    device = get_device()
    logger.info("Loading embedding model %s on device: %s", name, device)
    
    start_time = time.perf_counter()
    try:
        # Load model using sentence-transformers
        model = SentenceTransformer(name, device=device)
        logger.info("Loaded real embedding model %s", name)
    except Exception as exc:
        logger.warning("Could not load real embedding model %s (%s). Falling back to mock encoder.", name, exc)
        model = MockSentenceTransformer(name, device=device)
        
    load_time = time.perf_counter() - start_time
    
    _MODEL_CACHE[name] = model
    _MODEL_LOAD_TIMES[name] = load_time
    logger.info("Loaded embedding model %s in %.4f seconds", name, load_time)
    
    return model


def get_model_load_time(model_name: str | None = None) -> float:
    """Get the time taken to load the specified model, or 0.0 if not loaded yet."""
    name = model_name or get_embedding_model_name()
    return _MODEL_LOAD_TIMES.get(name, 0.0)


def get_model_version(model: Any) -> str:
    """Get model version or config parameters for cache key versioning."""
    try:
        if isinstance(model, MockSentenceTransformer):
            return f"mock_dim_{model.dim}"
        num_layers = len(model[0].auto_model.encoder.layer) if hasattr(model[0], "auto_model") else 0
        hidden_size = model.get_sentence_embedding_dimension() or 384
        return f"layers_{num_layers}_dim_{hidden_size}"
    except Exception:
        return "v1"
