"""Unit tests for the embedding router, hybrid router, and router dispatcher."""

from __future__ import annotations

import os
from typing import Generator
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from app.config import Settings
from app.services.feature_extractor import extract_features
from app.embedding_router.embedding_utils import (
    get_device,
    get_embedding_model_name,
    get_model_load_time,
)
from app.embedding_router.embedding_extractor import EmbeddingExtractor, get_prompt_hash
from app.embedding_router.embedding_predict import (
    route_embedding,
    route_hybrid,
    clear_predict_cache,
)
from app.services.router_dispatcher import route as dispatch_route


def test_embedding_utils() -> None:
    """Test utility helpers for device and config."""
    device = get_device()
    assert device in {"cuda", "cpu"}
    
    model_name = get_embedding_model_name()
    assert isinstance(model_name, str)
    assert len(model_name) > 0


def test_prompt_hash() -> None:
    """Test SHA-256 prompt hashing function."""
    prompt = "Write hello world in Python"
    p_hash = get_prompt_hash(prompt)
    assert len(p_hash) == 64
    assert p_hash == get_prompt_hash(prompt)
    assert p_hash != get_prompt_hash("Different prompt")


def test_embedding_extractor() -> None:
    """Test the EmbeddingExtractor using mocked models to avoid downloading during unit tests."""
    with patch("app.embedding_router.embedding_extractor.get_model") as mock_get_model:
        # Mock SentenceTransformer model behavior
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(2, 384)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_get_model.return_value = mock_model
        
        # Instantiate extractor and clear its cache to guarantee encode is called
        extractor = EmbeddingExtractor(model_name="BAAI/bge-small-en-v1.5")
        extractor.cache = {}
        assert extractor.model_name == "BAAI/bge-small-en-v1.5"
        
        # Extract embeddings
        prompts = ["Test prompt 1", "Test prompt 2"]
        embeddings, stats = extractor.extract(prompts)
        
        assert embeddings.shape == (2, 384)
        assert stats["batch_size"] == 2
        assert mock_model.encode.called


def test_router_predictions() -> None:
    """Test predictions and responses from embedding and hybrid routing functions."""
    prompt = "Design a low-latency web server in Go."
    features = extract_features(prompt)
    
    # Test route_embedding response shape
    res_emb = route_embedding(prompt, features)
    assert "provider" in res_emb
    assert "selected_provider" in res_emb
    assert "prediction_probability" in res_emb
    assert "prediction_confidence" in res_emb
    assert "confidence" in res_emb
    assert "model_version" in res_emb
    assert "routing_method" in res_emb
    assert "reason" in res_emb
    assert res_emb["routing_method"] in {"Embedding", "Heuristic Fallback"}
    
    # Test route_hybrid response shape
    res_hyb = route_hybrid(prompt, features)
    assert "provider" in res_hyb
    assert "selected_provider" in res_hyb
    assert "prediction_probability" in res_hyb
    assert "prediction_confidence" in res_hyb
    assert "confidence" in res_hyb
    assert "model_version" in res_hyb
    assert "routing_method" in res_hyb
    assert "reason" in res_hyb
    assert res_hyb["routing_method"] in {"Hybrid", "Heuristic Fallback"}


def test_dispatcher_modes() -> None:
    """Test dispatcher execution across different router modes."""
    prompt = "Create a database schema for an e-commerce platform."
    features = extract_features(prompt)
    
    # 1. Heuristic mode
    with patch("app.services.router_dispatcher.get_settings") as mock_settings:
        mock_config = MagicMock(spec=Settings)
        mock_config.ROUTER_MODE = "heuristic"
        mock_settings.return_value = mock_config
        
        res = dispatch_route(features, prompt=prompt)
        assert res["routing_method"] == "Heuristic"
        assert "selected_provider" in res

    # 2. Traditional ML mode
    with patch("app.services.router_dispatcher.get_settings") as mock_settings:
        mock_config = MagicMock(spec=Settings)
        mock_config.ROUTER_MODE = "ml"
        mock_settings.return_value = mock_config
        
        res = dispatch_route(features, prompt=prompt)
        assert res["routing_method"] in {"ML", "Heuristic Fallback"}
        assert "selected_provider" in res

    # 3. Embedding mode
    with patch("app.services.router_dispatcher.get_settings") as mock_settings:
        mock_config = MagicMock(spec=Settings)
        mock_config.ROUTER_MODE = "embedding"
        mock_settings.return_value = mock_config
        
        res = dispatch_route(features, prompt=prompt)
        assert res["routing_method"] in {"Embedding", "Heuristic Fallback"}
        assert "selected_provider" in res

    # 4. Hybrid mode
    with patch("app.services.router_dispatcher.get_settings") as mock_settings:
        mock_config = MagicMock(spec=Settings)
        mock_config.ROUTER_MODE = "hybrid"
        mock_settings.return_value = mock_config
        
        res = dispatch_route(features, prompt=prompt)
        assert res["routing_method"] in {"Hybrid", "Heuristic Fallback"}
        assert "selected_provider" in res
