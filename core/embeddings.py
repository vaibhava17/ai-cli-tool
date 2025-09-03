"""
Embeddings - Text Embedding Generation and Management

This module handles text embedding generation using sentence-transformers models.
It provides efficient caching, batch processing, and dimension management for
semantic similarity operations in the AI CLI Tool.

Key Features:
- Efficient text embedding generation using sentence-transformers
- Model caching with LRU cache for performance optimization
- Batch processing support for multiple texts
- Automatic dimension detection and validation
- Error handling and fallback strategies
- Normalized embeddings for cosine similarity

The module serves as the foundation for semantic search and caching functionality,
converting text queries into high-dimensional vectors for similarity matching
in the vector database.

Dependencies:
- sentence-transformers: Pre-trained transformer models for text embeddings
- functools: LRU cache for model caching
- numpy: Numerical operations and array handling
- typing: Type hints for better code documentation
- logging: Structured logging for debugging and monitoring

Author: AI CLI Tool Team
License: MIT
"""

from sentence_transformers import SentenceTransformer
from functools import lru_cache
from typing import List
import numpy as np
from config.logger import get_logger

logger = get_logger(__name__)

@lru_cache(maxsize=2)  # Allow caching multiple models
def _get_model(model_name: str) -> SentenceTransformer:
    """Load and cache embedding model"""
    try:
        logger.info(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        logger.info(f"Successfully loaded {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to load embedding model {model_name}: {e}")
        raise


def embed_texts(texts: List[str], model_name: str) -> List[List[float]]:
    """Embed multiple texts using specified model"""
    if not texts:
        return []
    
    try:
        model = _get_model(model_name)
        embeddings = model.encode(
            texts, 
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Ensure consistent output format
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        
        logger.debug(f"Generated embeddings for {len(texts)} texts")
        return embeddings
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise


def embed_text(text: str, model_name: str) -> List[float]:
    """Embed single text using specified model"""
    if not text.strip():
        logger.warning("Attempting to embed empty text")
        return []
    
    return embed_texts([text], model_name)[0]


def get_embedding_dimension(model_name: str) -> int:
    """Get the dimension of embeddings for a given model"""
    try:
        model = _get_model(model_name)
        return model.get_sentence_embedding_dimension()
    except Exception as e:
        logger.error(f"Error getting embedding dimension for {model_name}: {e}")
        # Return default dimension as fallback
        return 384