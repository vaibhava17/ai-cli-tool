# core/embeddings.py - Text Embedding Documentation

## Overview

`core/embeddings.py` provides text embedding functionality using sentence-transformers models. It converts text queries into high-dimensional vectors that enable semantic similarity matching for the caching and RAG systems.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Text Input    │────│  Sentence        │────│   Vector        │
│   (Queries)     │    │  Transformers    │    │  Embeddings     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Preprocessing  │    │  Model Caching   │    │  Normalization  │
│  & Validation   │    │  (LRU Cache)     │    │  & Formatting   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Functions

### `_get_model()` - Model Loading and Caching

Loads and caches sentence-transformer models with LRU caching for performance.

```python
@lru_cache(maxsize=2)
def _get_model(model_name: str) -> SentenceTransformer
```

**Features:**
- **LRU Caching**: Caches up to 2 models in memory
- **Lazy Loading**: Models loaded only when first requested
- **Error Handling**: Comprehensive error logging and propagation
- **Memory Management**: Automatic cleanup of unused models

**Supported Models:**
- `sentence-transformers/all-MiniLM-L6-v2`: Fast, lightweight (384 dims)
- `sentence-transformers/all-mpnet-base-v2`: Higher quality (768 dims)
- `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`: QA optimized (384 dims)
- Custom models: Any HuggingFace sentence-transformer model

**Usage:**
```python
model = _get_model("sentence-transformers/all-MiniLM-L6-v2")
print(f"Model loaded: {model.get_sentence_embedding_dimension()} dimensions")
```

### `embed_texts()` - Batch Text Embedding

Generates embeddings for multiple texts efficiently using batch processing.

```python
def embed_texts(texts: List[str], model_name: str) -> List[List[float]]
```

**Parameters:**
- `texts`: List of strings to embed
- `model_name`: Name of the sentence-transformer model to use

**Returns:** List of embedding vectors (each vector is List[float])

**Features:**
- **Batch Processing**: Efficiently processes multiple texts together
- **Normalization**: Automatic L2 normalization for cosine similarity
- **Error Handling**: Graceful handling of empty inputs and model errors
- **Progress Tracking**: Optional progress bar for large batches (disabled by default)
- **Memory Optimization**: Automatic conversion to Python lists

**Example:**
```python
texts = [
    "What is machine learning?",
    "How does neural network work?",
    "Explain deep learning concepts"
]
embeddings = embed_texts(texts, "sentence-transformers/all-MiniLM-L6-v2")
print(f"Generated {len(embeddings)} embeddings, each with {len(embeddings[0])} dimensions")
```

### `embed_text()` - Single Text Embedding

Convenience function for embedding a single text string.

```python
def embed_text(text: str, model_name: str) -> List[float]
```

**Parameters:**
- `text`: Single string to embed
- `model_name`: Name of the sentence-transformer model to use

**Returns:** Single embedding vector as List[float]

**Implementation:**
```python
def embed_text(text: str, model_name: str) -> List[float]:
    if not text.strip():
        logger.warning("Attempting to embed empty text")
        return []
    
    return embed_texts([text], model_name)[0]
```

**Example:**
```python
query = "What is artificial intelligence?"
embedding = embed_text(query, "sentence-transformers/all-MiniLM-L6-v2")
print(f"Query embedded to {len(embedding)} dimensions")
```

### `get_embedding_dimension()` - Dimension Detection

Gets the embedding dimension for a specific model without generating embeddings.

```python
def get_embedding_dimension(model_name: str) -> int
```

**Parameters:**
- `model_name`: Name of the sentence-transformer model

**Returns:** Embedding dimension as integer

**Use Cases:**
- Vector database configuration
- Dimension validation
- Model comparison and selection

**Example:**
```python
dim = get_embedding_dimension("sentence-transformers/all-MiniLM-L6-v2")
print(f"Model produces {dim}-dimensional embeddings")  # Output: 384
```

## Model Selection Guide

### Performance vs Quality Trade-offs

| Model | Dimensions | Speed | Quality | Use Case |
|-------|------------|--------|---------|----------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | General purpose, quick retrieval |
| all-mpnet-base-v2 | 768 | Medium | Excellent | High-quality semantic search |
| multi-qa-MiniLM-L6-cos-v1 | 384 | Fast | Good | Question-answering optimized |
| all-MiniLM-L12-v2 | 384 | Medium | Better | Balance of speed and quality |

### Model Selection Criteria

**Choose `all-MiniLM-L6-v2` for:**
- Real-time applications requiring fast embedding generation
- Large-scale deployments with memory constraints
- General semantic similarity tasks

**Choose `all-mpnet-base-v2` for:**
- High-precision semantic search applications
- Quality-critical applications
- When computational resources are abundant

**Choose `multi-qa-MiniLM-L6-cos-v1` for:**
- Question-answering systems
- FAQ matching and retrieval
- Customer support applications

## Performance Optimization

### Caching Strategy

```python
@lru_cache(maxsize=2)  # Cache up to 2 models
def _get_model(model_name: str) -> SentenceTransformer:
```

**Benefits:**
- **Memory Efficiency**: Avoids reloading models for repeated use
- **Startup Time**: Faster subsequent embeddings after initial load
- **Resource Management**: Automatic cleanup of unused models

### Batch Processing

```python
# Efficient batch processing
embeddings = model.encode(
    texts, 
    normalize_embeddings=True,
    show_progress_bar=False,
    convert_to_numpy=True
)
```

**Optimizations:**
- **Normalize Embeddings**: Direct L2 normalization during encoding
- **Disable Progress Bar**: Reduces overhead for programmatic use
- **NumPy Conversion**: Efficient numerical operations

### Memory Management

```python
# Ensure consistent output format
if isinstance(embeddings, np.ndarray):
    embeddings = embeddings.tolist()
```

**Considerations:**
- **Consistent Types**: Always return Python lists for JSON serialization
- **Memory Usage**: Convert NumPy arrays to lists to reduce memory footprint
- **Garbage Collection**: Automatic cleanup of intermediate arrays

## Error Handling

### Model Loading Errors

```python
try:
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info(f"Successfully loaded {model_name}")
    return model
except Exception as e:
    logger.error(f"Failed to load embedding model {model_name}: {e}")
    raise
```

**Common Issues:**
- **Network Connectivity**: Model download failures
- **Disk Space**: Insufficient space for model files
- **Invalid Model Names**: Typos or non-existent models
- **Version Conflicts**: Incompatible sentence-transformers versions

### Embedding Generation Errors

```python
try:
    model = _get_model(model_name)
    embeddings = model.encode(
        texts, 
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    logger.debug(f"Generated embeddings for {len(texts)} texts")
    return embeddings
except Exception as e:
    logger.error(f"Error generating embeddings: {e}")
    raise
```

**Error Recovery:**
- **Retry Logic**: Automatic retries for transient errors
- **Fallback Models**: Switch to alternative models on failure
- **Graceful Degradation**: Return empty embeddings with warnings

## Integration Examples

### Vector Database Integration

```python
from core.embeddings import embed_text
from core.vectorstore import upsert_interaction

# Embed query for vector storage
query = "How does machine learning work?"
embedding = embed_text(query, "sentence-transformers/all-MiniLM-L6-v2")

# Store in vector database
upsert_interaction(
    vec=embedding,
    payload={"user_input": query, "ai_response": "..."},
    ai_provider="openai/gpt-4o",
    task_category="reasoning"
)
```

### Semantic Search

```python
from core.embeddings import embed_text
from core.vectorstore import search_similar

# Search for similar queries
query = "Explain neural networks"
query_embedding = embed_text(query, "sentence-transformers/all-MiniLM-L6-v2")

# Find similar interactions
similar = search_similar(
    vec=query_embedding,
    top_k=5,
    score_threshold=0.8
)
```

### Batch Processing for RAG

```python
from core.embeddings import embed_texts

# Process multiple context documents
documents = [
    "Machine learning is a subset of AI...",
    "Neural networks are computing systems...",
    "Deep learning uses multiple layers..."
]

# Generate embeddings for all documents
doc_embeddings = embed_texts(documents, "sentence-transformers/all-MiniLM-L6-v2")

# Use embeddings for context ranking
for i, embedding in enumerate(doc_embeddings):
    print(f"Document {i}: {len(embedding)} dimensions")
```

## Testing and Validation

### Unit Tests

```python
import pytest
from core.embeddings import embed_text, embed_texts, get_embedding_dimension

def test_single_embedding():
    text = "Test text for embedding"
    embedding = embed_text(text, "sentence-transformers/all-MiniLM-L6-v2")
    assert isinstance(embedding, list)
    assert len(embedding) == 384
    assert all(isinstance(x, float) for x in embedding)

def test_batch_embeddings():
    texts = ["Text 1", "Text 2", "Text 3"]
    embeddings = embed_texts(texts, "sentence-transformers/all-MiniLM-L6-v2")
    assert len(embeddings) == 3
    assert all(len(emb) == 384 for emb in embeddings)

def test_dimension_detection():
    dim = get_embedding_dimension("sentence-transformers/all-MiniLM-L6-v2")
    assert dim == 384
```

### Integration Tests

```python
def test_embedding_similarity():
    # Test semantic similarity
    text1 = "Machine learning algorithms"
    text2 = "ML models and techniques"
    text3 = "Cooking recipes and food"
    
    emb1 = embed_text(text1, "sentence-transformers/all-MiniLM-L6-v2")
    emb2 = embed_text(text2, "sentence-transformers/all-MiniLM-L6-v2")
    emb3 = embed_text(text3, "sentence-transformers/all-MiniLM-L6-v2")
    
    # Calculate cosine similarity
    import numpy as np
    sim_1_2 = np.dot(emb1, emb2)  # Should be high (similar topics)
    sim_1_3 = np.dot(emb1, emb3)  # Should be low (different topics)
    
    assert sim_1_2 > sim_1_3
```

### Performance Tests

```python
import time
from core.embeddings import embed_texts

def test_batch_performance():
    # Test batch vs individual processing
    texts = ["Test text"] * 100
    
    # Batch processing
    start_time = time.time()
    batch_embeddings = embed_texts(texts, "sentence-transformers/all-MiniLM-L6-v2")
    batch_time = time.time() - start_time
    
    # Individual processing
    start_time = time.time()
    individual_embeddings = [
        embed_text(text, "sentence-transformers/all-MiniLM-L6-v2") 
        for text in texts
    ]
    individual_time = time.time() - start_time
    
    # Batch should be significantly faster
    assert batch_time < individual_time * 0.5
```

## Troubleshooting

### Common Issues

1. **Model Download Failures**
   ```python
   # Check internet connectivity and disk space
   # Manually download models if needed
   from sentence_transformers import SentenceTransformer
   SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
   ```

2. **Memory Issues with Large Models**
   ```python
   # Use smaller models for memory-constrained environments
   model_name = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims, ~23MB
   # Instead of "all-mpnet-base-v2"  # 768 dims, ~420MB
   ```

3. **Slow Embedding Generation**
   ```python
   # Use batch processing for multiple texts
   embeddings = embed_texts(texts, model_name)  # Efficient
   # Instead of: [embed_text(t, model_name) for t in texts]  # Slow
   ```

4. **Dimension Mismatches**
   ```python
   # Always check dimensions before storing in vector DB
   expected_dim = get_embedding_dimension(model_name)
   actual_dim = len(embedding)
   assert expected_dim == actual_dim
   ```

### Debug Logging

Enable debug logging to troubleshoot embedding issues:

```python
import logging
logging.getLogger('core.embeddings').setLevel(logging.DEBUG)

# This will show detailed embedding generation logs
embedding = embed_text("test", "sentence-transformers/all-MiniLM-L6-v2")
```

### Performance Monitoring

```python
import time
from core.embeddings import embed_texts

def monitor_embedding_performance(texts, model_name):
    start_time = time.time()
    embeddings = embed_texts(texts, model_name)
    end_time = time.time()
    
    print(f"Embedded {len(texts)} texts in {end_time - start_time:.2f}s")
    print(f"Average: {(end_time - start_time) / len(texts):.4f}s per text")
    print(f"Throughput: {len(texts) / (end_time - start_time):.1f} texts/second")
```