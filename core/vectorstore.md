# core/vectorstore.py - Vector Database Operations Documentation

## Overview

`core/vectorstore.py` manages all vector database operations using Qdrant. It provides the core functionality for semantic caching and RAG by storing, retrieving, and analyzing interactions with AI providers based on vector similarity.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Application   │────│   Vector Store   │────│     Qdrant      │
│   (AI Client)   │    │   Operations     │    │    Database     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Text Queries   │    │  Vector Storage  │    │  Similarity     │
│  & Responses    │    │  with Metadata   │    │  Search         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Global Configuration

### Client Initialization

```python
_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
```

The module initializes a global Qdrant client using configuration from `config.setting`:
- **Host**: Default 127.0.0.1 (configurable via `QDRANT_HOST`)
- **Port**: Default 6333 (configurable via `QDRANT_PORT`)
- **Connection**: Persistent connection reused across operations

## Core Functions

### `ensure_collection()` - Collection Management

Creates the vector collection with proper schema and indexing if it doesn't exist.

```python
def ensure_collection() -> None
```

**Collection Configuration:**
- **Vector Config**: Cosine distance, dimensions from `EMBED_DIM`
- **Payload Indexes**: Optimized indexes for filtering by provider, category, and timestamp
- **Auto-creation**: Called automatically by storage and search functions

**Schema Definition:**
```python
_client.create_collection(
    collection_name=QDRANT_COLLECTION,
    vectors_config=models.VectorParams(
        size=EMBED_DIM,           # Default: 384 dimensions
        distance=models.Distance.COSINE,  # Cosine similarity
    ),
)

# Optimized indexes for fast filtering
_client.create_payload_index(
    collection_name=QDRANT_COLLECTION,
    field_name="ai_provider",         # Index by AI provider
    field_schema=models.PayloadSchemaType.KEYWORD,
)
_client.create_payload_index(
    collection_name=QDRANT_COLLECTION,
    field_name="task_category",       # Index by task category
    field_schema=models.PayloadSchemaType.KEYWORD,
)
_client.create_payload_index(
    collection_name=QDRANT_COLLECTION,
    field_name="timestamp",           # Index by timestamp
    field_schema=models.PayloadSchemaType.DATETIME,
)
```

### `upsert_interaction()` - Store Interactions

Stores AI interactions with comprehensive metadata for future retrieval.

```python
def upsert_interaction(
    vec: List[float], 
    payload: Dict[str, Any], 
    point_id: Optional[str] = None,
    ai_provider: Optional[str] = None,
    task_category: Optional[str] = None
) -> str
```

**Parameters:**
- `vec`: Embedding vector for the interaction (384 dimensions by default)
- `payload`: Core interaction data (user_input, ai_response, attachments)
- `point_id`: Optional unique identifier (auto-generated if not provided)
- `ai_provider`: AI provider identifier (e.g., "openai/gpt-4o")
- `task_category`: Task category (reasoning/coding/general)

**Returns:** The point ID of the stored interaction

**Enhanced Payload Structure:**
```python
enhanced_payload = {
    **payload,                    # Original payload
    "timestamp": datetime.now().isoformat(),
    "ai_provider": ai_provider,   # Provider metadata
    "task_category": task_category,
    "point_id": point_id,         # Unique identifier
}
```

**Example Usage:**
```python
from core.embeddings import embed_text
from core.vectorstore import upsert_interaction

# Embed the query
query = "How does machine learning work?"
embedding = embed_text(query, "sentence-transformers/all-MiniLM-L6-v2")

# Store the interaction
point_id = upsert_interaction(
    vec=embedding,
    payload={
        "user_input": query,
        "ai_response": "Machine learning is a method of data analysis...",
        "attachments": {"image": None, "file": None}
    },
    ai_provider="openai/gpt-4o",
    task_category="reasoning"
)

print(f"Stored interaction: {point_id}")
```

### `search_similar()` - Semantic Search

Performs vector similarity search with optional filtering by provider or category.

```python
def search_similar(
    vec: List[float], 
    top_k: int = 5, 
    score_threshold: Optional[float] = None,
    filter_by_provider: Optional[str] = None,
    filter_by_category: Optional[str] = None
) -> List[Any]
```

**Parameters:**
- `vec`: Query vector for similarity search
- `top_k`: Number of similar results to return (default: 5)
- `score_threshold`: Minimum similarity score (0.0-1.0)
- `filter_by_provider`: Filter by specific AI provider
- `filter_by_category`: Filter by task category

**Returns:** List of similar interactions sorted by similarity score

**Filtering Logic:**
```python
# Build filter conditions
must_conditions = []

if filter_by_provider:
    must_conditions.append(
        models.FieldCondition(
            key="ai_provider",
            match=models.MatchValue(value=filter_by_provider)
        )
    )

if filter_by_category:
    must_conditions.append(
        models.FieldCondition(
            key="task_category",
            match=models.MatchValue(value=filter_by_category)
        )
    )

# Create filter if conditions exist
query_filter = models.Filter(must=must_conditions) if must_conditions else None
```

**Example Usage:**
```python
# Basic similarity search
similar_results = search_similar(
    vec=query_embedding,
    top_k=3,
    score_threshold=0.8
)

# Provider-specific search
claude_results = search_similar(
    vec=query_embedding,
    top_k=5,
    filter_by_provider="anthropic/claude-3-5-sonnet-20241022"
)

# Category-specific search
coding_results = search_similar(
    vec=query_embedding,
    top_k=3,
    filter_by_category="coding",
    score_threshold=0.7
)
```

### `get_provider_statistics()` - Analytics

Retrieves comprehensive statistics about AI provider usage and performance.

```python
def get_provider_statistics() -> Dict[str, Any]
```

**Returns:** Dictionary containing detailed usage statistics

**Statistics Structure:**
```python
{
    "total_interactions": 1247,
    "recent_activity": 89,          # Last 24 hours
    "providers": {
        "openai/gpt-4o": 456,
        "anthropic/claude-3-5-sonnet-20241022": 412,
        "gemini/gemini-pro": 234
    },
    "categories": {
        "coding": 523,
        "reasoning": 387,
        "general": 337
    }
}
```

**Implementation Details:**
```python
def get_provider_statistics() -> Dict[str, Any]:
    ensure_collection()
    
    try:
        # Get all points with minimal payload for counting
        scroll_result = _client.scroll(
            collection_name=QDRANT_COLLECTION,
            with_payload=["ai_provider", "task_category", "timestamp"],
            limit=10000  # Adjust based on expected data size
        )
        
        stats = {
            "total_interactions": len(scroll_result[0]),
            "providers": {},
            "categories": {},
            "recent_activity": 0
        }
        
        # Count by provider and category
        for point in scroll_result[0]:
            payload = point.payload or {}
            provider = payload.get("ai_provider", "unknown")
            category = payload.get("task_category", "unknown")
            
            stats["providers"][provider] = stats["providers"].get(provider, 0) + 1
            stats["categories"][category] = stats["categories"].get(category, 0) + 1
            
            # Count recent activity (last 24 hours)
            timestamp_str = payload.get("timestamp")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if (datetime.now() - timestamp).days < 1:
                        stats["recent_activity"] += 1
                except ValueError:
                    pass
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting provider statistics: {e}")
        return {"error": str(e)}
```

## Database Schema

### Collection Configuration

```python
# Vector configuration
vectors_config=models.VectorParams(
    size=384,                    # Embedding dimensions (configurable)
    distance=models.Distance.COSINE,  # Similarity metric
)
```

### Payload Schema

Each stored interaction contains the following payload structure:

```python
{
    # Core interaction data
    "user_input": "User's question or request",
    "ai_response": "AI-generated response",
    "attachments": {
        "image": "path/to/image.jpg",    # Optional image attachment
        "file": "path/to/document.pdf"   # Optional file attachment
    },
    
    # Metadata for categorization and analytics
    "ai_provider": "openai/gpt-4o",           # Provider identifier
    "task_category": "reasoning",              # Task category
    "timestamp": "2024-01-01T12:00:00",       # ISO timestamp
    "point_id": "uuid-string"                 # Unique identifier
}
```

### Indexes

The collection uses optimized indexes for fast filtering:

1. **ai_provider**: Keyword index for provider filtering
2. **task_category**: Keyword index for category filtering  
3. **timestamp**: Datetime index for temporal filtering

## Performance Optimization

### Vector Search Optimization

```python
# Efficient similarity search with proper parameters
res = _client.search(
    collection_name=QDRANT_COLLECTION,
    query_vector=vec,
    limit=top_k,                    # Limit results for performance
    with_payload=True,              # Include metadata
    score_threshold=score_threshold, # Filter by similarity
    query_filter=query_filter,      # Index-optimized filtering
)
```

### Batch Operations

For bulk operations, consider using batch upserts:

```python
# Example batch upsert (not implemented in current version)
points = [
    models.PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding,
        payload=enhanced_payload
    )
    for embedding, payload in zip(embeddings, payloads)
]

_client.upsert(collection_name=QDRANT_COLLECTION, points=points)
```

### Memory Management

```python
# Efficient scroll operations for large datasets
scroll_result = _client.scroll(
    collection_name=QDRANT_COLLECTION,
    with_payload=["ai_provider", "task_category", "timestamp"],
    limit=10000  # Reasonable batch size
)
```

## Error Handling

### Connection Management

```python
try:
    _client.get_collections()
except Exception as e:
    logger.error(f"Qdrant connection failed: {e}")
    raise RuntimeError(f"Vector database unavailable: {e}")
```

### Collection Operations

```python
try:
    existing = [c.name for c in _client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        # Create collection with proper error handling
        _client.create_collection(...)
except Exception as e:
    logger.error(f"Collection creation failed: {e}")
    raise
```

### Search Operations

```python
try:
    res = _client.search(...)
    return res
except Exception as e:
    logger.error(f"Vector search failed: {e}")
    return []  # Return empty results instead of crashing
```

## Integration Examples

### Complete Caching Workflow

```python
from core.embeddings import embed_text
from core.vectorstore import upsert_interaction, search_similar
from core.rag import build_context
from core.ai import AIClient

def cached_ai_interaction(query: str, threshold: float = 0.85):
    # 1. Embed the query
    embedding = embed_text(query, "sentence-transformers/all-MiniLM-L6-v2")
    
    # 2. Search for similar past interactions
    similar = search_similar(embedding, top_k=1, score_threshold=threshold)
    
    if similar and similar[0].score >= threshold:
        # Cache hit - return cached response
        cached_response = similar[0].payload.get("ai_response")
        provider_used = similar[0].payload.get("ai_provider")
        print(f"[CACHED - {provider_used}] {cached_response}")
        return cached_response
    
    # 3. No cache hit - get fresh response
    context = build_context(search_similar(embedding, top_k=5, score_threshold=0.7))
    
    ai_client = AIClient()
    response = ai_client.chat(query, context=context)
    
    # 4. Store the new interaction
    upsert_interaction(
        vec=embedding,
        payload={
            "user_input": query,
            "ai_response": response,
            "attachments": {"image": None, "file": None}
        },
        ai_provider="openai/gpt-4o",
        task_category="general"
    )
    
    return response
```

### Provider Analytics Dashboard

```python
from core.vectorstore import get_provider_statistics
import json

def display_analytics():
    stats = get_provider_statistics()
    
    print("📊 AI CLI Tool Analytics")
    print(f"Total Interactions: {stats['total_interactions']}")
    print(f"Recent Activity (24h): {stats['recent_activity']}")
    
    print("\n🤖 Provider Usage:")
    for provider, count in sorted(stats['providers'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_interactions']) * 100
        print(f"  {provider}: {count} ({percentage:.1f}%)")
    
    print("\n📋 Category Distribution:")
    for category, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_interactions']) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")

# Run analytics
display_analytics()
```

## Testing and Debugging

### Unit Tests

```python
import pytest
from core.vectorstore import ensure_collection, upsert_interaction, search_similar
import numpy as np

def test_collection_creation():
    # Test collection is created properly
    ensure_collection()
    # Should not raise any exceptions

def test_upsert_and_search():
    # Create test embedding
    test_vector = np.random.rand(384).tolist()
    
    # Store test interaction
    point_id = upsert_interaction(
        vec=test_vector,
        payload={
            "user_input": "Test query",
            "ai_response": "Test response"
        },
        ai_provider="test/model",
        task_category="general"
    )
    
    assert isinstance(point_id, str)
    
    # Search for similar interactions
    results = search_similar(test_vector, top_k=1)
    assert len(results) > 0
    assert results[0].payload["user_input"] == "Test query"

def test_provider_filtering():
    # Test provider-specific search
    results = search_similar(
        vec=np.random.rand(384).tolist(),
        filter_by_provider="test/model"
    )
    # Should return results from test provider only
    for result in results:
        assert result.payload.get("ai_provider") == "test/model"
```

### Integration Tests

```python
def test_full_workflow():
    from core.embeddings import embed_text
    
    # Test complete workflow
    query = "Integration test query"
    embedding = embed_text(query, "sentence-transformers/all-MiniLM-L6-v2")
    
    # Store interaction
    point_id = upsert_interaction(
        vec=embedding,
        payload={
            "user_input": query,
            "ai_response": "Integration test response"
        },
        ai_provider="test/integration",
        task_category="testing"
    )
    
    # Search for stored interaction
    results = search_similar(embedding, top_k=1)
    
    assert len(results) > 0
    assert results[0].payload["user_input"] == query
    assert results[0].score > 0.95  # Should be very similar to itself
```

### Debug Tools

```python
def debug_collection_info():
    """Debug tool to inspect collection state"""
    try:
        collections = _client.get_collections().collections
        for collection in collections:
            if collection.name == QDRANT_COLLECTION:
                print(f"Collection: {collection.name}")
                print(f"Vectors count: {collection.vectors_count}")
                print(f"Status: {collection.status}")
                
                # Get sample points
                sample = _client.scroll(
                    collection_name=QDRANT_COLLECTION,
                    limit=3,
                    with_payload=True
                )
                
                print(f"Sample points: {len(sample[0])}")
                for point in sample[0][:3]:
                    print(f"  ID: {point.id}")
                    print(f"  Payload keys: {list(point.payload.keys())}")
                    
    except Exception as e:
        print(f"Debug failed: {e}")

# Run debug info
debug_collection_info()
```

## Troubleshooting

### Common Issues

1. **Connection Failed**
   ```bash
   # Check if Qdrant is running
   curl http://localhost:6333/health
   
   # Start Qdrant with Docker
   docker run -p 6333:6333 qdrant/qdrant
   ```

2. **Collection Already Exists Error**
   ```python
   # The ensure_collection() function handles this automatically
   # Check collection status manually:
   collections = _client.get_collections()
   print([c.name for c in collections.collections])
   ```

3. **Dimension Mismatch**
   ```python
   # Ensure embedding dimensions match collection configuration
   expected_dim = EMBED_DIM  # From config
   actual_dim = len(embedding)
   assert expected_dim == actual_dim, f"Dimension mismatch: {expected_dim} != {actual_dim}"
   ```

4. **Search Returns No Results**
   ```python
   # Check if data exists in collection
   info = _client.get_collection(QDRANT_COLLECTION)
   print(f"Collection has {info.vectors_count} vectors")
   
   # Lower similarity threshold
   results = search_similar(embedding, score_threshold=0.5)
   ```

### Performance Issues

1. **Slow Searches**
   - Ensure proper indexing is enabled
   - Use appropriate `top_k` limits
   - Consider using score thresholds to reduce results

2. **Memory Usage**
   - Use scroll operations for large datasets
   - Limit payload fields in search results
   - Implement pagination for analytics

3. **Storage Growth**
   - Implement data cleanup policies
   - Archive old interactions
   - Monitor disk usage regularly