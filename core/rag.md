# core/rag.py - RAG Context Building Documentation

## Overview

`core/rag.py` implements Retrieval-Augmented Generation (RAG) functionality for the AI CLI Tool. It builds contextual information from similar past interactions retrieved from the vector database, enhancing AI responses with relevant background knowledge.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Vector Search  │────│  Context Builder │────│  Enhanced AI    │
│  Similar Items  │    │  (RAG System)    │    │  Response       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Similarity     │    │  Metadata        │    │  Context        │
│  Scoring        │    │  Enhancement     │    │  Formatting     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Functions

### `build_context()` - Main Context Builder

Constructs formatted context from vector search results with metadata enhancement.

```python
def build_context(
    hits: List[Any], 
    max_chars: int = 2000, 
    include_metadata: bool = True
) -> str
```

**Parameters:**
- `hits`: List of search results from vector database
- `max_chars`: Maximum character limit for the built context (default: 2000)
- `include_metadata`: Whether to include provider and similarity metadata (default: True)

**Returns:** Formatted context string ready for AI consumption

**Features:**
- **Smart Truncation**: Respects character limits while preserving complete entries
- **Metadata Enhancement**: Includes provider information and similarity scores
- **Structured Formatting**: Clean, readable format with separators
- **Performance Logging**: Tracks context building statistics

**Example Output:**
```
[anthropic/claude-3-5-sonnet-20241022/reasoning - similarity: 0.892]
Q: How do neural networks learn?
A: Neural networks learn through backpropagation, adjusting weights based on errors...

---

[openai/gpt-4o/coding - similarity: 0.845]
Q: Implement a simple neural network
A: Here's a basic neural network implementation in Python...
```

**Implementation Details:**
```python
def build_context(hits: List[Any], max_chars: int = 2000, include_metadata: bool = True) -> str:
    if not hits:
        return ""
    
    parts = []
    total_chars = 0
    
    for i, hit in enumerate(hits):
        payload = hit.payload or {}
        question = payload.get("user_input", "").strip()
        answer = payload.get("ai_response", "").strip()
        
        if not question or not answer:
            continue
        
        # Build context entry with metadata if requested
        context_entry = ""
        
        if include_metadata:
            provider = payload.get("ai_provider", "unknown")
            category = payload.get("task_category", "unknown")
            score = getattr(hit, 'score', 0)
            context_entry = f"[{provider}/{category} - similarity: {score:.3f}]\n"
        
        context_entry += f"Q: {question}\nA: {answer}"
        
        # Check if adding this entry would exceed the limit
        entry_chars = len(context_entry)
        if total_chars + entry_chars > max_chars and parts:
            break
        
        parts.append(context_entry)
        total_chars += entry_chars
        
        # Limit number of context entries for performance
        if len(parts) >= 5:
            break
    
    context = "\n\n---\n\n".join(parts)
    
    if context:
        logger.info(f"Built RAG context from {len(parts)} similar interactions ({total_chars} chars)")
    
    return context
```

### `filter_relevant_context()` - Context Relevance Filtering

Filters and ranks context entries by relevance to the current query using word overlap analysis.

```python
def filter_relevant_context(
    context: str, 
    query: str, 
    max_relevance: int = 3
) -> str
```

**Parameters:**
- `context`: Existing context string to filter
- `query`: Current user query for relevance comparison
- `max_relevance`: Maximum number of relevant entries to keep (default: 3)

**Returns:** Filtered context with only the most relevant entries

**Relevance Algorithm:**
1. **Tokenization**: Split query and context into words (lowercase)
2. **Overlap Calculation**: Count shared words between query and each context entry
3. **Score Normalization**: Calculate relevance as overlap/query_words
4. **Threshold Filtering**: Keep only entries with relevance score > 0.1
5. **Top-K Selection**: Return the most relevant entries up to max_relevance limit

**Example:**
```python
original_context = """
[provider1] Q: How do neural networks work? A: They use layers...
[provider2] Q: What is cooking? A: Cooking is food preparation...
[provider3] Q: Explain backpropagation A: Backpropagation updates weights...
"""

query = "How does machine learning training work?"
filtered = filter_relevant_context(original_context, query, max_relevance=2)

# Result: Only neural network and backpropagation entries (cooking filtered out)
```

**Implementation:**
```python
def filter_relevant_context(context: str, query: str, max_relevance: int = 3) -> str:
    if not context or not query:
        return context
    
    query_words = set(query.lower().split())
    context_parts = context.split("\n\n---\n\n")
    
    # Score each context part by word overlap
    scored_parts = []
    for part in context_parts:
        part_words = set(part.lower().split())
        overlap = len(query_words.intersection(part_words))
        relevance_score = overlap / len(query_words) if query_words else 0
        scored_parts.append((relevance_score, part))
    
    # Sort by relevance and take top entries
    scored_parts.sort(key=lambda x: x[0], reverse=True)
    relevant_parts = [part for score, part in scored_parts[:max_relevance] if score > 0.1]
    
    return "\n\n---\n\n".join(relevant_parts)
```

## Context Building Strategies

### 1. Similarity-Based Selection

Context entries are selected based on vector similarity scores from the database:

```python
# High similarity (>0.85) indicates very relevant past interactions
# Medium similarity (0.7-0.85) provides related context
# Low similarity (<0.7) may still offer useful background
```

### 2. Metadata Enhancement

Each context entry includes rich metadata:

```python
context_entry = f"[{provider}/{category} - similarity: {score:.3f}]\n"
# Example: [anthropic/claude-3-5-sonnet-20241022/coding - similarity: 0.892]
```

**Metadata Benefits:**
- **Provider Tracking**: Shows which AI generated the response
- **Category Context**: Indicates the type of task (reasoning/coding/general)
- **Similarity Score**: Helps assess relevance confidence
- **Quality Indicators**: Users can see high-quality responses

### 3. Token-Aware Limiting

Context building respects token limits for AI models:

```python
# Character limits approximate token usage
max_chars = 2000  # ~500 tokens for most models
max_chars = 4000  # ~1000 tokens for longer context needs
```

### 4. Entry Prioritization

Context entries are prioritized by:
1. **Similarity Score**: Higher similarity = higher priority
2. **Recency**: More recent interactions may be prioritized
3. **Quality**: Responses from better-performing providers
4. **Completeness**: Entries with both questions and answers

## Integration Examples

### Basic RAG Workflow

```python
from core.embeddings import embed_text
from core.vectorstore import search_similar
from core.rag import build_context
from core.ai import AIClient

# 1. Embed user query
query = "How do I optimize database performance?"
query_embedding = embed_text(query, "sentence-transformers/all-MiniLM-L6-v2")

# 2. Search for similar past interactions
similar_hits = search_similar(
    vec=query_embedding,
    top_k=5,
    score_threshold=0.7
)

# 3. Build context from similar interactions
context = build_context(similar_hits, max_chars=2000)

# 4. Send enhanced request to AI
ai_client = AIClient()
response = ai_client.chat(
    prompt=query,
    context=context,
    category="coding"
)

print(f"Context used: {len(context)} characters")
print(f"AI Response: {response}")
```

### Advanced RAG with Filtering

```python
from core.rag import build_context, filter_relevant_context

# Build initial context
raw_context = build_context(similar_hits, max_chars=4000)

# Filter for relevance
query = "Explain neural network backpropagation"
filtered_context = filter_relevant_context(
    context=raw_context,
    query=query,
    max_relevance=3
)

# Use filtered context for AI request
response = ai_client.chat(
    prompt=query,
    context=filtered_context
)
```

### Provider-Specific Context

```python
from core.vectorstore import search_similar

# Search for context from specific provider
provider_hits = search_similar(
    vec=query_embedding,
    top_k=3,
    filter_by_provider="anthropic/claude-3-5-sonnet-20241022"
)

# Build context from high-quality provider responses
context = build_context(provider_hits, include_metadata=True)
```

## Performance Optimization

### Character Limit Tuning

```python
# Adjust character limits based on model capabilities
MODEL_CONTEXT_LIMITS = {
    "gpt-4o": 4000,           # High-capacity model
    "claude-3-5-haiku": 2000, # Efficient model
    "gemini-pro": 3000,       # Medium capacity
}

max_chars = MODEL_CONTEXT_LIMITS.get(model_name, 2000)
context = build_context(hits, max_chars=max_chars)
```

### Context Entry Limiting

```python
# Limit number of entries to prevent information overload
if len(parts) >= 5:  # Maximum 5 context entries
    break
```

### Relevance Threshold Optimization

```python
# Adjust relevance threshold based on query complexity
def adaptive_threshold(query: str) -> float:
    word_count = len(query.split())
    if word_count > 10:  # Complex queries
        return 0.05  # Lower threshold for more context
    else:  # Simple queries
        return 0.15  # Higher threshold for precision
```

## Quality Assessment

### Context Quality Metrics

```python
def assess_context_quality(context: str, query: str) -> dict:
    """Assess the quality of built context"""
    if not context:
        return {"quality": "empty", "score": 0}
    
    parts = context.split("\n\n---\n\n")
    
    metrics = {
        "entry_count": len(parts),
        "total_chars": len(context),
        "avg_entry_length": len(context) / len(parts),
        "has_metadata": "[" in context,
        "relevance_scores": []
    }
    
    # Extract similarity scores if available
    for part in parts:
        if "similarity:" in part:
            try:
                score_start = part.find("similarity: ") + 12
                score_end = part.find("]", score_start)
                score = float(part[score_start:score_end])
                metrics["relevance_scores"].append(score)
            except:
                pass
    
    if metrics["relevance_scores"]:
        metrics["avg_similarity"] = sum(metrics["relevance_scores"]) / len(metrics["relevance_scores"])
        metrics["min_similarity"] = min(metrics["relevance_scores"])
        metrics["max_similarity"] = max(metrics["relevance_scores"])
    
    return metrics
```

### Context Validation

```python
def validate_context(context: str) -> bool:
    """Validate context format and content"""
    if not context:
        return True  # Empty context is valid
    
    # Check for proper Q&A format
    if "Q:" not in context or "A:" not in context:
        logger.warning("Context missing Q&A format")
        return False
    
    # Check for reasonable length
    if len(context) > 10000:  # Very long context
        logger.warning(f"Context very long: {len(context)} chars")
        return False
    
    # Check for proper separators
    if "\n\n---\n\n" not in context and len(context.split("Q:")) > 2:
        logger.warning("Context missing proper separators")
        return False
    
    return True
```

## Testing and Debugging

### Unit Tests

```python
import pytest
from core.rag import build_context, filter_relevant_context

class MockHit:
    def __init__(self, payload, score=0.8):
        self.payload = payload
        self.score = score

def test_build_context():
    hits = [
        MockHit({
            "user_input": "What is AI?",
            "ai_response": "AI is artificial intelligence...",
            "ai_provider": "openai/gpt-4o",
            "task_category": "reasoning"
        }, score=0.9),
        MockHit({
            "user_input": "How to code?",
            "ai_response": "Start with basics...",
            "ai_provider": "anthropic/claude",
            "task_category": "coding"
        }, score=0.8)
    ]
    
    context = build_context(hits, max_chars=1000, include_metadata=True)
    
    assert "Q: What is AI?" in context
    assert "A: AI is artificial intelligence" in context
    assert "[openai/gpt-4o/reasoning - similarity: 0.900]" in context
    assert "\n\n---\n\n" in context

def test_filter_relevant_context():
    context = """[provider1] Q: Python programming A: Use variables...
    
    ---
    
    [provider2] Q: Cooking recipes A: Mix ingredients..."""
    
    query = "How to program in Python?"
    filtered = filter_relevant_context(context, query, max_relevance=1)
    
    assert "Python programming" in filtered
    assert "Cooking recipes" not in filtered
```

### Integration Tests

```python
def test_end_to_end_rag():
    from core.embeddings import embed_text
    from core.vectorstore import search_similar, upsert_interaction
    from core.rag import build_context
    
    # Store test interaction
    embedding = embed_text("Test query", "sentence-transformers/all-MiniLM-L6-v2")
    upsert_interaction(
        vec=embedding,
        payload={
            "user_input": "Test query",
            "ai_response": "Test response"
        },
        ai_provider="test/model",
        task_category="general"
    )
    
    # Search and build context
    search_embedding = embed_text("Similar test query", "sentence-transformers/all-MiniLM-L6-v2")
    hits = search_similar(search_embedding, top_k=1)
    context = build_context(hits)
    
    assert "Test query" in context
    assert "Test response" in context
```

### Debug Logging

Enable debug logging to troubleshoot RAG context building:

```python
import logging
logging.getLogger('core.rag').setLevel(logging.DEBUG)

# This will show detailed context building information
context = build_context(hits, max_chars=2000)
```

## Troubleshooting

### Common Issues

1. **Empty Context**
   - Check if vector search returns results
   - Verify similarity thresholds aren't too high
   - Ensure database contains relevant interactions

2. **Context Too Long**
   - Reduce `max_chars` parameter
   - Limit number of context entries
   - Use relevance filtering

3. **Poor Relevance**
   - Adjust relevance thresholds
   - Improve embedding model quality
   - Use category-specific filtering

4. **Missing Metadata**
   - Ensure interactions stored with provider information
   - Check `include_metadata=True` parameter
   - Verify payload structure in database

### Performance Issues

```python
# Profile context building performance
import time

start_time = time.time()
context = build_context(hits, max_chars=2000)
build_time = time.time() - start_time

print(f"Context building took {build_time:.3f}s")
print(f"Context length: {len(context)} characters")
print(f"Entries included: {len(context.split('---'))}")
```