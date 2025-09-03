"""
Vector Store - Qdrant-based Vector Database Operations

This module manages all vector database operations using Qdrant for semantic
search and caching. It handles collection management, document storage,
similarity search, and provider analytics with proper indexing and metadata.

Key Features:
- Automatic collection creation with proper schemas
- AI provider categorization and metadata tracking
- Efficient similarity search with filtering capabilities
- Provider usage statistics and analytics
- Comprehensive error handling and logging
- Optimized indexing for fast retrieval

The vector store serves as the foundation for semantic caching and RAG
functionality, enabling the system to find and retrieve similar past
interactions based on vector similarity.

Dependencies:
- qdrant-client: Official Qdrant vector database client
- config.setting: Configuration management for database connection
- uuid: Unique identifier generation for documents
- datetime: Timestamp management for interactions
- typing: Type hints for better code documentation
- logging: Structured logging for debugging and monitoring

Author: AI CLI Tool Team
License: MIT
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models
from config.setting import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, EMBED_DIM
from config.logger import get_logger, log_vector_operation
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime

logger = get_logger(__name__)

_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def ensure_collection():
    """Ensure collection exists with proper schema for AI provider categorization"""
    existing = [c.name for c in _client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        _client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=EMBED_DIM,
                distance=models.Distance.COSINE,
            ),
        )
        
        # Create payload indexes for efficient filtering
        _client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="ai_provider",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        
        _client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="task_category",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        
        _client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="timestamp",
            field_schema=models.PayloadSchemaType.DATETIME,
        )
        
        log_vector_operation("CREATE_COLLECTION", QDRANT_COLLECTION, True, "with AI provider categorization indexes")
        logger.info(f"Created collection {QDRANT_COLLECTION} with AI provider categorization indexes")


def upsert_interaction(
    vec: List[float], 
    payload: Dict[str, Any], 
    point_id: Optional[str] = None,
    ai_provider: Optional[str] = None,
    task_category: Optional[str] = None
) -> str:
    """Store interaction with AI provider categorization metadata"""
    ensure_collection()
    
    # Generate ID if not provided
    if not point_id:
        point_id = str(uuid.uuid4())
    
    # Enhance payload with categorization metadata
    enhanced_payload = {
        **payload,
        "timestamp": datetime.now().isoformat(),
        "ai_provider": ai_provider,
        "task_category": task_category,
        "point_id": point_id,
    }
    
    _client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[models.PointStruct(id=point_id, vector=vec, payload=enhanced_payload)],
    )
    
    log_vector_operation("UPSERT", QDRANT_COLLECTION, True, f"interaction {point_id}")
    logger.info(f"Stored interaction {point_id} with provider: {ai_provider}, category: {task_category}")
    return point_id


def search_similar(
    vec: List[float], 
    top_k: int = 5, 
    score_threshold: Optional[float] = None,
    filter_by_provider: Optional[str] = None,
    filter_by_category: Optional[str] = None
) -> List[Any]:
    """Search similar interactions with optional filtering by AI provider or task category"""
    ensure_collection()
    
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
    query_filter = None
    if must_conditions:
        query_filter = models.Filter(must=must_conditions)
    
    res = _client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=vec,
        limit=top_k,
        with_payload=True,
        score_threshold=score_threshold,
        query_filter=query_filter,
    )
    
    return res


def get_provider_statistics() -> Dict[str, Any]:
    """Get statistics about AI provider usage and performance"""
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