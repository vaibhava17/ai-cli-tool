"""
RAG (Retrieval-Augmented Generation) - Context Building and Enhancement

This module implements RAG functionality by building contextual information
from similar past interactions. It enhances AI responses by providing relevant
background context retrieved from the vector database.

Key Features:
- Context building from similar interactions with metadata
- Relevance scoring and filtering based on query similarity
- Formatted context with provider and category information
- Character limit management for optimal token usage
- Intelligent context ranking and selection

The RAG system improves response quality by leveraging past interactions,
creating a knowledge base effect where the AI can reference similar
questions and provide more consistent, informed answers.

Dependencies:
- typing: Type hints for better code documentation
- logging: Structured logging for debugging and monitoring

Author: AI CLI Tool Team
License: MIT
"""

from typing import List, Any, Optional
from config.logger import get_logger

logger = get_logger(__name__)

def build_context(hits: List[Any], max_chars: int = 2000, include_metadata: bool = True) -> str:
    """Build RAG context from similar interactions with enhanced formatting"""
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


def filter_relevant_context(context: str, query: str, max_relevance: int = 3) -> str:
    """Filter context entries by relevance to current query"""
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