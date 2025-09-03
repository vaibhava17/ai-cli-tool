"""
Centralized Logging Configuration - Standardized Logger for AI CLI Tool

This module provides a centralized logging configuration that can be imported
across all modules in the AI CLI Tool. It ensures consistent logging format,
levels, and enhanced output for better debugging and user feedback.

Key Features:
- Standardized logging format with timestamps and module names
- Environment-configurable log levels (DEBUG, INFO, WARN, ERROR)
- Enhanced response source indicators for cache vs AI responses
- Structured logging for better debugging and monitoring
- Color-coded console output for better readability

Dependencies:
- logging: Python standard library for logging functionality
- os: Environment variable access for log level configuration

Author: AI CLI Tool Team
License: MIT
"""

import logging
import os
from typing import Optional

# Configure logging level from environment (default: INFO)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Map string levels to logging constants
LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

def setup_logger(name: str) -> logging.Logger:
    """
    Create a standardized logger with consistent formatting
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only add handler if logger doesn't already have one
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler()
        
        # Create formatter with enhanced format
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Set level
        log_level = LEVEL_MAP.get(LOG_LEVEL, logging.INFO)
        logger.setLevel(log_level)
        handler.setLevel(log_level)
    
    return logger

def print_response_source(source_type: str, provider: str, category: str, similarity: float = None, context_used: bool = False):
    """
    Print enhanced response source information with visual indicators
    
    Args:
        source_type: "cache" or "ai"
        provider: AI provider name (e.g., "openai/gpt-4o")
        category: Task category (reasoning, coding, general)
        similarity: Similarity score for cache hits
        context_used: Whether RAG context was used
    """
    logger = setup_logger("response_tracker")
    
    if source_type == "cache":
        cache_quality = "PERFECT" if similarity >= 0.95 else "HIGH" if similarity >= 0.85 else "MEDIUM"
        print(f"\n🔄 CACHE HIT ({cache_quality}) - Similarity: {similarity:.3f}")
        print(f"📦 Original Source: {provider} | Category: {category}")
        print(f"⚡ Response Time: Instant (0 API calls)")
        logger.info(f"Cache hit: {provider}/{category} with similarity {similarity:.3f}")
        
    elif source_type == "ai":
        context_indicator = " + RAG Context" if context_used else ""
        print(f"\n🤖 AI RESPONSE{context_indicator}")
        print(f"🔗 Provider: {provider} | Category: {category}")
        print(f"💰 API Call: Made fresh request to {provider}")
        if context_used:
            print(f"📚 Enhanced: Using context from previous interactions")
        logger.info(f"AI response: {provider}/{category} with RAG: {context_used}")
    
    print("─" * 60)

def log_cache_search(query: str, hits_found: int, threshold: float):
    """Log cache search operations"""
    logger = setup_logger("cache_search")
    if hits_found > 0:
        logger.info(f"Cache search: Found {hits_found} similar interactions for query (threshold: {threshold})")
    else:
        logger.info(f"Cache search: No similar interactions found (threshold: {threshold})")

def log_rag_context(context_length: int, interactions_used: int):
    """Log RAG context building"""
    logger = setup_logger("rag_context")
    logger.info(f"RAG context built: {context_length} characters from {interactions_used} interactions")

def log_vector_operation(operation: str, collection: str, success: bool, details: str = None):
    """Log vector database operations"""
    logger = setup_logger("vector_db")
    status = "SUCCESS" if success else "FAILED"
    message = f"Vector DB {operation} on '{collection}': {status}"
    if details:
        message += f" - {details}"
    
    if success:
        logger.info(message)
    else:
        logger.error(message)

def log_ai_request(provider: str, model: str, category: str, token_usage: Optional[dict] = None):
    """Log AI provider requests"""
    logger = setup_logger("ai_requests")
    message = f"AI request: {provider}/{model} for {category} task"
    if token_usage:
        message += f" (tokens: {token_usage.get('total', 'unknown')})"
    logger.info(message)

def log_provider_selection(query_category: str, selected_provider: str, reason: str):
    """Log provider selection logic"""
    logger = setup_logger("provider_selection")
    logger.info(f"Provider selection: {query_category} -> {selected_provider} ({reason})")

# Global logger instance for general use
main_logger = setup_logger("ai_cli_tool")

# Export commonly used loggers
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the standardized configuration"""
    return setup_logger(name)