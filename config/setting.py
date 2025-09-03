"""
Configuration Settings - Environment and Provider Configuration Management

This module centralizes all configuration management for the AI CLI Tool. It handles:
- Environment variable loading and validation
- AI provider configuration with fallback strategies
- Vector database and embedding settings
- Task categorization rules and keywords
- Storage configuration for file attachments

The module follows a hierarchical configuration approach:
1. Environment variables (highest priority)
2. Default values (fallback)
3. Provider-specific overrides

Key Features:
- Multi-provider AI configuration with automatic fallback
- Dynamic provider discovery and validation
- Flexible task categorization system
- Centralized environment variable management
- Type-safe configuration with validation

Dependencies:
- python-dotenv: Environment variable loading from .env files
- typing: Type hints for better code documentation and IDE support

Author: AI CLI Tool Team
License: MIT
"""

from dotenv import load_dotenv
import os
from typing import Dict, Any

# Load environment variables from .env file if present
load_dotenv()

# Vector Database Settings
QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "ai_cache")

# Embedding Settings
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

# Available Providers and Models
AVAILABLE_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "api_key_env": "OPENAI_API_KEY"
    },
    "anthropic": {
        "models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        "api_key_env": "CLAUDE_API_KEY"
    },
    "google": {
        "models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
        "api_key_env": "GEMINI_API_KEY"
    },
    "ollama": {
        "models": ["llama3", "llama3.1", "codellama", "mistral", "phi3"],
        "api_key_env": None  # Ollama doesn't require API key
    }
}

# AI Provider Configurations
AI_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "reasoning": {
        "provider": os.getenv("REASONING_PROVIDER", "openai"),
        "model": os.getenv("REASONING_MODEL", "gpt-4o"),
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    "coding": {
        "provider": os.getenv("CODING_PROVIDER", "anthropic"),
        "model": os.getenv("CODING_MODEL", "claude-3-5-sonnet-20241022"),
        "api_key": os.getenv("CLAUDE_API_KEY"),
    },
    "general": {
        "provider": os.getenv("GENERAL_PROVIDER", "gemini"),
        "model": os.getenv("GENERAL_MODEL", "gemini-1.5-flash"),
        "api_key": os.getenv("GEMINI_API_KEY"),
    },
}

def get_api_key_for_provider(provider: str) -> str:
    """Get API key for a specific provider"""
    provider_info = AVAILABLE_PROVIDERS.get(provider, {})
    api_key_env = provider_info.get("api_key_env")
    if api_key_env:
        return os.getenv(api_key_env)
    return None

# Storage Settings
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET = os.getenv("MINIO_BUCKET")

# Task categorization keywords
TASK_CATEGORIES = {
    "reasoning": ["analyze", "explain", "reason", "why", "how", "compare", "evaluate"],
    "coding": ["code", "implement", "fix", "debug", "refactor", "function", "class", "script"],
    "general": []  # fallback category
}