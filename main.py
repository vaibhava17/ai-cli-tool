#!/usr/bin/env python3
"""
AI CLI Tool - Multi-provider AI assistant with semantic caching and RAG

This is the main entry point for the AI CLI Tool. It provides a command-line interface
for interacting with multiple AI providers (OpenAI, Anthropic, Google Gemini, Ollama)
with intelligent provider selection, semantic caching, and RAG capabilities.

Features:
- Multi-provider AI support with automatic task categorization
- Semantic caching using vector databases (Qdrant)
- RAG (Retrieval-Augmented Generation) for context-aware responses
- File and image processing capabilities
- Provider usage statistics and monitoring
- Configurable AI providers and models

Usage:
    python main.py ask "Your question here"
    python main.py ask "Debug this code" --provider anthropic --model claude-3-5-sonnet-20241022
    python main.py ask "Explain this image" --image photo.jpg
    python main.py providers  # List available providers
    python main.py stats      # Show usage statistics
    python main.py configure  # Show current configuration

Environment Setup:
    Copy .env.example to .env and configure your API keys:
    - OPENAI_API_KEY
    - CLAUDE_API_KEY  
    - GEMINI_API_KEY

Dependencies:
    - litellm: Multi-provider AI API client
    - qdrant-client: Vector database for semantic caching
    - sentence-transformers: Text embeddings
    - typer: CLI framework
    - python-dotenv: Environment variable management

Author: AI CLI Tool Team
License: MIT
"""

from cli.runner import app

if __name__ == "__main__":
    app()