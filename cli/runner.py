"""
CLI Runner - Command Line Interface for AI CLI Tool

This module implements the main CLI interface using Typer. It provides commands for:
- ask: Interactive AI conversations with provider selection
- stats: Usage statistics and analytics  
- configure: Configuration management
- providers: List available AI providers and models

The module handles:
- Command-line argument parsing and validation
- AI provider orchestration and routing
- Semantic caching and RAG context building
- File and image attachment processing
- Error handling and user feedback

Key Features:
- Multi-provider support with intelligent routing
- Semantic caching for performance optimization
- RAG-enhanced responses using vector similarity
- Flexible provider/model overrides via CLI arguments
- Comprehensive usage analytics and monitoring

Dependencies:
- typer: CLI framework with type hints and auto-completion
- pathlib: Cross-platform file path handling
- logging: Structured logging for debugging and monitoring

Author: AI CLI Tool Team
License: MIT
"""

import typer
import json
import base64
from pathlib import Path
from config.setting import EMBED_MODEL
from config.logger import (
    get_logger, 
    print_response_source, 
    log_cache_search, 
    log_rag_context,
    log_provider_selection
)
from core.embeddings import embed_text
from core.vectorstore import search_similar, upsert_interaction, get_provider_statistics
from core.rag import build_context, filter_relevant_context
from core.ai import AIClient

# Get standardized logger for this module
logger = get_logger(__name__)

# Initialize Typer app with help text and configuration
app = typer.Typer(
    help="AI CLI Tool with multi-provider support, semantic caching, and RAG enhancement for intelligent conversations",
    add_completion=True,
    rich_markup_mode="rich"
)

@app.command()
def ask(
    query: str = typer.Argument(..., help="Your prompt"),
    image: Path = typer.Option(None, "--image", "-i", help="Path to image"),
    file: Path = typer.Option(None, "--file", "-f", help="Path to document"),
    category: str = typer.Option(None, "--category", "-c", help="AI task category: reasoning, coding, general"),
    provider_filter: str = typer.Option(None, "--filter-provider", help="Filter cache by AI provider"),
    use_provider: str = typer.Option(None, "--provider", "-p", help="Force specific AI provider: openai, anthropic, google, gemini, ollama"),
    use_model: str = typer.Option(None, "--model", "-m", help="Force specific model (e.g., gpt-4o, claude-3-5-sonnet-20241022, gemini-pro)"),
    k: int = typer.Option(5, help="Top-K retrieved for context"),
    threshold: float = typer.Option(0.85, help="Similarity threshold for cache hit (0-1)"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Skip semantic cache lookup")
):
    """Ask a question with intelligent AI provider selection, semantic caching, and RAG enhancement"""
    
    try:
        # 1) Embed query
        vec = embed_text(query, EMBED_MODEL)

        # 2) Semantic search with optional provider filtering
        context = None
        cached_response = None
        
        if not no_cache:
            hits = search_similar(
                vec, 
                top_k=k, 
                score_threshold=None,
                filter_by_provider=provider_filter,
                filter_by_category=category
            )

            # Log cache search results
            log_cache_search(query, len(hits), threshold)

            # 3) Check for cache hit
            if hits:
                top = hits[0]
                if top.score and top.score >= threshold:
                    # Cache hit - return cached response
                    payload = top.payload or {}
                    cached_response = payload.get("ai_response", "[Cache hit but no response]")
                    provider_used = payload.get("ai_provider", "unknown")
                    task_cat = payload.get("task_category", "unknown")
                    
                    # Enhanced cache hit display
                    print_response_source("cache", provider_used, task_cat, top.score)
                    print(cached_response)
                    return
                else:
                    # Build RAG context from similar interactions
                    context = build_context(hits, max_chars=2000, include_metadata=True)
                    if context:
                        # Filter context for relevance to current query
                        context = filter_relevant_context(context, query, max_relevance=3)
                        if context:
                            log_rag_context(len(context), len(hits))
                            logger.info(f"Built and filtered RAG context from {len(hits)} similar interactions")
                        else:
                            logger.info("No relevant context found after filtering")

        # 4) Prepare attachments
        image_bytes = file_bytes = None
        if image and image.exists():
            image_bytes = image.read_bytes()
        if file and file.exists():
            file_bytes = file.read_bytes()

        # 5) Call AI with appropriate provider
        ai_client = AIClient()
        answer = ai_client.chat(
            query, 
            context=context, 
            files=[file_bytes] if file_bytes else None, 
            images=[image_bytes] if image_bytes else None,
            category=category,
            override_provider=use_provider,
            override_model=use_model
        )

        # Determine which provider was actually used
        used_category = category or ai_client._categorize_task(query)
        used_config = ai_client._get_provider_config(used_category, use_provider, use_model)
        provider_name = f"{used_config['provider']}/{used_config['model']}"
        
        # Log provider selection reasoning
        selection_reason = "CLI override" if use_provider else f"auto-categorized as {used_category}"
        log_provider_selection(used_category, provider_name, selection_reason)

        # 6) Store interaction in vector DB with provider metadata
        payload = {
            "user_input": query,
            "ai_response": answer,
            "attachments": {
                "image": str(image) if image else None,
                "file": str(file) if file else None,
            },
        }
        
        upsert_interaction(
            vec, 
            payload, 
            ai_provider=provider_name,
            task_category=used_category
        )

        # 7) Enhanced AI response display
        print_response_source("ai", provider_name, used_category, context_used=bool(context))
        print(answer)
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command()
def stats():
    """Show AI provider usage statistics"""
    try:
        stats = get_provider_statistics()
        
        if "error" in stats:
            typer.echo(f"Error getting statistics: {stats['error']}", err=True)
            return
        
        typer.echo(f"📊 AI Provider Statistics")
        typer.echo(f"Total interactions: {stats['total_interactions']}")
        typer.echo(f"Recent activity (24h): {stats['recent_activity']}")
        
        if stats['providers']:
            typer.echo("\n🤖 Usage by Provider:")
            for provider, count in sorted(stats['providers'].items(), key=lambda x: x[1], reverse=True):
                typer.echo(f"  {provider}: {count}")
        
        if stats['categories']:
            typer.echo("\n📋 Usage by Category:")
            for category, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True):
                typer.echo(f"  {category}: {count}")
                
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        typer.echo(f"Error: {str(e)}", err=True)


@app.command()
def providers():
    """List available AI providers and their models"""
    try:
        from config.setting import AVAILABLE_PROVIDERS, get_api_key_for_provider
        
        typer.echo("🤖 Available AI Providers:")
        typer.echo()
        
        for provider, info in AVAILABLE_PROVIDERS.items():
            api_key = get_api_key_for_provider(provider)
            status = "✅ Configured" if api_key else "❌ Missing API Key"
            
            typer.echo(f"📋 {provider.upper()} - {status}")
            typer.echo(f"   API Key Required: {info['api_key_env'] or 'None (Local)'}")
            typer.echo(f"   Available Models:")
            for model in info['models']:
                typer.echo(f"     • {model}")
            typer.echo()
        
        typer.echo("💡 Usage Examples:")
        typer.echo("  python main.py ask 'Hello' --provider openai")
        typer.echo("  python main.py ask 'Hello' --provider google --model gemini-1.5-pro")
        typer.echo("  python main.py ask 'Hello' --provider anthropic")
        
    except Exception as e:
        logger.error(f"Error listing providers: {e}")
        typer.echo(f"Error: {str(e)}", err=True)


@app.command()
def configure():
    """Show current AI provider configuration"""
    try:
        from config.setting import AI_PROVIDERS
        
        typer.echo("🔧 AI Provider Configuration:")
        for category, config in AI_PROVIDERS.items():
            status = "✅ Configured" if config.get("api_key") else "❌ Missing API Key"
            typer.echo(f"  {category}: {config['provider']}/{config['model']} - {status}")
            
        typer.echo("\n💡 To configure providers, set these environment variables:")
        typer.echo("  OPENAI_API_KEY, CLAUDE_API_KEY, GEMINI_API_KEY")
        typer.echo("  REASONING_PROVIDER, CODING_PROVIDER, GENERAL_PROVIDER")
        typer.echo("  REASONING_MODEL, CODING_MODEL, GENERAL_MODEL")
        
    except Exception as e:
        logger.error(f"Error showing configuration: {e}")
        typer.echo(f"Error: {str(e)}", err=True)


if __name__ == "__main__":
    app()