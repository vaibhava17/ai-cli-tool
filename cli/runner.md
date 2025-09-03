# cli/runner.py - CLI Interface Documentation

## Overview

`cli/runner.py` is the core CLI interface module that implements all user-facing commands using the Typer framework. It serves as the orchestration layer that coordinates AI providers, semantic caching, RAG context building, and user interactions.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CLI Runner    │────│  AI Client       │────│  LiteLLM        │
│   (Typer App)   │    │  (Multi-Provider) │    │  (API Gateway)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Vector Store   │    │  RAG Context     │    │  File/Image     │
│  (Qdrant)       │    │  Builder         │    │  Processing     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Commands

### 1. `ask` - Main AI Interaction Command

The primary command for asking questions to AI providers with intelligent routing and caching.

#### Signature
```python
def ask(
    query: str,                    # The question/prompt (required)
    image: Path = None,           # Path to image file
    file: Path = None,            # Path to document file
    category: str = None,         # Force task category
    provider_filter: str = None,  # Filter cache by provider
    use_provider: str = None,     # Force specific provider
    use_model: str = None,        # Force specific model
    k: int = 5,                   # Top-K for RAG context
    threshold: float = 0.85,      # Cache similarity threshold
    no_cache: bool = False        # Skip cache lookup
)
```

#### Workflow
1. **Query Embedding**: Convert user query to vector using sentence-transformers
2. **Semantic Search**: Search vector database for similar past interactions
3. **Cache Check**: If similarity > threshold, return cached response
4. **RAG Context**: Build context from similar interactions if no cache hit
5. **Provider Selection**: Choose AI provider based on task category or CLI override
6. **API Call**: Send request to selected provider via LiteLLM
7. **Response Storage**: Store interaction in vector database with metadata
8. **Output**: Display response with provider information

#### Examples
```bash
# Basic usage - auto-categorized
python main.py ask "What is quantum computing?"

# Force specific provider
python main.py ask "Write a Python script" --provider anthropic

# Force specific model  
python main.py ask "Analyze this code" --provider openai --model gpt-4o

# Process image
python main.py ask "What's in this image?" --image photo.jpg

# Process document
python main.py ask "Summarize this document" --file report.pdf

# Skip cache for fresh response
python main.py ask "Current time" --no-cache

# Filter cache by provider
python main.py ask "Previous OpenAI responses" --filter-provider "openai/gpt-4o"
```

### 2. `stats` - Usage Analytics

Display comprehensive statistics about AI provider usage and performance.

#### Features
- Total interaction count
- Recent activity (24-hour window)
- Usage breakdown by provider
- Usage breakdown by task category
- Error handling for database issues

#### Output Example
```
📊 AI Provider Statistics
Total interactions: 1,247
Recent activity (24h): 89

🤖 Usage by Provider:
  anthropic/claude-3-5-sonnet-20241022: 456
  openai/gpt-4o: 412
  gemini/gemini-pro: 234
  anthropic/claude-3-5-haiku-20241022: 145

📋 Usage by Category:
  coding: 523
  general: 387
  reasoning: 337
```

### 3. `configure` - Configuration Display

Show current AI provider configuration and setup instructions.

#### Features
- Display configured providers per category
- Show API key status (configured/missing)
- Provide setup instructions
- Environment variable guidance

#### Output Example
```
🔧 AI Provider Configuration:
  reasoning: openai/gpt-4o - ✅ Configured
  coding: anthropic/claude-3-5-sonnet-20241022 - ✅ Configured
  general: anthropic/claude-3-5-haiku-20241022 - ✅ Configured

💡 To configure providers, set these environment variables:
  OPENAI_API_KEY, CLAUDE_API_KEY, GEMINI_API_KEY
  REASONING_PROVIDER, CODING_PROVIDER, GENERAL_PROVIDER
  REASONING_MODEL, CODING_MODEL, GENERAL_MODEL
```

### 4. `providers` - Available Providers List

List all supported AI providers, their models, and configuration status.

#### Features
- Show all supported providers (OpenAI, Anthropic, Gemini, Ollama)
- List available models per provider
- Display API key requirements
- Show configuration status
- Provide usage examples

#### Output Example
```
🤖 Available AI Providers:

✅ openai
   Models: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
   API Key: OPENAI_API_KEY

✅ anthropic
   Models: claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022, claude-3-opus-20240229
   API Key: CLAUDE_API_KEY

❌ gemini
   Models: gemini-pro, gemini-pro-vision, gemini-1.5-pro, gemini-1.5-flash
   API Key: GEMINI_API_KEY

❌ ollama
   Models: llama2, codellama, mistral, phi
   API Key: Not required

💡 Usage examples:
  python main.py ask 'Hello' --provider openai
  python main.py ask 'Hello' --provider gemini --model gemini-pro
  python main.py ask 'Write code' --provider anthropic --model claude-3-5-sonnet-20241022
```

## Key Features

### Intelligent Provider Routing

The system automatically categorizes tasks and routes to appropriate providers:

```python
TASK_CATEGORIES = {
    "reasoning": ["analyze", "explain", "reason", "why", "how", "compare", "evaluate"],
    "coding": ["code", "implement", "fix", "debug", "refactor", "function", "class", "script"],
    "general": []  # fallback category
}
```

### Semantic Caching

Uses vector similarity to avoid redundant API calls:
- Queries are embedded using sentence-transformers
- Similar queries (>85% similarity by default) return cached responses
- Cache can be filtered by provider or category
- Significant cost savings and performance improvements

### RAG Context Enhancement

When no cache hit occurs, similar past interactions provide context:
- Retrieves top-K similar interactions from vector database
- Builds formatted context with provider metadata
- Enhances AI responses with relevant background information
- Improves consistency across interactions

### File Processing

Supports multiple file types as attachments:
- **Images**: JPG, PNG, GIF (base64 encoded for AI APIs)
- **Documents**: PDF, TXT, MD (text extraction and chunking)
- **Size limits**: 2MB per file, 2000 characters for text content
- **Error handling**: Graceful degradation if files can't be processed

### Error Handling

Comprehensive error management:
- **Configuration errors**: Missing API keys, invalid providers
- **Network errors**: API timeouts, rate limits, connection issues
- **File errors**: Invalid paths, unsupported formats, size limits
- **User errors**: Invalid arguments, malformed queries

## Dependencies

### Core Libraries
- **typer**: CLI framework with type hints and auto-completion
- **pathlib**: Cross-platform file path handling
- **logging**: Structured logging for debugging and monitoring
- **json/base64**: Data serialization and encoding

### Integration Modules
- **config.setting**: Environment and provider configuration
- **core.embeddings**: Text embedding generation
- **core.vectorstore**: Vector database operations
- **core.rag**: Context building and relevance filtering
- **core.ai**: AI provider client and routing

## Performance Optimizations

### Parallel Operations
- Multiple tool calls executed concurrently when possible
- Batch embedding generation for multiple texts
- Concurrent vector search and file processing

### Caching Strategies
- **Model caching**: Embedding models cached with `@lru_cache`
- **Vector caching**: Query embeddings cached for session
- **Response caching**: Semantic similarity prevents redundant API calls

### Memory Management
- **Streaming responses**: Large responses streamed to reduce memory usage
- **File chunking**: Large files processed in chunks
- **Context limiting**: RAG context limited to prevent token overflow

## Security Considerations

### API Key Protection
- Keys loaded from environment variables only
- Never logged or exposed in error messages
- Secure transmission to AI providers

### Input Validation
- All user inputs validated before processing
- Path traversal protection for file operations
- Injection attack prevention

### Sandboxed Execution
- No arbitrary code execution from user input
- Limited file system access
- Safe image and document processing

## Monitoring and Logging

### Structured Logging
```python
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Key Log Events
- Provider selection and usage
- Cache hits and misses
- Error conditions and recovery
- Performance metrics and timings

### Usage Analytics
- Provider usage statistics
- Task category distribution
- Response time metrics
- Error rate monitoring

## Extending the CLI

### Adding New Commands

```python
@app.command()
def new_command(
    param: str = typer.Argument(..., help="Parameter description"),
    option: bool = typer.Option(False, "--flag", help="Option description")
):
    """Command description for help text"""
    try:
        # Implementation here
        typer.echo("Success message")
    except Exception as e:
        logger.error(f"Error in new_command: {e}")
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(1)
```

### Adding New Parameters

Follow the established patterns:
- Use type hints for all parameters
- Provide helpful descriptions
- Include examples in docstrings
- Add proper error handling
- Update documentation

### Testing Commands

```bash
# Test basic functionality
python main.py ask "test query"

# Test with files
python main.py ask "test" --image test.jpg --file test.txt

# Test provider overrides
python main.py ask "test" --provider openai --model gpt-4o

# Test error conditions
python main.py ask "" --provider invalid_provider
```