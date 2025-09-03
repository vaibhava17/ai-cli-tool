# config/setting.py - Configuration Management Documentation

## Overview

`config/setting.py` serves as the central configuration hub for the AI CLI Tool. It manages environment variables, AI provider configurations, vector database settings, and task categorization rules. The module implements a hierarchical configuration system with environment variables taking precedence over defaults.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Environment    │────│  Configuration   │────│  Application    │
│  Variables      │    │  Validation      │    │  Components     │
│  (.env file)    │    │  & Defaults      │    │  (AI, Vector)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Configuration Categories

### 1. Vector Database Settings

Controls Qdrant vector database connection and collection management:

```python
QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "ai_cache")
```

**Environment Variables:**
- `QDRANT_HOST`: Vector database host address (default: 127.0.0.1)
- `QDRANT_PORT`: Vector database port (default: 6333)
- `QDRANT_COLLECTION`: Collection name for storing interactions (default: ai_cache)

### 2. Embedding Settings

Configures text embedding model and dimensions:

```python
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))
```

**Environment Variables:**
- `EMBED_MODEL`: Sentence transformer model name (default: all-MiniLM-L6-v2)
- `EMBED_DIM`: Embedding vector dimensions (default: 384)

**Supported Models:**
- `all-MiniLM-L6-v2`: Fast, lightweight, 384 dimensions
- `all-mpnet-base-v2`: Higher quality, 768 dimensions
- `multi-qa-MiniLM-L6-cos-v1`: Optimized for question-answering

### 3. AI Provider Configurations

Defines the multi-provider AI system with task-specific routing:

```python
AI_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "reasoning": {
        "provider": os.getenv("REASONING_PROVIDER", "openai"),
        "model": os.getenv("REASONING_MODEL", "gpt-4o"),
        "api_key": os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY"),
        "api_base": os.getenv("OPENAI_API_BASE"),
    },
    "coding": {
        "provider": os.getenv("CODING_PROVIDER", "anthropic"),
        "model": os.getenv("CODING_MODEL", "claude-3-5-sonnet-20241022"),
        "api_key": os.getenv("CLAUDE_API_KEY"),
        "api_base": os.getenv("CLAUDE_API_BASE"),
    },
    "general": {
        "provider": os.getenv("GENERAL_PROVIDER", "anthropic"),
        "model": os.getenv("GENERAL_MODEL", "claude-3-5-haiku-20241022"),
        "api_key": os.getenv("CLAUDE_API_KEY"),
        "api_base": os.getenv("CLAUDE_API_BASE"),
    },
}
```

**Task Categories:**
- **Reasoning**: Analysis, explanations, comparisons → OpenAI GPT-4o by default
- **Coding**: Implementation, debugging, refactoring → Anthropic Claude Sonnet by default
- **General**: Conversations, simple questions → Anthropic Claude Haiku by default

### 4. Available Providers Registry

Comprehensive registry of all supported AI providers and their capabilities:

```python
AVAILABLE_PROVIDERS = {
    "openai": {
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "api_key_env": "OPENAI_API_KEY",
        "api_base_env": "OPENAI_API_BASE"
    },
    "anthropic": {
        "models": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
        "api_key_env": "CLAUDE_API_KEY",
        "api_base_env": "CLAUDE_API_BASE"
    },
    "gemini": {
        "models": ["gemini-pro", "gemini-pro-vision", "gemini-1.5-pro", "gemini-1.5-flash"],
        "api_key_env": "GEMINI_API_KEY",
        "api_base_env": "GEMINI_API_BASE"
    },
    "ollama": {
        "models": ["llama2", "codellama", "mistral", "phi"],
        "api_key_env": None,  # Ollama doesn't require API key
        "api_base_env": "OLLAMA_API_BASE"
    }
}
```

### 5. Task Categorization System

Keyword-based automatic task categorization for intelligent provider routing:

```python
TASK_CATEGORIES = {
    "reasoning": ["analyze", "explain", "reason", "why", "how", "compare", "evaluate"],
    "coding": ["code", "implement", "fix", "debug", "refactor", "function", "class", "script"],
    "general": []  # fallback category
}
```

**How It Works:**
1. User query is analyzed for keywords
2. If reasoning keywords found → routing to reasoning provider
3. If coding keywords found → routing to coding provider  
4. Otherwise → routing to general provider

### 6. Storage Configuration

Optional MinIO object storage for file attachments:

```python
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET = os.getenv("MINIO_BUCKET")
```

## Helper Functions

### `get_api_key_for_provider(provider: str) -> str`

Retrieves API key for a specific provider with proper fallback handling.

```python
def get_api_key_for_provider(provider: str) -> str:
    """Get API key for a specific provider"""
    provider_info = AVAILABLE_PROVIDERS.get(provider, {})
    api_key_env = provider_info.get("api_key_env")
    if api_key_env:
        return os.getenv(api_key_env)
    return None
```

**Usage:**
```python
from config.setting import get_api_key_for_provider

openai_key = get_api_key_for_provider("openai")
claude_key = get_api_key_for_provider("anthropic")
gemini_key = get_api_key_for_provider("gemini")
```

## Environment Variable Reference

### Required Variables (at least one)
```bash
OPENAI_API_KEY=your_openai_api_key
CLAUDE_API_KEY=your_claude_api_key
GEMINI_API_KEY=your_gemini_api_key
```

### Optional Provider Overrides
```bash
# Task-specific provider configuration
REASONING_PROVIDER=openai
REASONING_MODEL=gpt-4o
CODING_PROVIDER=anthropic
CODING_MODEL=claude-3-5-sonnet-20241022
GENERAL_PROVIDER=anthropic
GENERAL_MODEL=claude-3-5-haiku-20241022

# Custom API endpoints
OPENAI_API_BASE=https://api.openai.com/v1
CLAUDE_API_BASE=https://api.anthropic.com
GEMINI_API_BASE=https://generativelanguage.googleapis.com/v1beta
OLLAMA_API_BASE=http://localhost:11434
```

### Vector Database Configuration
```bash
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6333
QDRANT_COLLECTION=ai_cache
```

### Embedding Configuration
```bash
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBED_DIM=384
```

### Storage Configuration (Optional)
```bash
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=ai-cli-files
```

## Configuration Validation

The module includes validation logic to ensure proper setup:

1. **Provider Validation**: Ensures at least one AI provider has a valid API key
2. **Type Validation**: Converts string environment variables to appropriate types
3. **Fallback Strategy**: Provides sensible defaults for all optional settings
4. **Error Handling**: Graceful handling of missing or invalid configuration

## Usage Examples

### Basic Configuration Check
```python
from config.setting import AI_PROVIDERS, AVAILABLE_PROVIDERS

# Check configured providers
for category, config in AI_PROVIDERS.items():
    if config.get("api_key"):
        print(f"{category}: {config['provider']}/{config['model']} ✅")
    else:
        print(f"{category}: {config['provider']}/{config['model']} ❌")
```

### Dynamic Provider Discovery
```python
from config.setting import AVAILABLE_PROVIDERS, get_api_key_for_provider

configured_providers = []
for provider_name in AVAILABLE_PROVIDERS.keys():
    if get_api_key_for_provider(provider_name):
        configured_providers.append(provider_name)

print(f"Available providers: {configured_providers}")
```

### Task Category Lookup
```python
from config.setting import TASK_CATEGORIES

def categorize_query(query: str) -> str:
    query_lower = query.lower()
    for category, keywords in TASK_CATEGORIES.items():
        if any(keyword in query_lower for keyword in keywords):
            return category
    return "general"

# Examples
print(categorize_query("How does encryption work?"))  # → reasoning
print(categorize_query("Write a Python function"))     # → coding
print(categorize_query("Hello there"))                 # → general
```

## Best Practices

### 1. Environment File Management
- Use `.env` file for local development
- Use system environment variables for production
- Never commit `.env` files to version control
- Regularly rotate API keys

### 2. Provider Configuration
- Configure multiple providers for redundancy
- Use different models for different task types
- Monitor usage and costs across providers
- Implement rate limiting and error handling

### 3. Security Considerations
- Store API keys securely
- Use environment-specific configurations
- Implement proper access controls
- Monitor for unusual usage patterns

### 4. Performance Optimization
- Choose embedding models based on requirements
- Configure appropriate vector dimensions
- Use local Qdrant for development
- Scale vector database for production

## Troubleshooting

### Common Issues

1. **No Providers Configured**
   - Ensure at least one API key is set
   - Check environment variable names
   - Verify `.env` file is loaded correctly

2. **Qdrant Connection Failed**
   - Check QDRANT_HOST and QDRANT_PORT
   - Ensure Qdrant server is running
   - Verify network connectivity

3. **Embedding Model Not Found**
   - Check EMBED_MODEL name
   - Ensure sentence-transformers is installed
   - Verify model availability

4. **API Authentication Failed**
   - Verify API key validity
   - Check API endpoint URLs
   - Confirm provider-specific requirements

## Extending Configuration

### Adding New Providers
```python
AVAILABLE_PROVIDERS["new_provider"] = {
    "models": ["model1", "model2"],
    "api_key_env": "NEW_PROVIDER_API_KEY",
    "api_base_env": "NEW_PROVIDER_API_BASE"
}
```

### Adding New Task Categories
```python
TASK_CATEGORIES["new_category"] = ["keyword1", "keyword2", "keyword3"]
```

### Adding New Configuration Options
```python
NEW_FEATURE_ENABLED = os.getenv("NEW_FEATURE_ENABLED", "false").lower() == "true"
NEW_FEATURE_CONFIG = os.getenv("NEW_FEATURE_CONFIG", "default_value")
```