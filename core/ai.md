# core/ai.py - AI Client Documentation

## Overview

`core/ai.py` implements the `AIClient` class, which serves as the central interface for all AI provider interactions. It provides a unified API that abstracts away provider-specific implementations while offering intelligent routing, task categorization, and comprehensive fallback strategies.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│    AIClient     │────│     LiteLLM      │────│   AI Providers  │
│  (Orchestrator) │    │  (Unified API)   │    │ (OpenAI, Claude)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Task Category   │    │  Provider Config │    │  Multimodal     │
│ Classification  │    │  & Fallbacks     │    │  Processing     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Class: AIClient

The main class that handles all AI interactions with intelligent provider routing.

### Constructor: `__init__(self)`

Initializes the AI client with provider validation and LiteLLM configuration.

```python
def __init__(self):
    self._validate_providers()
    self._setup_litellm()
```

**Initialization Steps:**
1. Validates that at least one AI provider is configured
2. Sets up environment variables for LiteLLM
3. Configures logging and error handling
4. Prepares provider fallback strategies

### Core Methods

#### `chat()` - Main AI Interaction Method

The primary method for sending requests to AI providers with intelligent routing.

```python
def chat(
    self, 
    prompt: str, 
    context: Optional[str] = None, 
    files: Optional[List[bytes]] = None, 
    images: Optional[List[bytes]] = None,
    category: Optional[str] = None,
    override_provider: Optional[str] = None,
    override_model: Optional[str] = None
) -> str
```

**Parameters:**
- `prompt`: The user's question or request (required)
- `context`: Optional RAG context from similar interactions
- `files`: List of file bytes for document processing
- `images`: List of image bytes for visual analysis
- `category`: Force specific task category (reasoning/coding/general)
- `override_provider`: Force specific provider (openai/anthropic/gemini/ollama)
- `override_model`: Force specific model (e.g., gpt-4o, claude-3-5-sonnet-20241022)

**Returns:** AI response as string

**Workflow:**
1. **Task Categorization**: Automatically categorizes the prompt if no category specified
2. **Provider Selection**: Chooses appropriate provider based on category or overrides
3. **Message Construction**: Builds properly formatted messages for the AI API
4. **Multimodal Handling**: Processes images and files if provided
5. **API Call**: Sends request via LiteLLM with error handling
6. **Response Processing**: Extracts and returns the AI response

#### `_categorize_task()` - Intelligent Task Classification

Analyzes prompts to determine the most appropriate task category.

```python
def _categorize_task(self, prompt: str) -> str
```

**Classification Logic:**
- **Reasoning**: Keywords like "analyze", "explain", "compare", "why", "how"
- **Coding**: Keywords like "code", "implement", "debug", "function", "class"
- **General**: Default fallback for everything else

**Examples:**
```python
_categorize_task("How does quantum computing work?")      # → "reasoning"
_categorize_task("Write a Python function to sort")      # → "coding"
_categorize_task("What's the weather like?")             # → "general"
```

#### `_get_provider_config()` - Dynamic Provider Configuration

Retrieves provider configuration with support for CLI overrides and fallbacks.

```python
def _get_provider_config(
    self, 
    category: str, 
    override_provider: Optional[str] = None, 
    override_model: Optional[str] = None
) -> Dict[str, Any]
```

**Fallback Strategy:**
1. Use CLI overrides if provided
2. Use category-specific default provider
3. Fall back to general category provider
4. Use any configured provider as last resort

#### `_validate_providers()` - Configuration Validation

Ensures at least one AI provider is properly configured with valid API keys.

```python
def _validate_providers(self) -> None
```

**Validation Checks:**
- At least one provider has a valid API key
- Provider configurations are complete
- Environment variables are accessible
- Logs available providers for debugging

#### `_setup_litellm()` - LiteLLM Configuration

Configures LiteLLM with all available provider credentials and settings.

```python
def _setup_litellm(self) -> None
```

**Setup Process:**
1. Iterates through all configured providers
2. Sets appropriate environment variables for each
3. Configures custom API endpoints if specified
4. Sets LiteLLM global options and logging level

## Supported Providers

### OpenAI
- **Models**: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
- **API Key**: `OPENAI_API_KEY`
- **Custom Endpoint**: `OPENAI_API_BASE`
- **Default Category**: Reasoning

### Anthropic
- **Models**: claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022, claude-3-opus-20240229
- **API Key**: `CLAUDE_API_KEY`
- **Custom Endpoint**: `CLAUDE_API_BASE`
- **Default Categories**: Coding (Sonnet), General (Haiku)

### Google Gemini
- **Models**: gemini-pro, gemini-pro-vision, gemini-1.5-pro, gemini-1.5-flash
- **API Key**: `GEMINI_API_KEY`
- **Custom Endpoint**: `GEMINI_API_BASE`
- **Specialty**: Multimodal processing

### Ollama
- **Models**: llama2, codellama, mistral, phi
- **API Key**: Not required (local)
- **Custom Endpoint**: `OLLAMA_API_BASE`
- **Specialty**: Local/offline processing

## Multimodal Support

### Image Processing

The client supports image analysis through base64 encoding:

```python
# Image processing workflow
content = [{"type": "text", "text": user_content}]
for img_bytes in images:
    img_b64 = base64.b64encode(img_bytes).decode()
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
    })
```

**Supported Formats:**
- JPEG, PNG, GIF
- Base64 encoding for API transmission
- Automatic format detection

### File Processing

Documents are processed as text attachments:

```python
# File processing workflow
for file_bytes in files:
    try:
        file_text = file_bytes.decode('utf-8', errors='ignore')[:2000]
        content.append({
            "type": "text", 
            "text": f"\nFile content:\n{file_text}"
        })
    except Exception as e:
        logger.warning(f"Could not process file: {e}")
```

**Supported Operations:**
- Text extraction from various formats
- Size limiting (2000 characters max)
- Error handling for unsupported formats
- UTF-8 decoding with error tolerance

## Usage Examples

### Basic Usage

```python
from core.ai import AIClient

# Initialize client
client = AIClient()

# Simple question
response = client.chat("What is machine learning?")
print(response)
```

### Advanced Usage with Provider Override

```python
# Force specific provider
response = client.chat(
    "Write a Python function", 
    override_provider="anthropic",
    override_model="claude-3-5-sonnet-20241022"
)
```

### Multimodal Usage

```python
# Process image and document
with open("image.jpg", "rb") as f:
    image_data = f.read()

with open("document.txt", "rb") as f:
    file_data = f.read()

response = client.chat(
    "Analyze this image and document",
    images=[image_data],
    files=[file_data]
)
```

### RAG-Enhanced Usage

```python
# With context from vector search
context = "Previous similar questions and answers..."
response = client.chat(
    "New question",
    context=context,
    category="reasoning"
)
```

## Error Handling

### Provider Failures

```python
try:
    response = litellm.completion(
        model=model_name,
        messages=messages,
        temperature=0.7,
        max_tokens=4000,
    )
    return response.choices[0].message.content
except Exception as e:
    logger.error(f"Error calling AI provider {config['provider']}: {e}")
    raise RuntimeError(f"AI request failed: {str(e)}")
```

### Common Error Types

1. **Authentication Errors**: Invalid API keys
2. **Rate Limiting**: Too many requests
3. **Network Errors**: Connection timeouts
4. **Model Errors**: Unsupported models or parameters
5. **Content Errors**: Rejected content or safety filters

### Fallback Strategies

1. **Provider Fallback**: Switch to alternative provider
2. **Model Fallback**: Try different model from same provider
3. **Parameter Adjustment**: Reduce max_tokens or temperature
4. **Retry Logic**: Exponential backoff for transient errors

## Performance Considerations

### Optimization Strategies

1. **Model Selection**: Choose appropriate models for task complexity
2. **Token Management**: Limit context and response length
3. **Caching**: Leverage semantic caching to avoid redundant calls
4. **Batch Processing**: Group multiple requests when possible

### Monitoring and Logging

```python
# Structured logging for monitoring
logger.info(f"Used {category} provider: {model_name}")
logger.warning(f"Unknown provider: {override_provider}. Using default.")
logger.error(f"Error calling AI provider {config['provider']}: {e}")
```

**Log Levels:**
- **INFO**: Successful operations, provider selection
- **WARNING**: Non-critical issues, fallbacks triggered  
- **ERROR**: Failed API calls, configuration problems

## Security Considerations

### API Key Management

- Environment variables for secure storage
- Never log API keys or tokens
- Automatic key rotation support
- Provider-specific security requirements

### Content Safety

- Input validation and sanitization
- Output filtering for sensitive information
- Compliance with provider content policies
- Audit logging for sensitive operations

## Extending the AI Client

### Adding New Providers

1. Update `AVAILABLE_PROVIDERS` in config/setting.py
2. Add provider-specific environment setup in `_setup_litellm()`
3. Test provider integration with sample requests
4. Update documentation and examples

### Adding New Features

```python
def new_feature_method(self, parameter: str) -> Any:
    """
    New feature implementation
    """
    try:
        # Feature logic here
        result = self._process_feature(parameter)
        logger.info(f"Feature executed successfully")
        return result
    except Exception as e:
        logger.error(f"Feature failed: {e}")
        raise
```

### Customizing Task Categories

Modify `TASK_CATEGORIES` in config/setting.py to add new categories or keywords:

```python
TASK_CATEGORIES = {
    "reasoning": ["analyze", "explain", "reason"],
    "coding": ["code", "implement", "debug"],
    "creative": ["write", "story", "poem"],  # New category
    "general": []
}
```

## Testing and Debugging

### Unit Testing

```python
import pytest
from core.ai import AIClient

def test_task_categorization():
    client = AIClient()
    assert client._categorize_task("How does this work?") == "reasoning"
    assert client._categorize_task("Write Python code") == "coding"
    assert client._categorize_task("Hello there") == "general"

def test_provider_config():
    client = AIClient()
    config = client._get_provider_config("reasoning")
    assert "provider" in config
    assert "model" in config
    assert "api_key" in config
```

### Integration Testing

```python
def test_ai_response():
    client = AIClient()
    response = client.chat("What is 2+2?")
    assert isinstance(response, str)
    assert len(response) > 0
```

### Debugging Tips

1. **Enable Verbose Logging**: Set `litellm.set_verbose = True`
2. **Check Environment Variables**: Verify API keys are loaded
3. **Test Individual Providers**: Isolate provider-specific issues
4. **Monitor API Usage**: Track requests and responses
5. **Validate Configurations**: Ensure provider configs are complete