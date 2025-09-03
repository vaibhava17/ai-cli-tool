# AI CLI Tool >

A powerful command-line AI assistant with multi-provider support, semantic caching, and RAG (Retrieval-Augmented Generation) capabilities. Built for developers who want intelligent AI interactions with advanced features like provider routing, semantic search, and conversation memory.

## <� Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CLI Runner    │────│  AI Client       │────│  LiteLLM        │
│   (Typer App)   │    │  (Multi-Provider) │    │  (API Gateway)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Embeddings     │    │  Vector Store    │    │   AI Providers  │
│ (Transformers)  │    │   (Qdrant)       │    │ OpenAI/Claude/  │
│                 │    │                  │    │ Gemini/Ollama   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        
         ▼                        ▼                        
┌─────────────────┐    ┌──────────────────┐               
│  RAG Context    │    │  Semantic Cache  │               
│   Builder       │    │   & Search       │               
└─────────────────┘    └──────────────────┘               

Flow: Query → Embed → Search Cache → [Hit: Return | Miss: RAG + AI] → Store → Response
```

## ( Features

### <� **Multi-Provider AI Support**
- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo
- **Anthropic**: Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus
- **Google Gemini**: Gemini Pro, Gemini Pro Vision, Gemini 1.5 Pro/Flash
- **Ollama**: Local models (Llama2, CodeLlama, Mistral, Phi)

### >� **Intelligent Task Routing**
- **Reasoning Tasks** � OpenAI (analysis, explanations, comparisons)
- **Coding Tasks** � Anthropic Claude (implementation, debugging, refactoring)
- **General Tasks** � Anthropic Haiku (conversations, simple questions)

### � **Advanced Capabilities**
- **Semantic Caching**: Vector-based response caching for instant retrieval
- **RAG Enhancement**: Context-aware responses using similar past interactions
- **Multimodal Support**: Process images and documents alongside text
- **Provider Analytics**: Track usage statistics and performance metrics
- **Dynamic Provider Override**: Force specific providers/models via CLI

### =� **Analytics & Monitoring**
- Provider usage statistics
- Task category distribution
- Performance metrics
- Recent activity tracking

## =� Quick Start

### Prerequisites

- Python 3.8+
- Qdrant vector database
- At least one AI provider API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-cli-tool
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Qdrant (Vector Database)**
   ```bash
   # Using Docker (recommended)
   docker run -p 6333:6333 qdrant/qdrant
   
   # Or install locally
   # See: https://qdrant.tech/documentation/quick-start/
   ```

5. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

### Configuration

Create a `.env` file with your API keys:

```env
# AI Provider API Keys (at least one required)
OPENAI_API_KEY=your_openai_api_key
CLAUDE_API_KEY=your_claude_api_key
GEMINI_API_KEY=your_gemini_api_key

# Vector Database (Qdrant)
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6333
QDRANT_COLLECTION=ai_cache

# Optional: Provider Overrides
REASONING_PROVIDER=openai
REASONING_MODEL=gpt-4o
CODING_PROVIDER=anthropic
CODING_MODEL=claude-3-5-sonnet-20241022
GENERAL_PROVIDER=anthropic
GENERAL_MODEL=claude-3-5-haiku-20241022
```

## =� Usage

### Basic Commands

```bash
# Simple question (auto-categorized and routed)
python main.py ask "What is machine learning?"

# Coding question (auto-routed to coding provider)  
python main.py ask "Write a Python function to sort a list"

# Analysis question (auto-routed to reasoning provider)
python main.py ask "Compare REST vs GraphQL APIs"
```

### Example RAG Flow in Action

```bash
# First question - gets stored in vector DB
$ python main.py ask "How does neural network training work?"
[openai/gpt-4o/reasoning] Neural networks learn through backpropagation...

# Similar question - gets RAG context enhancement 
$ python main.py ask "Explain backpropagation in deep learning"
[anthropic/claude-3-5-sonnet-20241022/reasoning + RAG] Based on previous context about neural networks...

# Very similar question - gets cached response
$ python main.py ask "How do neural networks learn?"  
[CACHED - openai/gpt-4o/reasoning - similarity: 0.912] Neural networks learn through backpropagation...
```

### Advanced Usage

```bash
# Force specific provider
python main.py ask "Hello world" --provider openai

# Force specific model
python main.py ask "Write code" --provider anthropic --model claude-3-5-sonnet-20241022

# Process image
python main.py ask "What's in this image?" --image photo.jpg

# Process document
python main.py ask "Summarize this document" --file document.pdf

# Skip cache for fresh response
python main.py ask "Current weather" --no-cache

# Filter cache by provider
python main.py ask "Show me previous Claude responses" --filter-provider "anthropic/claude-3-5-sonnet-20241022"
```

### Management Commands

```bash
# List available providers and models
python main.py providers

# Show usage statistics
python main.py stats

# Show current configuration
python main.py configure
```

## =� Command Reference

### `ask` - Main AI Interaction

```bash
python main.py ask "Your question" [OPTIONS]
```

**Options:**
- `--image, -i PATH`: Path to image file
- `--file, -f PATH`: Path to document file  
- `--provider, -p TEXT`: Force specific provider (openai/anthropic/gemini/ollama)
- `--model, -m TEXT`: Force specific model
- `--category, -c TEXT`: Force task category (reasoning/coding/general)
- `--filter-provider TEXT`: Filter cache by AI provider
- `--no-cache`: Skip semantic cache lookup
- `--k INTEGER`: Top-K retrieved for context (default: 5)
- `--threshold FLOAT`: Similarity threshold for cache hit (default: 0.85)

### `providers` - List Available Providers

```bash
python main.py providers
```

Shows all supported providers, their models, and configuration status.

### `stats` - Usage Analytics  

```bash
python main.py stats
```

Displays comprehensive usage statistics and analytics.

### `configure` - Configuration Status

```bash
python main.py configure
```

Shows current AI provider configuration and setup instructions.

## =' Configuration

### Environment Variables

#### Required (at least one)
```env
OPENAI_API_KEY=your_openai_key
CLAUDE_API_KEY=your_claude_key
GEMINI_API_KEY=your_gemini_key
```

#### Optional Provider Overrides
```env
REASONING_PROVIDER=openai
REASONING_MODEL=gpt-4o
CODING_PROVIDER=anthropic  
CODING_MODEL=claude-3-5-sonnet-20241022
GENERAL_PROVIDER=anthropic
GENERAL_MODEL=claude-3-5-haiku-20241022
```

#### Vector Database
```env
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6333
QDRANT_COLLECTION=ai_cache
```

#### Embeddings
```env
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBED_DIM=384
```

### Task Categories

The system automatically categorizes tasks based on keywords:

- **Reasoning**: analyze, explain, reason, why, how, compare, evaluate
- **Coding**: code, implement, fix, debug, refactor, function, class, script
- **General**: Everything else (fallback)

## <� Architecture Details

### Core Components

1. **CLI Interface** (`cli/runner.py`)
   - Command parsing and validation
   - User interaction and output formatting
   - Error handling and user feedback

2. **AI Client** (`core/ai.py`)
   - Multi-provider integration via LiteLLM
   - Intelligent task categorization
   - Provider routing and fallback strategies

3. **Vector Store** (`core/vectorstore.py`)
   - Qdrant-based semantic search
   - Provider metadata tracking
   - Usage analytics and statistics

4. **Embeddings** (`core/embeddings.py`)
   - Text-to-vector conversion
   - Model caching and optimization
   - Batch processing support

5. **RAG System** (`core/rag.py`)
   - Context building from similar interactions
   - Relevance scoring and filtering
   - Token-aware context management

6. **Configuration** (`config/setting.py`)
   - Environment variable management
   - Provider configuration
   - Task categorization rules

### Enhanced Data Flow

1. **User Input** → CLI parses command and options
2. **Embedding** → Convert query to vector using sentence-transformers  
3. **Vector Search** → Find similar past interactions in Qdrant
4. **Cache Check** → Return cached response if similarity ≥ threshold (0.85)
5. **RAG Enhancement** → If cache miss, build filtered context from similar interactions
6. **Provider Selection** → Choose AI provider based on task categorization  
7. **AI Request** → Send enhanced prompt with RAG context to provider via LiteLLM
8. **Response Processing** → Receive and validate AI response
9. **Storage** → Store new interaction with full metadata in vector DB
10. **Output** → Return formatted response with provider and similarity info

## =� Performance Features

### Semantic Caching
- Vector similarity matching (cosine distance)
- Configurable similarity thresholds (default: 0.85)
- Provider-specific cache filtering
- Significant cost and time savings for repeated queries

### RAG (Retrieval-Augmented Generation) 
- Context building from similar past interactions
- Relevance filtering based on query word overlap
- Provider and category metadata enhancement
- Token-aware context limiting (max 2000 chars)
- Improves response quality and consistency

### Intelligent Provider Routing
- Automatic task categorization
- Provider specialization (reasoning/coding/general)
- Fallback strategies for failures
- Dynamic override capabilities

### Memory Optimization
- Model caching with LRU eviction
- Batch embedding processing
- Token-aware context limiting
- Efficient vector operations

## >� Development

### Project Structure

```
ai-cli-tool/
   main.py                 # Entry point
   requirements.txt        # Dependencies
   .env.example           # Environment template
   cli/
      __init__.py
      runner.py          # CLI interface
      runner.md          # CLI documentation
   config/
      __init__.py
      setting.py         # Configuration management
      setting.md         # Config documentation
   core/
      __init__.py
      ai.py              # AI client
      ai.md              # AI documentation
      embeddings.py      # Text embeddings
      embeddings.md      # Embeddings documentation
      rag.py             # RAG system
      rag.md             # RAG documentation
      vectorstore.py     # Vector database
      vectorstore.md     # Vector store documentation
   docs/                  # Additional documentation
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/ -v

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=. --cov-report=html
```

### Development Setup

```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install black mypy flake8

# Format code
black .

# Type checking
mypy .

# Linting
flake8 .
```

## =3 Docker Deployment

### Using Docker Compose

```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  ai-cli-tool:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - CLAUDE_API_KEY=${CLAUDE_API_KEY}
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    depends_on:
      - qdrant
    volumes:
      - ./data:/app/data

volumes:
  qdrant_data:
```

```bash
# Start services
docker-compose up -d

# Use the CLI
docker-compose exec ai-cli-tool python main.py ask "Hello world"
```

### Standalone Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

```bash
# Build and run
docker build -t ai-cli-tool .
docker run -e OPENAI_API_KEY=your_key ai-cli-tool ask "Hello"
```

## = Security

### API Key Management
- Store keys in environment variables only
- Never commit keys to version control
- Use different keys for development/production
- Implement key rotation policies

### Network Security
- Use HTTPS for all API communications
- Implement rate limiting
- Monitor API usage for anomalies
- Use secure vector database connections

### Data Privacy
- No sensitive data logged
- Optional data encryption at rest
- Configurable data retention policies
- GDPR compliance considerations

## =� Troubleshooting

### Common Issues

1. **No AI providers configured**
   ```bash
   # Check environment variables
   python main.py configure
   
   # Verify .env file exists and contains API keys
   cat .env
   ```

2. **Qdrant connection failed**
   ```bash
   # Check if Qdrant is running
   curl http://localhost:6333/health
   
   # Start Qdrant
   docker run -p 6333:6333 qdrant/qdrant
   ```

3. **Embedding model download failed**
   ```bash
   # Clear cache and retry
   rm -rf ~/.cache/torch/sentence_transformers/
   python main.py ask "test query"
   ```

4. **Permission denied errors**
   ```bash
   # Use virtual environment
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python main.py ask "debug query"

# Check vector database status
python main.py stats
```

### Performance Issues

```bash
# Use smaller embedding model
export EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Reduce context size
python main.py ask "query" --k 3

# Skip cache for testing
python main.py ask "query" --no-cache
```

## > Contributing

We welcome contributions! Here's how to get started:

### Development Workflow

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make changes and add tests**
4. **Run tests and linting**
   ```bash
   pytest tests/ -v
   black .
   mypy .
   ```
5. **Commit changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
6. **Push and create PR**
   ```bash
   git push origin feature/amazing-feature
   ```

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write docstrings for modules and functions
- Include tests for new features
- Update documentation as needed

### Adding New Providers

1. Update `AVAILABLE_PROVIDERS` in `config/setting.py`
2. Add provider setup in `core/ai.py`
3. Test integration with sample requests
4. Update documentation and examples

## =� License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## =O Acknowledgments

- [LiteLLM](https://github.com/BerriAI/litellm) for universal AI provider integration
- [Qdrant](https://qdrant.tech/) for high-performance vector search
- [Sentence Transformers](https://www.sbert.net/) for text embeddings
- [Typer](https://typer.tiangolo.com/) for the elegant CLI framework

## =� Support

- **Documentation**: Check the individual `.md` files for each module
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Join GitHub Discussions for questions and ideas
- **Email**: [support@example.com](mailto:support@example.com)

---

**Made with d by the AI CLI Tool Team**

> < **Star this repo** if you find it useful!
> = **Report issues** to help us improve!
> > **Contribute** to make it even better!