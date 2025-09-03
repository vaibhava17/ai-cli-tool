# requirements.txt - Dependencies Documentation

## Overview

`requirements.txt` defines all Python dependencies required for the AI CLI Tool. Each dependency serves a specific purpose in the system architecture, from AI provider integration to vector database operations.

## Dependencies Breakdown

### Core Framework Dependencies

#### `typer`
- **Purpose**: Modern CLI framework with type hints and auto-completion
- **Version**: Latest stable
- **Usage**: Powers the main CLI interface in `cli/runner.py`
- **Features**: 
  - Type-safe command definitions
  - Automatic help generation
  - Rich output formatting
  - Command validation and error handling

**Installation:**
```bash
pip install typer
```

**Key Features Used:**
- Command decorators (`@app.command()`)
- Type-hinted parameters with defaults
- Rich help text and error messages
- Auto-completion support

#### `python-dotenv`
- **Purpose**: Environment variable management from .env files
- **Version**: Latest stable
- **Usage**: Configuration loading in `config/setting.py`
- **Features**:
  - Automatic .env file loading
  - Environment variable precedence
  - Cross-platform compatibility

**Installation:**
```bash
pip install python-dotenv
```

**Configuration:**
```python
from dotenv import load_dotenv
load_dotenv()  # Loads variables from .env file
```

### AI Provider Dependencies

#### `litellm`
- **Purpose**: Universal API client for multiple AI providers
- **Version**: Latest stable
- **Usage**: Core AI provider integration in `core/ai.py`
- **Supported Providers**:
  - OpenAI (GPT models)
  - Anthropic (Claude models)
  - Google (Gemini models)
  - Ollama (Local models)

**Installation:**
```bash
pip install litellm
```

**Key Features:**
- Unified API across providers
- Automatic model routing
- Built-in retry logic
- Response streaming support

### Vector Database Dependencies

#### `qdrant-client`
- **Purpose**: Official Python client for Qdrant vector database
- **Version**: Latest stable
- **Usage**: Vector operations in `core/vectorstore.py`
- **Features**:
  - High-performance vector search
  - Advanced filtering capabilities
  - Collection management
  - Batch operations

**Installation:**
```bash
pip install qdrant-client
```

**Vector Database Setup:**
```bash
# Run Qdrant with Docker
docker run -p 6333:6333 qdrant/qdrant
```

### Embedding Dependencies

#### `sentence-transformers`
- **Purpose**: Pre-trained transformer models for text embeddings
- **Version**: Latest stable
- **Usage**: Text embedding generation in `core/embeddings.py`
- **Models Supported**:
  - `all-MiniLM-L6-v2` (384 dims, fast)
  - `all-mpnet-base-v2` (768 dims, high quality)
  - `multi-qa-MiniLM-L6-cos-v1` (384 dims, QA optimized)

**Installation:**
```bash
pip install sentence-transformers
```

**Model Caching:**
Models are automatically downloaded and cached on first use:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
```

### File Processing Dependencies

#### `pillow`
- **Purpose**: Python Imaging Library for image processing
- **Version**: Latest stable
- **Usage**: Image file handling for multimodal AI requests
- **Supported Formats**: JPEG, PNG, GIF, BMP, TIFF

**Installation:**
```bash
pip install pillow
```

**Usage Example:**
```python
from PIL import Image
image = Image.open("photo.jpg")
image_bytes = image.tobytes()
```

#### `pymupdf`
- **Purpose**: PDF document processing and text extraction
- **Version**: Latest stable
- **Usage**: PDF file processing in AI requests
- **Features**:
  - Text extraction from PDFs
  - Metadata extraction
  - Page-by-page processing

**Installation:**
```bash
pip install pymupdf
```

**Usage Example:**
```python
import fitz  # pymupdf
doc = fitz.open("document.pdf")
text = doc[0].get_text()  # Extract text from first page
```

## Dependency Categories

### Production Dependencies

These are required for core functionality:

```txt
typer              # CLI framework
qdrant-client      # Vector database
sentence-transformers  # Text embeddings
python-dotenv      # Environment configuration
litellm            # AI provider integration
```

### Optional Dependencies

These enhance functionality but aren't strictly required:

```txt
pillow             # Image processing (for multimodal requests)
pymupdf            # PDF processing (for document analysis)
```

### Development Dependencies (Not in requirements.txt)

For development and testing:

```txt
pytest             # Testing framework
black              # Code formatting
mypy               # Type checking
pytest-asyncio     # Async testing
```

## Installation Guide

### Basic Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-cli-tool

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Development Installation

```bash
# Install with development dependencies
pip install -r requirements.txt
pip install pytest black mypy pytest-asyncio

# Install in development mode
pip install -e .
```

### Docker Installation

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

## Version Management

### Pinning Versions

For production deployments, consider pinning specific versions:

```txt
typer==0.9.0
qdrant-client==1.6.9
sentence-transformers==2.2.2
python-dotenv==1.0.0
litellm==1.17.3
pillow==10.0.1
pymupdf==1.23.8
```

### Updating Dependencies

```bash
# Check for outdated packages
pip list --outdated

# Update all packages
pip install -r requirements.txt --upgrade

# Update specific package
pip install --upgrade litellm
```

## Platform-Specific Considerations

### Windows

```bash
# Some packages may require Visual Studio Build Tools
# Install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Alternative: Use conda
conda install -c conda-forge sentence-transformers
```

### macOS

```bash
# No special requirements for most packages
# Use Homebrew for system dependencies if needed
brew install python3
```

### Linux

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev build-essential

# For GPU acceleration (optional)
pip install sentence-transformers[gpu]
```

## GPU Acceleration (Optional)

For improved embedding performance:

```bash
# Install CUDA-enabled versions
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

## Troubleshooting

### Common Installation Issues

1. **Build Errors**
   ```bash
   # Install build tools
   pip install --upgrade setuptools wheel
   pip install --upgrade pip
   ```

2. **Memory Issues**
   ```bash
   # Install packages one by one
   pip install typer
   pip install qdrant-client
   # ... continue for each package
   ```

3. **Permission Errors**
   ```bash
   # Use user installation
   pip install --user -r requirements.txt
   
   # Or use virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate
   ```

4. **Network Issues**
   ```bash
   # Use alternative index
   pip install -r requirements.txt -i https://pypi.org/simple/
   
   # Use proxy if needed
   pip install -r requirements.txt --proxy http://proxy.example.com:8080
   ```

### Package-Specific Issues

#### Sentence Transformers

```bash
# Clear model cache if corrupted
rm -rf ~/.cache/torch/sentence_transformers/

# Manual model download
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

#### Qdrant Client

```bash
# Test Qdrant connection
python -c "from qdrant_client import QdrantClient; client = QdrantClient('localhost', port=6333); print('Connected!')"

# Start Qdrant server
docker run -p 6333:6333 qdrant/qdrant
```

#### LiteLLM

```bash
# Test API connectivity
python -c "import litellm; print(litellm.completion(model='gpt-3.5-turbo', messages=[{'role': 'user', 'content': 'test'}]))"
```

## Security Considerations

### Package Verification

```bash
# Verify package integrity
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Use hash verification
pip install -r requirements.txt --require-hashes
```

### Dependency Scanning

```bash
# Check for known vulnerabilities
pip install safety
safety check -r requirements.txt

# Alternative: Use pip-audit
pip install pip-audit
pip-audit -r requirements.txt
```

## Production Deployment

### Requirements for Production

```txt
# Minimal production requirements
typer>=0.9.0
qdrant-client>=1.6.0
sentence-transformers>=2.2.0
python-dotenv>=1.0.0
litellm>=1.17.0

# Optional but recommended
pillow>=10.0.0
pymupdf>=1.23.0
```

### Docker Multi-stage Build

```dockerfile
# Multi-stage build for smaller production image
FROM python:3.9-slim as builder

COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.9-slim
COPY --from=builder /root/.local /root/.local
COPY . /app
WORKDIR /app

ENV PATH=/root/.local/bin:$PATH
CMD ["python", "main.py"]
```

### Environment Variables

```bash
# Production environment variables
OPENAI_API_KEY=your_production_openai_key
CLAUDE_API_KEY=your_production_claude_key
QDRANT_HOST=production-qdrant-host
QDRANT_PORT=6333
```

## Performance Optimization

### Memory Usage

```bash
# Install with minimal dependencies
pip install --no-deps typer
pip install --no-deps qdrant-client

# Use lighter embedding models
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2  # 384 dims
# Instead of: all-mpnet-base-v2  # 768 dims
```

### Startup Time

```bash
# Pre-download models during build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

### Network Optimization

```bash
# Use CDN mirrors for faster downloads
pip install -r requirements.txt -i https://pypi.douban.com/simple/  # China mirror
pip install -r requirements.txt -i https://pypi.org/simple/  # Official
```