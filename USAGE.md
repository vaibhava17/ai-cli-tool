# AI CLI Tool - Complete Usage Guide

This guide shows you exactly how to use the AI CLI Tool with real examples and explanations of what happens behind the scenes.

---

## 🚀 Getting Started

### Prerequisites Check
```bash
# Check if you have everything needed
python --version        # Should be Python 3.8+
docker --version        # Should be installed
which git              # Should be installed
```

### Step-by-Step Setup
```bash
# 1. Clone and enter directory
git clone <repository-url>
cd ai-cli-tool

# 2. Create isolated Python environment  
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Start vector database (runs in background)
docker run -d --name ai-cli-qdrant -p 6333:6333 qdrant/qdrant

# 5. Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Verify Setup
```bash
# Check if everything is working
python main.py configure   # Should show configured providers
python main.py providers   # Should list available AI providers
curl http://localhost:6333  # Should return Qdrant version info
```

---

## 💡 Understanding the System

### How It Works
1. **You ask a question** → System embeds it into a vector
2. **Searches cache** → Looks for similar past questions  
3. **Cache hit?** → Returns instant answer if similarity > 0.85
4. **Cache miss?** → Uses RAG context + calls fresh AI
5. **Stores response** → Saves for future similar questions

### The Smart Output
```bash
# What you'll see:
🔄 CACHE HIT (PERFECT) - Similarity: 1.000     # Instant from cache
📦 Original Source: openai/gpt-4o | Category: reasoning  
⚡ Response Time: Instant (0 API calls)

# OR:
🤖 AI RESPONSE + RAG Context                   # Fresh AI call
🔗 Provider: anthropic/claude-3.5-sonnet | Category: coding
💰 API Call: Made fresh request to anthropic/claude-3-5-sonnet
📚 Enhanced: Using context from previous interactions
```

---

## 🎯 Basic Usage Patterns

### Simple Questions
```bash
# General questions → Routed to Gemini (fast & cheap)
python main.py ask "What's the weather like today?"
python main.py ask "Hello! How are you?"
python main.py ask "Tell me a joke"

# Expected output:
🤖 AI RESPONSE
🔗 Provider: gemini/gemini-1.5-flash | Category: general
💰 API Call: Made fresh request to gemini/gemini-1.5-flash
```

### Coding Questions
```bash
# Programming questions → Routed to Claude/OpenAI (code specialists)
python main.py ask "Write a Python function to check if a number is prime"
python main.py ask "Fix this code: def hello(): print('hi'"
python main.py ask "Explain how recursion works"

# Expected output:
🤖 AI RESPONSE
🔗 Provider: anthropic/claude-3-5-sonnet | Category: coding
💰 API Call: Made fresh request to anthropic/claude-3-5-sonnet
```

### Analysis Questions  
```bash
# Complex reasoning → Routed to OpenAI GPT-4o (reasoning specialist)
python main.py ask "Why is quantum computing important?"
python main.py ask "Compare democracy vs authoritarianism"
python main.py ask "Explain the economic impact of AI"

# Expected output:
🤖 AI RESPONSE
🔗 Provider: openai/gpt-4o | Category: reasoning
💰 API Call: Made fresh request to openai/gpt-4o
```

---

## 🔄 Caching in Action

### First Time vs Cached
```bash
# First time asking (calls AI)
$ python main.py ask "What is machine learning?"
🤖 AI RESPONSE
🔗 Provider: gemini/gemini-1.5-flash | Category: general
💰 API Call: Made fresh request to gemini/gemini-1.5-flash
[Response appears here...]

# Exact same question (instant cache)
$ python main.py ask "What is machine learning?"
🔄 CACHE HIT (PERFECT) - Similarity: 1.000
📦 Original Source: gemini/gemini-1.5-flash | Category: general
⚡ Response Time: Instant (0 API calls)
[Same response, but instant...]
```

### Similar Questions (RAG Enhancement)
```bash
# Related question gets enhanced context
$ python main.py ask "How does machine learning differ from AI?"
🤖 AI RESPONSE + RAG Context
🔗 Provider: openai/gpt-4o | Category: reasoning
💰 API Call: Made fresh request to openai/gpt-4o
📚 Enhanced: Using context from previous interactions
[Enhanced response using previous ML conversation...]
```

### Cache Quality Levels
- **PERFECT (0.95+)** - Nearly identical questions
- **HIGH (0.85-0.94)** - Very similar questions  
- **Below 0.85** - Different enough to call AI with RAG context

---

## 🛠 Advanced Features

### Provider Override
```bash
# Force specific provider (overrides smart routing)
python main.py ask "Write code" --provider openai
python main.py ask "Analyze data" --provider gemini
python main.py ask "Debug code" --provider anthropic --model claude-3-5-sonnet-20241022

# The system will tell you it was overridden:
Provider selection: coding -> openai/gpt-4o (CLI override)
```

### File Processing
```bash
# Process images
python main.py ask "What's in this image?" --image screenshot.png
python main.py ask "Describe this photo" --image vacation.jpg

# Process documents  
python main.py ask "Summarize this PDF" --file report.pdf
python main.py ask "Extract key points" --file document.txt
```

### Cache Control
```bash
# Skip cache entirely (always call AI fresh)
python main.py ask "What time is it?" --no-cache

# Require higher similarity for cache hits
python main.py ask "Question" --threshold 0.95  # Default is 0.85

# Control how many similar interactions to consider
python main.py ask "Question" --k 10  # Default is 5
```

### Category Override
```bash
# Force specific task category (affects provider selection)
python main.py ask "Hello world" --category coding     # Forces coding provider
python main.py ask "Debug code" --category general     # Forces general provider  
python main.py ask "Simple question" --category reasoning  # Forces reasoning provider
```

---

## 📊 Monitoring & Analytics

### Check System Status
```bash
# See all configured providers
python main.py configure

# Output:
🔧 AI Provider Configuration:
  reasoning: openai/gpt-4o - ✅ Configured
  coding: anthropic/claude-3-5-sonnet-20241022 - ❌ Missing API Key  
  general: gemini/gemini-1.5-flash - ✅ Configured
```

### View Usage Statistics
```bash
# See your usage patterns
python main.py stats

# Output:
📊 AI Provider Statistics
Total interactions: 25
Recent activity (24h): 12

🤖 Usage by Provider:
  openai/gpt-4o: 8
  gemini/gemini-1.5-flash: 15
  anthropic/claude-3-5-sonnet-20241022: 2

📋 Usage by Category:
  general: 15
  reasoning: 6
  coding: 4
```

### List All Available Providers
```bash
# See what providers and models are available
python main.py providers

# Output:
🤖 Available AI Providers:

✅ openai
   Models: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
   API Key: OPENAI_API_KEY

❌ anthropic  
   Models: claude-3-5-sonnet-20241022, claude-3-opus-20240229
   API Key: CLAUDE_API_KEY

✅ google
   Models: gemini-1.5-pro, gemini-1.5-flash, gemini-pro  
   API Key: GEMINI_API_KEY
```

---

## 🎯 Real-World Use Cases

### Developer Workflow
```bash
# Progressive code development with context
python main.py ask "Create a REST API in Python using FastAPI"
# [Gets detailed FastAPI code...]

python main.py ask "Add user authentication to that API"  
# [Uses previous FastAPI context to add auth...]

python main.py ask "Write tests for the authentication endpoints"
# [Builds on previous context for targeted tests...]

python main.py ask "How do I deploy this to AWS?"
# [Considers the FastAPI + auth context for deployment...]
```

### Research & Learning
```bash
# Build knowledge systematically
python main.py ask "What is blockchain technology?"
# [Basic explanation...]

python main.py ask "How do smart contracts work?"
# [Enhanced with blockchain context...]

python main.py ask "What are the pros and cons of blockchain for supply chain?"
# [Uses both blockchain and smart contract context...]
```

### Daily Development Tasks
```bash
# Quick coding help
python main.py ask "Python one-liner to read CSV file"
python main.py ask "Regular expression to validate email"
python main.py ask "How to handle exceptions in Python"

# These get cached for team members:
python main.py ask "Python one-liner to read CSV file"  # Instant cache hit
```

### Debugging Sessions
```bash
# Contextual debugging
python main.py ask "Fix this error: NameError: name 'x' is not defined"
python main.py ask "Why does my Python code run slowly?"  
python main.py ask "How to profile Python performance?"
```

---

## 🔧 Configuration Scenarios

### Single Provider Setup (Cost-Effective)
```bash
# Use only OpenAI for everything
OPENAI_API_KEY=your_key

# Override all categories to use OpenAI:
REASONING_PROVIDER=openai
CODING_PROVIDER=openai  
GENERAL_PROVIDER=openai

REASONING_MODEL=gpt-4o
CODING_MODEL=gpt-4o
GENERAL_MODEL=gpt-3.5-turbo  # Cheaper for general questions
```

### Multi-Provider Setup (Optimized)
```bash
# Best provider for each task type
OPENAI_API_KEY=your_openai_key      # For reasoning
CLAUDE_API_KEY=your_claude_key      # For coding (when you have credits)
GEMINI_API_KEY=your_gemini_key      # For general chat

# Let the system auto-route (recommended)
# Or customize:
REASONING_PROVIDER=openai
CODING_PROVIDER=anthropic
GENERAL_PROVIDER=gemini
```

### Local Development Setup
```bash
# Use faster/cheaper models for development
REASONING_MODEL=gpt-4o-mini
CODING_MODEL=claude-3-5-haiku-20241022  
GENERAL_MODEL=gemini-1.5-flash

# Use smaller embedding model for speed
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## 🚨 Common Issues & Solutions

### Issue: "No AI providers configured"
```bash
# Check configuration
python main.py configure

# Fix: Add at least one API key to .env
OPENAI_API_KEY=your_key_here
```

### Issue: "No route to host" (Qdrant)
```bash
# Check if Qdrant is running
curl http://localhost:6333

# Fix: Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# Or check if host is correct in .env:
QDRANT_HOST=127.0.0.1  # Not 192.168.x.x
```

### Issue: "Anthropic credit balance too low"
```bash
# Options:
# 1. Add credits at https://console.anthropic.com/account/billing
# 2. Switch to OpenAI for coding:
CODING_PROVIDER=openai
CODING_MODEL=gpt-4o
```

### Issue: Slow response times
```bash
# Use faster embedding model
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Reduce context search
python main.py ask "question" --k 3  # Instead of default 5

# Use faster models
GENERAL_MODEL=gemini-1.5-flash  # Instead of gemini-1.5-pro
```

### Issue: Too many cache misses
```bash
# Lower similarity threshold
python main.py ask "question" --threshold 0.7  # Instead of 0.85

# Check what's being cached
python main.py stats
```

---

## 🔍 Debug Mode

### Enable Verbose Logging
```bash
# See everything that's happening
export LOG_LEVEL=DEBUG
python main.py ask "test question"

# You'll see detailed logs:
# - Embedding model loading
# - Vector search results  
# - RAG context building
# - Provider selection logic
# - API call details
```

### Understanding Log Output
```bash
# Key log messages to watch for:
cache_search - INFO - Cache search: Found 3 similar interactions (threshold: 0.85)
provider_selection - INFO - Provider selection: reasoning -> openai/gpt-4o (auto-categorized)
ai_requests - INFO - AI request: openai/gpt-4o for reasoning task
vector_db - INFO - Vector DB UPSERT: SUCCESS - interaction abc123
response_tracker - INFO - AI response: openai/gpt-4o/reasoning with RAG: True
```

---

## 🚀 Power User Tips

### Batch Questions for Efficiency
```bash
# Ask related questions in sequence to build context
python main.py ask "What is Docker?"
python main.py ask "How to write a Dockerfile?"  # Enhanced with Docker context
python main.py ask "Docker vs Kubernetes?"       # Enhanced with both contexts
```

### Smart Cache Management
```bash
# Prime the cache with common questions
python main.py ask "How to install Python packages?"
python main.py ask "How to create Python virtual environment?"
python main.py ask "How to write Python unit tests?"

# Team members get instant answers to these common questions
```

### Optimize for Your Workflow
```bash
# Create aliases for common usage patterns
alias ai-code='python main.py ask --category coding'
alias ai-quick='python main.py ask --provider gemini --model gemini-1.5-flash'
alias ai-think='python main.py ask --provider openai --model gpt-4o'

# Use them:
ai-code "Write a Python decorator"
ai-quick "What's 2+2?"  
ai-think "Explain quantum entanglement"
```

### Monitor Your Usage
```bash
# Check stats regularly to optimize costs
python main.py stats

# High cache hit rate = good cost savings
# Low cache hit rate = might need to adjust threshold or ask more similar questions
```

---

**🎯 Remember: Always activate your virtual environment before using the tool!**
```bash
source venv/bin/activate  # Essential first step
python main.py ask "Your question here"
```

This tool learns from your interactions and gets better over time. The more you use it, the more context it builds and the more intelligent responses become! 🚀