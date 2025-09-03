# AI CLI Tool 🤖

**Smart command-line AI assistant with caching, multi-provider support, and context memory.**

Get instant responses from cached queries, or enhanced responses using RAG from similar past conversations. Automatically routes to the best AI provider for each task type.

---

## 🚀 Quick Start

### 1. Setup (5 minutes)
```bash
# Clone and setup
git clone <repository-url>
cd ai-cli-tool
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start vector database
docker run -d -p 6333:6333 qdrant/qdrant

# Configure API keys
cp .env.example .env
# Edit .env with your API keys (need at least one):
# OPENAI_API_KEY=your_key
# CLAUDE_API_KEY=your_key  
# GEMINI_API_KEY=your_key
```

### 2. Basic Usage
```bash
# Always activate virtual environment first
source venv/bin/activate

# Ask any question - automatic provider selection
python main.py ask "What is machine learning?"

# Coding questions → routes to coding provider
python main.py ask "Write a Python function to reverse a string"

# Analysis questions → routes to reasoning provider  
python main.py ask "Compare Python vs JavaScript for web development"

# General conversation → routes to general provider
python main.py ask "Hello! How are you today?"
```

---

## 💡 Key Features in Action

### 🔄 **Semantic Caching** - Same/similar questions get instant responses
```bash
# First time - calls AI provider
$ python main.py ask "What is artificial intelligence?"
🤖 AI RESPONSE
🔗 Provider: gemini/gemini-1.5-flash | Category: general
💰 API Call: Made fresh request to gemini/gemini-1.5-flash

# Exact same question - instant cache hit  
$ python main.py ask "What is artificial intelligence?"
🔄 CACHE HIT (PERFECT) - Similarity: 1.000
📦 Original Source: gemini/gemini-1.5-flash | Category: general
⚡ Response Time: Instant (0 API calls)
```

### 📚 **RAG Context Enhancement** - Similar questions get enhanced with context
```bash
# Similar but different question gets RAG context
$ python main.py ask "Explain AI and machine learning differences"
🤖 AI RESPONSE + RAG Context
🔗 Provider: openai/gpt-4o | Category: reasoning  
💰 API Call: Made fresh request to openai/gpt-4o
📚 Enhanced: Using context from previous interactions
```

### 🎯 **Smart Provider Routing** - Right AI for each task type
```bash
# Coding → Claude (if configured) or OpenAI
python main.py ask "Debug this Python code: def hello(): print('hi'"

# Analysis → OpenAI GPT-4o
python main.py ask "Why is quantum computing important?"

# General chat → Gemini (fast and cost-effective)
python main.py ask "Good morning! Nice weather today"
```

---

## 🛠 Advanced Usage

### Force Specific Provider
```bash
python main.py ask "Write a function" --provider openai
python main.py ask "Explain quantum physics" --provider gemini --model gemini-1.5-pro
```

### Process Files
```bash
python main.py ask "What's in this image?" --image photo.jpg
python main.py ask "Summarize this document" --file report.pdf
```

### Cache Control
```bash
python main.py ask "Current weather" --no-cache           # Skip cache
python main.py ask "Question" --threshold 0.95            # Higher similarity needed
```

### Management Commands
```bash
python main.py providers    # List available AI providers
python main.py stats        # Usage statistics
python main.py configure    # Check configuration
```

---

## 📊 Understanding the Output

### Cache Hit Example:
```
🔄 CACHE HIT (PERFECT) - Similarity: 1.000
📦 Original Source: openai/gpt-4o | Category: reasoning
⚡ Response Time: Instant (0 API calls)
```

### AI Response Example:
```
🤖 AI RESPONSE + RAG Context  
🔗 Provider: anthropic/claude-3-5-sonnet | Category: coding
💰 API Call: Made fresh request to anthropic/claude-3-5-sonnet
📚 Enhanced: Using context from previous interactions
```

### What This Tells You:
- **🔄/🤖** = Cache hit vs Fresh AI response
- **Provider Used** = Which AI service answered
- **Category** = How the question was classified (reasoning/coding/general)
- **RAG Context** = Whether similar past conversations enhanced the response
- **Similarity Score** = How closely it matched cached responses (0-1)

---

## 🔧 Configuration

### Essential Environment Variables
```bash
# At least one API key required:
OPENAI_API_KEY=your_openai_key
CLAUDE_API_KEY=your_claude_key  
GEMINI_API_KEY=your_gemini_key

# Vector database (auto-configured):
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6333
```

### Customizing Provider Routing
```bash
# Override default provider assignments:
REASONING_PROVIDER=openai       # For analysis questions
CODING_PROVIDER=anthropic       # For programming questions  
GENERAL_PROVIDER=gemini         # For general conversation

# Specify models:
REASONING_MODEL=gpt-4o
CODING_MODEL=claude-3-5-sonnet-20241022
GENERAL_MODEL=gemini-1.5-flash
```

---

## 🎯 Use Cases

### **Developer Workflow**
```bash
# Code help with context memory
python main.py ask "Write a Python REST API using FastAPI"
python main.py ask "Add authentication to that API"  # Uses previous context
python main.py ask "Write tests for the auth endpoints"  # Enhanced with context
```

### **Learning & Research**
```bash
# Build knowledge progressively  
python main.py ask "What is machine learning?"
python main.py ask "What are neural networks?"
python main.py ask "How do neural networks relate to machine learning?"  # Gets context
```

### **Instant Answers**
```bash
# Frequently asked questions get cached
python main.py ask "How to install Python packages?"  # First time: calls AI
python main.py ask "How to install Python packages?"  # Second time: instant cache
```

---

## ⚡ Performance Benefits

- **⚡ Instant Responses** - Cached answers in milliseconds
- **💰 Cost Savings** - Avoid duplicate API calls  
- **🧠 Context Memory** - Enhanced responses using conversation history
- **🎯 Smart Routing** - Best AI for each question type
- **📈 Usage Analytics** - Track your AI usage patterns

---

## 🚨 Troubleshooting

### Quick Fixes
```bash
# Qdrant not running?
docker run -d -p 6333:6333 qdrant/qdrant

# Missing dependencies?
source venv/bin/activate
pip install -r requirements.txt

# API key issues?
python main.py configure  # Check configuration status

# Clear cache?
rm -rf qdrant_storage/    # Reset all cached data
```

### Getting Help
```bash
python main.py --help              # Main help
python main.py ask --help          # Ask command options
python main.py stats               # Usage statistics
```

---

## 📚 More Information

- **[Complete Documentation](main.md)** - Comprehensive guide with all features
- **[Architecture Details](main.md#architecture-details)** - How it works under the hood
- **[Development Guide](main.md#development)** - Contributing and extending
- **[Docker Deployment](main.md#docker-deployment)** - Production setup

---

## 🌟 Why Use This Tool?

✅ **Saves Time** - Instant answers for repeated questions  
✅ **Saves Money** - Cached responses reduce API costs  
✅ **Context Aware** - Remembers and uses conversation history  
✅ **Multi-Provider** - Best AI for each task automatically  
✅ **Developer Friendly** - CLI tool that integrates into workflows  
✅ **Open Source** - Fully customizable and extensible  

**Get started in 5 minutes and experience AI with memory! 🚀**