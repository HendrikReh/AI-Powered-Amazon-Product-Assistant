# CLAUDE.md - Project Changes Log

This file tracks major edits and changes made to the AI-Powered Amazon Product Assistant project.

## Recent Updates

### 2025-01-XX - Weave Tracing Integration
- **Added LLM call tracing** via Weave/Weights & Biases
- **Implemented configuration sliders** for temperature, max_tokens, top_p, top_k
- **Multi-provider support** for OpenAI, Groq, and Google Gemini
- **Docker compatibility fixes** for TTY-less environments
- **Enhanced error handling** with graceful fallback when tracing unavailable

#### Files Modified:
- `src/chatbot-ui/streamlit_app.py` - Added Weave initialization and @weave.op() decorator
- `src/chatbot-ui/core/config.py` - Added optional WANDB_API_KEY configuration
- `README.md` - Updated with tracing setup instructions and technical stack
- `Project_Canvas.md` - Added Task 4 for LLM Tracing & Monitoring

#### Technical Implementation:
- **Weave Integration**: `@weave.op()` decorator on `run_llm()` function
- **Authentication**: Automatic W&B login with environment variable
- **Configuration Tracking**: All LLM parameters logged with each call
- **Performance Monitoring**: Call latency and token usage tracking
- **Project Organization**: All traces logged to "Bootcamp" project

### Previous Changes
- Data processing pipeline implementation
- Comprehensive EDA with visualization notebooks
- Streamlit UI with multi-provider LLM support
- Docker containerization setup
- Project structure and documentation

## Setup Instructions

### Environment Variables Required:
```bash
# Required for LLM functionality
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key  
GOOGLE_API_KEY=your_google_key

# Optional for tracing
WANDB_API_KEY=your_wandb_key
```

### Quick Start:
```bash
# Install dependencies
uv sync

# Run Streamlit app
uv run streamlit run src/chatbot-ui/streamlit_app.py

# Or use Docker
make build-docker-streamlit && make run-docker-streamlit
```

## Project Status
- ✅ Data processing and EDA complete
- ✅ Streamlit UI with configurable parameters
- ✅ Multi-provider LLM support (OpenAI, Groq, Google)
- ✅ Weave tracing integration
- ✅ Docker deployment ready
- 🔄 RAG implementation (next phase) 