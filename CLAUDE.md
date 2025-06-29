# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Dependencies & Environment
```bash
# Install all dependencies
uv sync

# Set up Jupyter kernel for notebooks
uv run python -m ipykernel install --user --name ai-product-assistant
```

### Running Applications
```bash
# Run Streamlit chatbot interface with RAG
uv run streamlit run src/chatbot-ui/streamlit_app.py
# OR using Makefile
make run-streamlit

# Run data processing notebooks
uv run jupyter notebook notebooks/data_preprocessing.ipynb

# Run visualization notebooks  
uv run jupyter notebook notebooks/data_visualization.ipynb

# Test RAG system functionality
uv run python test_rag_system.py

# Initialize vector database manually
uv run python src/rag/vector_db.py
```

### Docker Operations
```bash
# Build Docker containers (Streamlit + ChromaDB)
make build-docker-streamlit

# Run containerized services (app + vector database)
make run-docker-streamlit

# View application logs
make logs-docker-streamlit

# Stop all services
make stop-docker-streamlit

# Restart services
make restart-docker-streamlit

# Clean up containers and volumes
make clean-docker
```

### Notebook Management
```bash
# Clear notebook outputs for clean commits
make clean-notebook-outputs
```

## Project Architecture

### Core Structure
- **Data Processing**: `notebooks/` contains Jupyter notebooks for EDA and data preprocessing
- **Chatbot Interface**: `src/chatbot-ui/` contains Streamlit-based chat application with RAG integration
- **RAG System**: `src/rag/` contains vector database and query processing components
- **Configuration**: `src/chatbot-ui/core/config.py` manages API keys and settings via Pydantic
- **Processed Data**: `data/processed/` contains cleaned datasets ready for RAG implementation
- **Vector Database**: `data/chroma_db/` contains ChromaDB persistent storage for embeddings

### Multi-Provider LLM Support
The chatbot supports three LLM providers with different parameter compatibility:
- **OpenAI**: GPT-4o, GPT-4o-mini (supports temperature, max_tokens, top_p)
- **Groq**: Llama-3.3-70b-versatile (supports temperature, max_tokens, top_p)  
- **Google**: Gemini-2.0-flash (supports all parameters: temperature, max_tokens, top_p, top_k)

### Environment Configuration
Required API keys in `.env` file:
```
OPENAI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here  
GOOGLE_API_KEY=your_key_here
WANDB_API_KEY=your_key_here  # Optional for Weave tracing
```

### Data Pipeline Architecture
1. **Raw Data**: Large JSONL files (Electronics.jsonl, meta_Electronics.jsonl)
2. **Processing**: Jupyter notebooks extract top 1000 products with 20K reviews
3. **RAG-Ready Output**: `electronics_rag_documents.jsonl` formatted for retrieval systems
4. **Vector Database**: ChromaDB ingests documents with semantic embeddings and metadata
5. **Visualization**: Comprehensive EDA with temporal, category, and rating analysis

### RAG System Architecture
- **Vector Database**: ChromaDB with persistent storage in `data/chroma_db/`
- **Document Types**: Products and review summaries with structured metadata
- **Query Processing**: Intelligent query type detection and context-aware retrieval
- **Search Capabilities**: Semantic search, metadata filtering, hybrid queries
- **Integration**: Seamless integration with existing LLM providers via enhanced prompts

### Weave Tracing Integration
- **Tracing Decorator**: `@weave.op()` on `run_llm()` function tracks all LLM calls
- **W&B Project**: Traces logged to "Bootcamp" project
- **Configuration Tracking**: All model parameters (temperature, tokens, top_p, top_k) captured
- **Error Handling**: Graceful degradation when WANDB_API_KEY unavailable
- **Docker Support**: Special TTY handling for containerized environments

### Package Management
- **uv**: Modern Python package manager used instead of pip/conda
- **Dependencies**: Defined in `pyproject.toml` with specific versions for reproducibility
- **Jupyter Integration**: Custom kernel installation required for notebook work
- **RAG Dependencies**: ChromaDB for vector database, automatic embedding generation

## Key Data Insights
- **Dataset Scale**: 1,000 electronics products, 20,000 reviews spanning 2003-2023
- **Review Quality**: 4.29/5.0 average rating, 68.2% five-star reviews
- **Price Range**: $3.49-$1,175.35 (median $26.99), heavily skewed toward budget items
- **Categories**: 312 unique categories with 3.8 average hierarchy depth
- **Temporal Patterns**: Tuesday most active day, January most active month

## Testing & Quality
Run the data processing notebooks to verify the pipeline:
```bash
uv run jupyter notebook notebooks/verify_api_keys.ipynb  # Test API configuration
uv run python test_rag_system.py  # Comprehensive RAG system testing
```

## RAG Query Examples
The system handles various query types with intelligent context retrieval:
- **Product Reviews**: "What do people say about iPhone charger cables?"
- **Use Case Evaluation**: "Is the Fire TV good for streaming?"
- **Product Comparison**: "Compare ethernet cables and USB cables"
- **Complaint Analysis**: "What are the main complaints about laptop backpacks?"
- **Budget Recommendations**: "Recommend a budget-friendly tablet under $100"

## RAG Performance Metrics
- **Response Time**: ~0.16 seconds average per query
- **Database Size**: 2,000 documents (1,000 products + 1,000 reviews)
- **Success Rate**: 100% across all query types
- **Retrieval Quality**: Contextual product and review matching

## Docker Notes
- **Multi-service setup**: Streamlit app + ChromaDB vector database
- **Non-root security**: Uses dedicated app user for container security
- **Persistent storage**: ChromaDB data persists in Docker volumes
- **Environment integration**: Mounts `.env` file for API key access
- **Automatic initialization**: Vector database auto-populates on first run
- **Health checks**: Built-in service connectivity verification
- **Network isolation**: Services communicate via dedicated Docker network
- **TTY compatibility**: Handles production deployment issues (see DOCKER_TTY_FIXES.md)
- **Weave tracing**: Full compatibility in containerized environments