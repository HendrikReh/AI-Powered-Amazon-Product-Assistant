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
- **Embedding Model**: GTE-large (thenlper/gte-large) with 1024-dimensional embeddings
- **Document Types**: Products and review summaries with structured metadata
- **Query Processing**: Intelligent query type detection and context-aware retrieval
- **Search Capabilities**: Semantic search, metadata filtering, hybrid queries
- **Integration**: Seamless integration with existing LLM providers via enhanced prompts

### Weave Tracing Integration
- **Enhanced Multi-Operation Tracing**: Granular `@weave.op()` decorators on individual pipeline components
- **Initialization Tracing**: `initialize_weave_tracing()` and `initialize_rag_processor()` with status tracking
- **RAG Enhancement Tracing**: `apply_rag_enhancement()` with timing, context metrics, and error handling
- **Provider-Specific LLM Tracing**: `call_llm_provider()` with request metadata, response metrics, and error types
- **Comprehensive Pipeline Monitoring**: Enhanced `run_llm()` with end-to-end timing and component breakdown
- **Real-Time UI Feedback**: Sidebar displays processing times, retrieval counts, and operation status
- **Detailed Metadata Capture**: Query types, extracted terms, context quality, and provider-specific parameters
- **Performance Analytics**: Sub-operation timing (RAG vs LLM), character counts, and success/failure rates
- **Error Classification**: Structured error handling with error types, fallback strategies, and user feedback
- **W&B Dashboard Integration**: Rich trace data with nested operations and comprehensive metadata
- **Production Monitoring**: Zero-impact tracing with graceful degradation and optional activation
- **Docker Compatibility**: Full containerized deployment support with trace persistence

### Package Management
- **uv**: Modern Python package manager used instead of pip/conda
- **Dependencies**: Defined in `pyproject.toml` with specific versions for reproducibility
- **Jupyter Integration**: Custom kernel installation required for notebook work
- **RAG Dependencies**: ChromaDB for vector database, sentence-transformers for GTE embeddings

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
- **Response Time**: ~0.29 seconds average per query (improved accuracy with GTE)
- **Database Size**: 2,000 documents (1,000 products + 1,000 reviews)
- **Success Rate**: 100% across all query types
- **Embedding Model**: GTE-large (1024 dimensions) for superior semantic understanding
- **Retrieval Quality**: Enhanced contextual product and review matching

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