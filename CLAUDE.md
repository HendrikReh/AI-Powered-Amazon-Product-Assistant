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

### Ollama Local LLM Setup
```bash
# Install Ollama (if not already installed)
# Visit https://ollama.com for installation instructions

# Pull and run your model (example with Gemma 3 4B)
ollama pull gemma3n:e4b
ollama run gemma3n:e4b

# Verify Ollama is running
ollama list

# The application will automatically connect to Ollama at http://localhost:11434
# Configure OLLAMA_BASE_URL in .env if using different host/port
```

### Running Applications
```bash
# Run enhanced Streamlit chatbot interface with RAG and tab-based UI
uv run streamlit run src/chatbot-ui/streamlit_app.py
# OR using Makefile
make run-streamlit

# Features:
# - 🔧 Configuration Tab: Model settings, RAG configuration, system status
# - 💬 Query Tab: Enhanced chat interface with smart suggestions and filters
# - 📊 Monitoring Tab: Real-time performance metrics and Enhanced Weave Tracing v2.0

# Run data processing notebooks
uv run jupyter notebook notebooks/data_preprocessing.ipynb

# Run visualization notebooks  
uv run jupyter notebook notebooks/data_visualization.ipynb

# Test RAG system functionality (auto-detects local vs Docker implementation)
uv run python test_rag_system.py

# Initialize vector database manually (local development with GTE-large)
uv run python src/rag/vector_db.py
```

### Evaluation & Testing
```bash
# Create evaluation dataset with ground truth examples
uv run python run_evaluation.py --create-dataset

# Run single query evaluation for testing
uv run python run_evaluation.py --single-query "What are iPhone charger features?" --mock-llm

# Run full RAG system evaluation using Weave
uv run python run_evaluation.py --mock-llm --project-name "rag-evaluation"

# Run evaluation with custom dataset
uv run python run_evaluation.py --dataset-path "data/evaluation/custom_dataset.json" --mock-llm

# Generate and evaluate synthetic test data
uv run python run_synthetic_evaluation.py --synthetic-only --num-synthetic 50

# Create mixed dataset (original + synthetic)
uv run python run_synthetic_evaluation.py --mixed-dataset --save-datasets

# Run synthetic data examples and demonstrations
uv run python examples/synthetic_data_examples.py
```

### Docker Operations
```bash
# Build Docker containers (Streamlit + ChromaDB service)
make build-docker-streamlit

# Run containerized services (uses vector_db_docker.py with optimized embeddings)
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
- **RAG System**: `src/rag/` contains dual vector database implementations and query processing
- **Configuration**: `src/chatbot-ui/core/config.py` manages API keys and settings via Pydantic
- **Processed Data**: `data/processed/` contains cleaned datasets ready for RAG implementation
- **Vector Database**: Dual storage approach (local files for dev, service for production)

### Dual Vector Database Architecture
- **Automatic Detection**: System selects implementation based on `CHROMA_HOST` environment variable
- **Local Development** (`vector_db.py`):
  - Uses `chromadb.PersistentClient` with local file storage
  - GTE-large embeddings (1024 dimensions) for maximum search quality
  - Best for development, testing, and debugging
- **Docker Production** (`vector_db_docker.py`):
  - Uses `chromadb.HttpClient` connecting to external ChromaDB service
  - Default embeddings (384 dimensions) for container optimization
  - Fallback to local storage if service unavailable
  - Saves 670MB in container images
- **Seamless Switching**: `query_processor.py` automatically imports correct implementation
- **Complete comparison guide**: See `docs/LOCAL_VS_DOCKER.md` for detailed analysis

### Multi-Provider LLM Support
The chatbot supports four LLM providers with different parameter compatibility:
- **OpenAI**: GPT-4o, GPT-4o-mini (supports temperature, max_tokens, top_p)
- **Groq**: Llama-3.3-70b-versatile (supports temperature, max_tokens, top_p)  
- **Google**: Gemini-2.0-flash (supports all parameters: temperature, max_tokens, top_p, top_k)
- **Ollama**: Local models (supports temperature, max_tokens) - requires Ollama installation

### Environment Configuration
Required API keys in `.env` file:
```
OPENAI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here  
GOOGLE_API_KEY=your_key_here
WANDB_API_KEY=your_key_here  # Optional for Weave tracing

# Ollama configuration (optional)
OLLAMA_BASE_URL=http://localhost:11434  # Default Ollama URL

# Docker environment detection (optional)
CHROMA_HOST=chromadb        # Triggers Docker vector database mode
CHROMA_PORT=8000           # ChromaDB service port
```

#### Vector Database Selection Logic
```python
# Automatic implementation selection in query_processor.py
is_docker = os.getenv("CHROMA_HOST") is not None

if is_docker:
    # Uses vector_db_docker.py with HTTP client and default embeddings
    from .vector_db_docker import ElectronicsVectorDBDocker as ElectronicsVectorDB
else:
    # Uses vector_db.py with persistent client and GTE-large embeddings
    from .vector_db import ElectronicsVectorDB
```

### Data Pipeline Architecture
1. **Raw Data**: Large JSONL files (Electronics.jsonl, meta_Electronics.jsonl)
2. **Processing**: Jupyter notebooks extract top 1000 products with 20K reviews
3. **RAG-Ready Output**: `electronics_rag_documents.jsonl` formatted for retrieval systems
4. **Vector Database**: ChromaDB ingests documents with semantic embeddings and metadata
5. **Visualization**: Comprehensive EDA with temporal, category, and rating analysis

### RAG System Architecture
- **Dual Vector Database**: Automatic environment detection for optimal implementation
  - **Local Development**: `vector_db.py` with GTE-large embeddings (1024D) for maximum quality
  - **Docker Production**: `vector_db_docker.py` with default embeddings (384D) for container optimization
- **Environment Detection**: Automatic selection via `CHROMA_HOST` environment variable
- **Document Types**: Products and review summaries with structured metadata
- **Query Processing**: Intelligent query type detection and context-aware retrieval
- **Search Capabilities**: Semantic search, metadata filtering, hybrid queries
- **Integration**: Seamless integration with existing LLM providers via enhanced prompts

### Enhanced Weave Tracing Integration v2.0
- **Three-Tier Architecture**: Context propagation (Tier 1), vector performance monitoring (Tier 2), business intelligence (Tier 3)
- **Context Propagation**: Session-based trace contexts with unique IDs propagated across all operations
- **Vector Performance Monitoring**: Embedding generation timing, search performance, cache hit rates, and result quality metrics
- **Business Intelligence Tracing**: User journey tracking, intent classification, satisfaction prediction, and conversion analysis
- **Multi-Operation Tracing**: Granular `@weave.op()` decorators on 53+ operations across 8 Python files
- **Real-Time Analytics**: Live performance metrics, business KPIs, and user behavior insights in Streamlit UI
- **Performance Optimization**: Embedding time tracking, search quality analysis, and cache performance monitoring
- **User Journey Intelligence**: User type classification (Researcher, Buyer, Casual, Troubleshooter) and journey stage tracking
- **Business Metrics**: Conversion potential, satisfaction prediction, recommendation effectiveness, and feature usage analytics
- **Advanced Query Analysis**: Intent classification, complexity scoring, specificity measurement, and product focus extraction
- **Session Management**: Cross-conversation context with user behavior patterns and pain point detection
- **Enhanced UI Monitoring**: Business intelligence dashboard, vector performance metrics, and session analytics
- **Production-Ready Fallbacks**: Zero single points of failure with comprehensive graceful degradation mechanisms
- **Comprehensive Coverage**: End-to-end pipeline monitoring from query analysis to response generation with business insights
- **📖 Complete tracing guide**: See `docs/WEAVE_TRACING_GUIDE.md` for detailed implementation and troubleshooting

### Package Management
- **uv**: Modern Python package manager used instead of pip/conda
- **Dependencies**: Defined in `pyproject.toml` with specific versions for reproducibility
- **Jupyter Integration**: Custom kernel installation required for notebook work
- **RAG Dependencies**: ChromaDB for vector database, sentence-transformers for GTE embeddings (local)
- **Container Optimization**: Docker implementation uses ChromaDB default embeddings for efficiency

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

### RAG System Evaluation
The project includes a comprehensive evaluation framework using Weave for systematic testing:
```bash
# Create evaluation dataset (14 examples across query types)
uv run python run_evaluation.py --create-dataset

# Quick single query test
uv run python run_evaluation.py --single-query "Your test query" --mock-llm

# Full evaluation with Weave integration
uv run python run_evaluation.py --mock-llm --project-name "rag-evaluation"
```

### Synthetic Test Data Generation
The project includes advanced synthetic data generation for comprehensive testing:
```bash
# Generate synthetic-only evaluation
uv run python run_synthetic_evaluation.py --synthetic-only --num-synthetic 30

# Create mixed dataset (original + synthetic)
uv run python run_synthetic_evaluation.py --mixed-dataset --save-datasets

# Custom difficulty distribution
uv run python run_synthetic_evaluation.py --difficulty-easy 0.2 --difficulty-hard 0.4

# Run comprehensive examples
uv run python examples/synthetic_data_examples.py
```

**Evaluation Metrics:**
- **Relevance**: Topic coverage and query alignment (0-1 score)
- **Accuracy**: Factual correctness and product mention accuracy
- **Completeness**: Response depth and length appropriateness
- **Factuality**: Contradiction detection and numerical claim verification
- **Quality**: Clarity, helpfulness, and coherence assessment

**Dataset Coverage:**
- 14 evaluation examples across 6 query types
- Difficulties: Easy (3), Medium (6), Hard (5)
- Query types: product_info, product_reviews, product_complaints, product_comparison, product_recommendation, use_case

### Synthetic Data Generation Framework
Advanced synthetic test data creation with multiple techniques:
- **Template-Based Generation**: 6 query types across 10 product categories
- **Variation Techniques**: Rephrase, specificity, context, perspective transformations
- **Quality Analysis**: Uniqueness, length distribution, topic coverage validation
- **Weave Integration**: Full traceability and metadata tracking
- **Mixed Datasets**: Combine original and synthetic for robust testing
- **Configurable Parameters**: Difficulty distribution, query type weights, generation methods
- **Best Practices**: Automated quality validation and comparative analysis

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

## Synthetic Data Generation
- **Template-Based Generation**: 6 query types with realistic product categories
- **Variation Techniques**: 4 approaches for query diversity (rephrase, specificity, context, perspective)
- **Quality Validation**: Automated analysis of uniqueness, length, and topic coverage
- **Weave Integration**: Full operation tracking and metadata capture
- **Mixed Datasets**: Combine original and synthetic data for comprehensive testing
- **Configurable Generation**: Customizable difficulty distributions and query type weights
- **Best Practices Implementation**: Following synthetic data generation standards
- **Production Testing**: Automated test case creation for systematic evaluation
- **Complete guide**: See `docs/SYNTHETIC_DATA.md` for detailed implementation

## Docker Notes
- **Dual-architecture deployment**: Automatically uses `vector_db_docker.py` for optimized containers
- **Multi-service setup**: Streamlit app + ChromaDB vector database service
- **Container optimization**: Uses default embeddings (saves 670MB vs GTE-large)
- **Service connection**: HTTP client connects to external ChromaDB service
- **Fallback mechanism**: Graceful degradation to local storage if service unavailable
- **Non-root security**: Uses dedicated app user for container security
- **Persistent storage**: ChromaDB data persists in Docker volumes
- **Environment integration**: Mounts `.env` file for API key access, detects `CHROMA_HOST`
- **Automatic initialization**: Vector database auto-populates on first run
- **Health checks**: Built-in service connectivity verification
- **Network isolation**: Services communicate via dedicated Docker network
- **TTY compatibility**: Handles production deployment issues (see DOCKER_TTY_FIXES.md)
- **Weave tracing**: Full compatibility in containerized environments
- **Implementation details**: See `docs/LOCAL_VS_DOCKER.md` for comprehensive comparison

## Memory Updates

### Enhanced Tracing v2.0 Implementation (Sprint 1)
- **Three-Tier Architecture**: Implemented context propagation, vector performance monitoring, and business intelligence tracking
- **53+ Weave Operations**: Comprehensive instrumentation across 8 Python files with granular tracing
- **Zero Single Points of Failure**: Production-ready fallback mechanisms across all components
- **Business Intelligence**: User journey tracking, intent classification, conversion analysis, and satisfaction prediction
- **Real-Time Analytics**: Live performance metrics and business KPIs in enhanced Streamlit dashboard
- **Production Documentation**: Complete implementation guide in `docs/WEAVE_TRACING_GUIDE.md` and Sprint 1 summary in `docs/SPRINT_1.md`
- **Fallback Coverage**: Comprehensive error handling for imports, vector database, performance monitoring, business intelligence, Weave tracing, and LLM providers
- **Session Management**: Cross-conversation context with user behavior patterns and pain point detection
- **Vector Performance**: Embedding optimization, cache analytics, and search quality monitoring
- **User Experience**: Enhanced Streamlit interface with business intelligence insights and real-time monitoring

### Code Management
- Updated code management workflow to include README.md and PROJECT_CANVAS.md updates alongside CLAUDE.md
- Ensured consistent memory tracking across project documentation files
- Comprehensive Sprint 1 documentation reflecting Enhanced Tracing v2.0 achievements

### UI/UX Design
- The Streamlit app follows professional UX typography standards with a clear visual hierarchy that guides users naturally through the interface