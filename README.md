# AI-Powered Amazon Product Assistant (B2C only)

An end-to-end AI engineering project that builds an intelligent product recommendation and analysis system using Amazon Electronics dataset with a complete RAG implementation. This capstone project demonstrates modern AI engineering practices including data processing, visualization, vector databases, and retrieval-augmented generation (RAG).

**Course**: End-to-End AI Engineering Bootcamp ([Maven](https://maven.com/swirl-ai/end-to-end-ai-engineering))

## Features

- **Data Processing Pipeline**: Automated processing of large-scale Amazon product and review data
- **Interactive Visualizations**: Comprehensive analysis dashboards with temporal trends, category insights, and rating patterns  
- **Complete RAG System**: Vector database with ChromaDB, intelligent query processing, and context-aware retrieval
- **Advanced Streamlit UI**: Professional tab-based interface with smart query suggestions, real-time monitoring, and enhanced response visualization
- **Multi-Provider Support**: Compatible with OpenAI, Groq, and Google Gemini models
- **Vector Database**: ChromaDB-powered semantic search with GTE-large embeddings, metadata filtering and hybrid queries
- **Query Intelligence**: Automatic query type detection for product reviews, comparisons, complaints, and recommendations
- **RAG Evaluation Framework**: Comprehensive evaluation system with Weave integration, 5 core metrics, and 14 test examples
- **Synthetic Test Data**: Advanced synthetic data generation with template-based queries, variation techniques, and quality analysis
- **Production Testing**: Automated test case generation with configurable difficulty distributions and Weave traceability
- **Optimized Weave Tracing**: Production-ready AI pipeline monitoring with efficient session-based initialization, zero-redundancy design, and comprehensive analytics

## Out-of-Scope (B2B Scope)

- contractual pricing
- account-specific catalogs
- procurement compliance
- multi-user workflows (approvers, requisitioners, etc.)
- Bulk ordering, BOM-style inputs, or quote-based negotiation are not captured
- ERP integration, punchout catalogs (OCI, cXML)
- product taxonomies (e.g. ETIM, UNSPSC)
  
## Dataset Overview

**Source**: Amazon Reviews 2023 - Electronics Category
- **Products**: 1,000 carefully selected electronics products
- **Reviews**: 20,000 customer reviews (10-20 reviews per product)
- **Date Range**: 2003-2023 (20 years of review data)
- **Categories**: Comprehensive electronics categories with hierarchical structure

### Key Statistics
- Average reviews per product: 20
- Review rating distribution: 4.2/5.0 average
- Most active day: Tuesday (3,068 reviews)
- Most active month: January (2,283 reviews)
- Recent activity: 37.8% of reviews from 2020 onwards
- **Embedding Model**: GTE-large (1024 dimensions) for superior semantic search

## Setup & Installation

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (optional, for containerized deployment)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AI-Powered-Amazon-Product-Assistant
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Configure environment variables**
   ```bash
   # Create .env file with your API keys
   cp .env.example .env  # if available, or create manually
   
   # Required for chatbot functionality
   echo "OPENAI_API_KEY=your_openai_key" >> .env
   echo "GROQ_API_KEY=your_groq_key" >> .env  
   echo "GOOGLE_API_KEY=your_google_key" >> .env
   
   # Optional for Weave tracing
   echo "WANDB_API_KEY=your_wandb_key" >> .env
   ```

4. **Set up Jupyter kernel**
   ```bash
   uv run python -m ipykernel install --user --name ai-product-assistant
   ```

5. **Run data processing** (if needed)
   ```bash
   uv run jupyter notebook notebooks/data_preprocessing.ipynb
   ```

6. **Launch applications**
   ```bash
   # Visualization dashboard
   uv run jupyter notebook notebooks/data_visualization.ipynb
   
   # Enhanced Streamlit chatbot interface with tab-based UI and RAG
   uv run streamlit run src/chatbot-ui/streamlit_app.py
   
   # Test RAG system functionality
   uv run python test_rag_system.py
   
   # Run RAG evaluation framework
   uv run python run_evaluation.py --create-dataset
   uv run python run_evaluation.py --mock-llm --project-name "rag-evaluation"
   
   # Generate and evaluate synthetic test data
   uv run python run_synthetic_evaluation.py --synthetic-only --num-synthetic 50
   
   # Create mixed dataset (original + synthetic)
   uv run python run_synthetic_evaluation.py --mixed-dataset --save-datasets
   
   # Run synthetic data examples
   uv run python examples/synthetic_data_examples.py
   ```

### Docker Deployment (with ChromaDB)

```bash
# Build the containers
make build-docker-streamlit

# Run both Streamlit app and ChromaDB service
make run-docker-streamlit

# View logs
make logs-docker-streamlit

# Stop services
make stop-docker-streamlit

# Restart services
make restart-docker-streamlit
```

**Docker Services:**
- **Streamlit App**: http://localhost:8501 (Enhanced tab-based interface)
- **ChromaDB Service**: http://localhost:8000
- **Persistent Storage**: Vector data persisted in Docker volume

### Enhanced Streamlit Interface

The application features a professional tab-based interface designed for optimal user experience:

**🔧 Configuration Tab:**
- **System Status**: Real-time monitoring of Weave tracing and RAG system initialization
- **Model Selection**: Choose from OpenAI (GPT-4o, GPT-4o-mini), Groq (Llama-3.3-70b), or Google (Gemini-2.0-flash)
- **Parameter Controls**: Fine-tune temperature, max tokens, top-p, and top-k with provider-specific support
- **RAG Configuration**: Enable/disable RAG with customizable product and review limits

**💬 Query Tab:**
- **Smart Examples**: 12+ categorized example queries across 6 use cases (Product Info, Reviews, Comparisons, Complaints, Recommendations, Use Cases)
- **Query History**: Access and reuse your last 10 queries with one click
- **Auto-Suggestions**: Get intelligent query completions based on partial input (3+ characters)
- **Quick Filters**: Filter by query type, product category, and price range
- **Enhanced Input**: Dynamic placeholders and integrated filter display

**📊 Monitoring Tab:**
- **Session Statistics**: Track message counts, query history, and usage patterns
- **Real-Time Performance**: View RAG vs LLM processing times with percentage breakdown
- **RAG Analytics**: Monitor retrieved products/reviews and query type detection
- **System Health**: Check API configurations and system component status
- **Weave Integration**: Direct links to W&B dashboard for detailed trace analysis

### Enhanced Weave Tracing Setup

The application includes comprehensive Weave tracing for end-to-end AI pipeline monitoring and performance analysis.

1. **Get W&B API Key**
   - Sign up at [wandb.ai](https://wandb.ai)
   - Get your API key from [User Settings](https://wandb.ai/authorize)

2. **Configure Tracing**
   ```bash
   # Add to your .env file
   echo "WANDB_API_KEY=your_wandb_api_key" >> .env
   ```

3. **Enhanced Features Tracked**
   - **Optimized Initialization**: Single-session setup with session state management
   - **RAG Pipeline Tracing**: Query analysis, context building, and retrieval metrics
   - **LLM Provider Tracking**: Detailed request/response metadata for OpenAI, Groq, and Google
   - **Performance Analytics**: Sub-operation timing, character counts, and success rates
   - **Error Classification**: Structured error handling with types and fallback strategies
   - **Real-Time UI Feedback**: Processing times and operation status in sidebar
   - **Context Quality Metrics**: Query type detection, extracted terms, and retrieval effectiveness
   - **Trace Optimization**: Eliminated redundant calls and duplicate initialization

4. **Optimized Operation Monitoring**
   - **Session-Based Initialization**: Single setup per session via `@st.cache_resource`
   - **Consolidated Tracing**: Primary trace points at key pipeline stages
   - **RAG Enhancement Metrics**: Query processing timing and context quality
   - **LLM Provider Analytics**: Request/response data with performance breakdown
   - **End-to-End Pipeline**: Complete timing analysis from query to response
   - **Zero-Redundancy Design**: Eliminated multiple trace calls for same operations

5. **Production-Ready Monitoring**
   - **Optimized Trace Volume**: Meaningful traces without duplication
   - **Session State Management**: Prevents repeated initialization calls
   - **Clean Dashboard Data**: Visit your [W&B dashboard](https://wandb.ai) for organized traces
   - **Performance Insights**: Navigate to "Bootcamp" project for analytics
   - **Error Tracking**: Structured error handling with fallback strategies
   - **Real-Time Feedback**: Processing times displayed in Streamlit sidebar

## Project Structure

```
AI-Powered-Amazon-Product-Assistant/
├── 📁 data/
│   ├── Electronics.jsonl                              # Raw review data (25GB)
│   ├── meta_Electronics.jsonl                         # Raw product metadata (4.9GB)
│   ├── 📁 processed/
│   │   ├── electronics_top1000_products.jsonl         # 1,000 product records
│   │   ├── electronics_top1000_products_reviews.jsonl # 20,000 review records
│   │   ├── electronics_rag_documents.jsonl            # 2,000 RAG-optimized documents
│   │   ├── dataset_summary.json                       # Processing metadata
│   │   └── README.md                                  # Data documentation
│   └── 📁 chroma_db/                                  # Vector database storage (local)
├── 📁 notebooks/
│   ├── data_preprocessing.ipynb                       # Data processing pipeline
│   ├── data_visualization.ipynb                       # Interactive EDA template
│   ├── data_visualization_executed.ipynb              # Complete analysis results
│   ├── verify_api_keys.ipynb                         # API configuration testing
│   └── README.md                                      # Notebook documentation
├── 📁 src/
│   ├── 📁 chatbot-ui/
│   │   ├── 📁 core/
│   │   │   └── config.py                              # Multi-provider configuration
│   │   └── streamlit_app.py                          # Main chatbot interface with RAG
│   ├── 📁 rag/
│   │   ├── vector_db.py                               # ChromaDB vector database (local, GTE-large)
│   │   ├── vector_db_docker.py                       # ChromaDB vector database (Docker, optimized)
│   │   └── query_processor.py                        # RAG query processing (auto-selects implementation)
│   └── 📁 evaluation/
│       ├── __init__.py                                # Evaluation module interface
│       ├── evaluator.py                               # Main RAG evaluator using Weave
│       ├── dataset.py                                 # Evaluation dataset creation and management
│       ├── scorers.py                                 # Scoring functions for 5 core metrics
│       └── synthetic_data_generator.py               # Advanced synthetic test data generation
├── 📁 examples/
│   └── synthetic_data_examples.py                    # Synthetic data usage demonstrations
├── 📁 docs/                                          # Technical documentation
│   ├── SPRINT_0.md                                   # Sprint 0 foundation summary (June 28, 2025)
│   ├── SPRINT_1.md                                   # Sprint 1 RAG implementation summary
│   ├── CHROMA.md                                      # ChromaDB integration guide
│   ├── LOCAL_VS_DOCKER.md                            # Local vs Docker implementation comparison
│   ├── WEAVE_TRACING_GUIDE.md                         # LLM tracing & monitoring guide
│   ├── EVALUATIONS.md                                # RAG evaluation framework documentation
│   ├── SYNTHETIC_DATA.md                             # Synthetic test data generation guide
│   ├── DASHBOARD_METRICS.md                          # Dashboard metrics interpretation & implementation
│   ├── GEMINI_MESSAGE_HANDLING.md                    # Google Gemini integration guide
│   └── DOCKER_TTY_FIXES.md                           # Container deployment fixes
├── 📄 pyproject.toml                                  # uv dependencies & config
├── 📄 docker-compose.yml                              # Multi-service container setup
├── 📄 Dockerfile                                      # Container deployment
├── 📄 docker-entrypoint.sh                           # Container initialization script
├── 📄 test_rag_system.py                               # RAG system testing script
├── 📄 run_evaluation.py                               # RAG evaluation framework runner
├── 📄 run_synthetic_evaluation.py                     # Synthetic data evaluation runner
├── 📄 Makefile                                        # Build automation
├── 📄 PROJECT_CANVAS.md                               # Project roadmap & tasks
├── 📄 CLAUDE.md                                       # AI assistant development log
└── 📄 README.md                                       # Project documentation
```

## Data Processing Pipeline

The project includes a comprehensive data processing pipeline:

1. **Raw Data Ingestion**: Processes large JSONL files from Amazon Reviews 2023
2. **Product Selection**: Intelligently selects top 1000 products based on review volume and quality
3. **Review Sampling**: Extracts representative reviews for each product
4. **Data Cleaning**: Handles missing values, validates data integrity
5. **RAG Optimization**: Formats data for retrieval-augmented generation systems
6. **Vector Database Creation**: Automatic ingestion into ChromaDB with embeddings and metadata
7. **Query Processing**: Intelligent context retrieval based on query type and intent

## Visualization Capabilities

The visualization notebook provides comprehensive insights:

- **Review Distribution Analysis**: Product popularity and rating patterns
- **Price Analysis**: Price ranges and correlation with ratings
- **Category Analysis**: Hierarchical category exploration
- **Store & Brand Analysis**: Top performers and market distribution
- **Temporal Analysis**: Review trends over time (2003-2023)
- **Text Analysis**: Review length and content characteristics

## Technical Stack

- **Data Processing**: pandas, numpy, json
- **Visualization**: matplotlib, seaborn, plotly
- **Vector Database**: Dual-architecture ChromaDB system (local: GTE-large, Docker: optimized)
- **Embedding Models**: GTE-large (development) and ChromaDB default (production) with automatic selection
- **RAG Implementation**: Custom query processing with intelligent context retrieval and environment detection
- **Notebook Environment**: Jupyter, IPython
- **Package Management**: uv (modern Python package manager)
- **Web Interface**: Professional Streamlit UI with tab-based architecture, smart query suggestions, and real-time monitoring
- **LLM Providers**: OpenAI GPT-4o, Groq Llama, Google Gemini 2.0
- **Monitoring**: Optimized Weave tracing via Weights & Biases with session state management
- **Configuration**: Pydantic settings with environment variables
- **Containerization**: Docker with non-root security

## Usage Examples

### Data Processing
```python
# Load processed data
import pandas as pd
import json

# Load products
products = []
with open('data/processed/electronics_top1000_products.jsonl', 'r') as f:
    for line in f:
        products.append(json.loads(line.strip()))

df_products = pd.DataFrame(products)
print(f"Loaded {len(df_products)} products")
```

### RAG System
```python
# Test RAG system
from src.rag.query_processor import create_rag_processor

# Initialize processor
processor = create_rag_processor()

# Process a query
result = processor.process_query("What do people say about iPhone charger cables?")
print(f"Found {result['metadata']['num_products']} products and {result['metadata']['num_reviews']} reviews")
```

### RAG Evaluation Framework
```python
# Run comprehensive evaluation
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.dataset import create_evaluation_dataset

# Create evaluator with your RAG processor and LLM client
evaluator = RAGEvaluator(rag_processor, llm_client)

# Run single query evaluation
metrics = evaluator.run_single_evaluation(
    query="What are iPhone charger features?",
    expected_answer="Expected response...",
    expected_products=["lightning_cable"],
    expected_topics=["features", "compatibility"],
    query_type="product_info"
)

print(f"Overall Score: {metrics.overall:.3f}")
```

### Synthetic Test Data Generation
```python
# Generate synthetic evaluation data
from src.evaluation.synthetic_data_generator import create_synthetic_dataset, SyntheticDataConfig

# Custom configuration
config = SyntheticDataConfig(
    num_examples_per_category=5,
    difficulty_distribution={"easy": 0.3, "medium": 0.5, "hard": 0.2},
    variation_techniques=["rephrase", "specificity", "context"]
)

# Generate synthetic examples
synthetic_examples = create_synthetic_dataset(config, num_examples=30)
print(f"Generated {len(synthetic_examples)} synthetic test cases")

# Create mixed dataset (original + synthetic)
from src.evaluation.synthetic_data_generator import create_mixed_dataset
original_examples = create_evaluation_dataset()
mixed_dataset = create_mixed_dataset(original_examples, synthetic_ratio=0.5)
```

### Visualization
```python
# Generate temporal analysis
from notebooks.data_visualization import temporal_analysis
temporal_analysis(df_reviews)
```

## Troubleshooting

### Common Issues

1. **Timestamp Conversion Error**: The project handles timestamps in milliseconds format. If you encounter `OutOfBoundsDatetime` errors, ensure you're using `unit='ms'` in pandas datetime conversions.

2. **Missing Dependencies**: Run `uv sync` to ensure all dependencies are installed.

3. **Kernel Issues**: Install the project kernel with:
   ```bash
   uv run python -m ipykernel install --user --name ai-product-assistant
   ```

4. **Large File Handling**: The raw data files are large (25GB+). Ensure sufficient disk space and memory.

5. **Multiple Weave Traces (Fixed)**: Previous versions created redundant trace calls due to improper interaction between `@st.cache_resource` and `@weave.op()` decorators. This has been resolved with session state management and optimized tracing architecture.

## Recent Improvements

### Enhanced Streamlit UI (v0.6.0)
- **New Feature**: Professional tab-based interface architecture
- **Smart Query Features**: Auto-suggestions, query history, and intelligent filters
- **Real-Time Monitoring**: Performance metrics, RAG analytics, and system health dashboard
- **Enhanced Response Display**: Context cards, structured information, and query analysis
- **Improved UX**: Organized configuration, categorized examples, and responsive design

### Weave Tracing Optimization (v0.5.0)
- **Issue Resolved**: Eliminated multiple/redundant Weave trace calls
- **Root Cause**: Improper interaction between Streamlit caching and Weave decorators
- **Solution**: Session state initialization + consolidated trace entry points
- **Result**: Clean, meaningful traces with zero redundancy

## Documentation

This project includes comprehensive documentation to help you understand and work with the system:

### [PROJECT_CANVAS.md](PROJECT_CANVAS.md)
**Project roadmap and task tracking**
- Complete project overview and goals
- Sprint 0 and Sprint 1 deliverables with detailed task breakdowns
- EDA findings and dataset analysis summary
- Configuration features and tracing implementation status
- Success criteria and architecture decisions

### [docs/SPRINT_0.md](docs/SPRINT_0.md)
**Sprint 0 foundation summary**
- Foundational components completed (June 28, 2025)
- Data processing pipeline, LLM configuration, monitoring setup
- Project setup, environment configuration, and architecture planning
- Technical achievements and development infrastructure
- Foundation established for RAG implementation

### [docs/SPRINT_1.md](docs/SPRINT_1.md)
**Sprint 1 RAG prototype implementation**
- Complete RAG system implementation following course requirements
- Vector database setup, basic RAG pipeline, instrumentation, and evaluation
- All 4 instructor-specified tasks completed (Lessons 3-6)
- Advanced features beyond scope: query intelligence, dual-architecture, synthetic data
- W&B integration with comprehensive evaluation framework

### [docs/CHROMA.md](docs/CHROMA.md)
**Complete ChromaDB integration guide**
- GTE-large embedding model implementation details
- Data loading process and timeline details
- Search capabilities and metadata schema
- Performance monitoring and logging
- Troubleshooting guide and best practices
- API reference and usage examples

### [docs/LOCAL_VS_DOCKER.md](docs/LOCAL_VS_DOCKER.md)
**Local development vs Docker production comparison**
- Dual-architecture approach explanation (vector_db.py vs vector_db_docker.py)
- Embedding strategy differences (GTE-large vs ChromaDB default)
- Connection architecture and storage configuration details
- Performance comparison and resource usage analysis
- Use case guidelines and migration considerations
- Troubleshooting and best practices for both environments

### [docs/WEAVE_TRACING_GUIDE.md](docs/WEAVE_TRACING_GUIDE.md)
**Comprehensive LLM tracing and monitoring guide**
- Complete Weave integration implementation details
- Configuration parameter tracking (temperature, max_tokens, top_p, top_k)
- W&B dashboard setup and trace analysis
- Provider-specific handling and error resilience
- Performance monitoring and debugging techniques
- Troubleshooting guide for common tracing issues

### [docs/EVALUATIONS.md](docs/EVALUATIONS.md)
**RAG evaluation framework documentation**
- Comprehensive evaluation system using Weave for experiment tracking
- 5 core metrics: Relevance, Accuracy, Completeness, Factuality, Quality
- 14 evaluation examples across 6 query types and 3 difficulty levels
- Command-line interface and integration examples
- Performance benchmarks and customization guide
- Continuous integration setup and troubleshooting

### [docs/SYNTHETIC_DATA.md](docs/SYNTHETIC_DATA.md)
**Synthetic test data generation guide**
- Advanced synthetic data generation with template-based queries and variation techniques
- Configurable generation parameters: difficulty distribution, query types, and variation methods
- Quality validation tools: uniqueness analysis, length distribution, and topic coverage
- Weave integration for full traceability and performance monitoring
- Mixed dataset creation combining original and synthetic data for robust testing
- Best practices implementation and troubleshooting guide

### [docs/DASHBOARD_METRICS.md](docs/DASHBOARD_METRICS.md)
**Dashboard metrics interpretation & implementation guide**
- Comprehensive documentation of all monitoring dashboard metrics
- Session statistics including conversation balance logic and message ratio handling
- Performance monitoring with provider-specific tracking and comparison
- RAG system metrics including vector performance and context quality
- Business intelligence integration with user journey analytics
- Configuration status monitoring and system health indicators
- Implementation details and troubleshooting guidelines

### [docs/GEMINI_MESSAGE_HANDLING.md](docs/GEMINI_MESSAGE_HANDLING.md)
**Google Gemini integration guide**
- Complete Google GenAI client message formatting requirements
- Role conversion and content validation for Gemini compatibility
- Error resolution for INVALID_ARGUMENT and empty message parts
- Performance monitoring and provider-specific baselines
- Integration with Enhanced Tracing v2.0 system
- Troubleshooting guide and best practices

### [docs/DOCKER_TTY_FIXES.md](docs/DOCKER_TTY_FIXES.md)
**Containerized deployment compatibility guide**
- Docker TTY issues and solutions for production deployment
- Non-root user configuration and security best practices
- Streamlit headless configuration for container environments
- Weave tracing compatibility in containerized setups
- Complete verification steps and troubleshooting

### [CLAUDE.md](CLAUDE.md)
**AI assistant development log**
- Detailed record of changes and improvements made by the AI assistant
- Implementation decisions and technical explanations
- Feature development timeline and reasoning
- Code modifications and their rationale

These documents provide in-depth technical guidance beyond the quick start instructions in this README, covering advanced topics like monitoring, containerization, and project management.

## Data Sources & Citations

This project uses data from the Amazon Reviews 2023 dataset:

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```

## Contributing

This is a capstone project for educational purposes. Feel free to explore, learn, and adapt the code for your own projects.

## License

This project is licensed under the terms specified in the LICENSE file.
