# AI-Powered Amazon Product Assistant (B2C only)

An end-to-end AI engineering project that builds an intelligent product recommendation and analysis system using Amazon Electronics dataset with a complete RAG implementation. This capstone project demonstrates modern AI engineering practices including data processing, visualization, vector databases, and retrieval-augmented generation (RAG).

**Course**: End-to-End AI Engineering Bootcamp ([Maven](https://maven.com/swirl-ai/end-to-end-ai-engineering))

## Features

- **Data Processing Pipeline**: Automated processing of large-scale Amazon product and review data
- **Interactive Visualizations**: Comprehensive analysis dashboards with temporal trends, category insights, and rating patterns  
- **Complete RAG System**: Vector database with ChromaDB, intelligent query processing, and context-aware retrieval
- **Streamlit UI**: Enhanced chat interface with RAG toggle, configurable LLM parameters, and example queries
- **Multi-Provider Support**: Compatible with OpenAI, Groq, and Google Gemini models
- **Vector Database**: ChromaDB-powered semantic search with metadata filtering and hybrid queries
- **Query Intelligence**: Automatic query type detection for product reviews, comparisons, complaints, and recommendations
- **Weave Tracing**: Optional LLM call tracking and performance monitoring via Weights & Biases

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
   
   # Streamlit chatbot interface with RAG
   uv run streamlit run src/chatbot-ui/streamlit_app.py
   
   # Test RAG system functionality
   uv run python test_rag_system.py
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
- **Streamlit App**: http://localhost:8501 
- **ChromaDB Service**: http://localhost:8000
- **Persistent Storage**: Vector data persisted in Docker volume

### Weave Tracing Setup

The application includes optional Weave tracing for LLM call monitoring and performance analysis.

1. **Get W&B API Key**
   - Sign up at [wandb.ai](https://wandb.ai)
   - Get your API key from [User Settings](https://wandb.ai/authorize)

2. **Configure Tracing**
   ```bash
   # Add to your .env file
   echo "WANDB_API_KEY=your_wandb_api_key" >> .env
   ```

3. **Features Tracked**
   - LLM call performance and latency
   - Model configuration (temperature, tokens, top-p, top-k)
   - Conversation flows and context
   - Provider comparison (OpenAI vs Groq vs Google)

4. **View Traces**
   - Visit your [W&B dashboard](https://wandb.ai)
   - Navigate to the "Bootcamp" project
   - Explore detailed traces and performance metrics

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
│   └── 📁 rag/
│       ├── vector_db.py                               # ChromaDB vector database (local)
│       ├── vector_db_docker.py                       # ChromaDB vector database (Docker)
│       └── query_processor.py                        # RAG query processing
├── 📁 docs/                                          # Technical documentation
│   ├── CHROMA.md                                      # ChromaDB integration guide
│   ├── WEAVE_TRACING_GUIDE.md                         # LLM tracing & monitoring guide
│   └── DOCKER_TTY_FIXES.md                           # Container deployment fixes
├── 📄 pyproject.toml                                  # uv dependencies & config
├── 📄 docker-compose.yml                              # Multi-service container setup
├── 📄 Dockerfile                                      # Container deployment
├── 📄 docker-entrypoint.sh                           # Container initialization script
├── 📄 test_rag_system.py                               # RAG system testing script
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
- **Vector Database**: ChromaDB with persistent storage and semantic search
- **RAG Implementation**: Custom query processing with intelligent context retrieval
- **Notebook Environment**: Jupyter, IPython
- **Package Management**: uv (modern Python package manager)
- **Web Interface**: Streamlit with RAG integration and configurable LLM parameters
- **LLM Providers**: OpenAI GPT-4o, Groq Llama, Google Gemini 2.0
- **Monitoring**: Weave tracing via Weights & Biases
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

## Documentation

This project includes comprehensive documentation to help you understand and work with the system:

### [PROJECT_CANVAS.md](PROJECT_CANVAS.md)
**Project roadmap and task tracking**
- Complete project overview and goals
- Sprint 0 deliverables with detailed task breakdown
- EDA findings and dataset analysis summary
- Configuration features and tracing implementation status
- Success criteria and architecture decisions

### [docs/CHROMA.md](docs/CHROMA.md)
**Complete ChromaDB integration guide**
- Local vs Docker environment setup and configuration
- Data loading process and timeline details
- Search capabilities and metadata schema
- Performance monitoring and logging
- Troubleshooting guide and best practices
- API reference and usage examples

### [docs/WEAVE_TRACING_GUIDE.md](docs/WEAVE_TRACING_GUIDE.md)
**Comprehensive LLM tracing and monitoring guide**
- Complete Weave integration implementation details
- Configuration parameter tracking (temperature, max_tokens, top_p, top_k)
- W&B dashboard setup and trace analysis
- Provider-specific handling and error resilience
- Performance monitoring and debugging techniques
- Troubleshooting guide for common tracing issues

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
