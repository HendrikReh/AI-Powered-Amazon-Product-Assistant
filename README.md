# AI-Powered Amazon Product Assistant

An end-to-end AI engineering project that builds an intelligent product recommendation and analysis system using Amazon Electronics dataset. This capstone project demonstrates modern AI engineering practices including data processing, visualization, and retrieval-augmented generation (RAG).

**Course**: End-to-End AI Engineering Bootcamp ([Maven](https://maven.com/swirl-ai/end-to-end-ai-engineering))

## Features

- **Data Processing Pipeline**: Automated processing of large-scale Amazon product and review data
- **Interactive Visualizations**: Comprehensive analysis dashboards with temporal trends, category insights, and rating patterns  
- **RAG-Ready Dataset**: Processed data optimized for retrieval-augmented generation systems
- **Streamlit UI**: User-friendly interface with configurable LLM parameters (temperature, max tokens, top-p, top-k)
- **Multi-Provider Support**: Compatible with OpenAI, Groq, and Google Gemini models
- **Weave Tracing**: Optional LLM call tracking and performance monitoring via Weights & Biases

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
   
   # Streamlit chatbot interface
   uv run streamlit run src/chatbot-ui/streamlit_app.py
   ```

### Docker Deployment

```bash
# Build the container
make build-docker-streamlit

# Run the application
make run-docker-streamlit
```

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
├── data/
│   ├── Electronics.jsonl              # Raw review data (21GB)
│   ├── meta_Electronics.jsonl         # Raw product metadata (4.9GB)
│   └── processed/
│       ├── electronics_top1000_products.jsonl           # Processed products
│       ├── electronics_top1000_products_reviews.jsonl   # Processed reviews
│       ├── electronics_rag_documents.jsonl              # RAG-optimized documents
│       └── dataset_summary.json                         # Dataset statistics
├── notebooks/
│   ├── data_preprocessing.ipynb        # Data processing pipeline
│   ├── data_visualization.ipynb        # Interactive visualizations
│   ├── verify_api_keys.ipynb          # API configuration
│   └── data_visualization_executed.ipynb  # Executed visualization results
├── src/
│   └── chatbot-ui/                     # Streamlit application
│       ├── core/
│       │   └── config.py              # Configuration management
│       └── streamlit_app.py           # Main application
├── pyproject.toml                      # Dependencies and project config
├── Dockerfile                         # Container configuration
└── Makefile                           # Build automation
```

## Data Processing Pipeline

The project includes a comprehensive data processing pipeline:

1. **Raw Data Ingestion**: Processes large JSONL files from Amazon Reviews 2023
2. **Product Selection**: Intelligently selects top 1000 products based on review volume and quality
3. **Review Sampling**: Extracts representative reviews for each product
4. **Data Cleaning**: Handles missing values, validates data integrity
5. **RAG Optimization**: Formats data for retrieval-augmented generation systems

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
- **Notebook Environment**: Jupyter, IPython
- **Package Management**: uv (modern Python package manager)
- **Web Interface**: Streamlit with configurable LLM parameters
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