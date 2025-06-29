# Project Canvas - "AI-Powered Amazon Product Assistant" (B2C only)

* **Prepared by:** [Hendrik Reh](hendrik.reh@gmail.com)
* **Prepared for:** Aurimas Griciūnas (Instrcutor) | [End-to-End AI Engineering Bootcamp](https://maven.com/swirl-ai/end-to-end-ai-engineering)
* **Date:** 2025/06/29
* **Version:** 0.2.0

## Project Name

AI-Powered Amazon Product Assistant (B2C)

## Project Goal

Develop a production-grade, AI-powered product assistant that helps users interact with Amazon-style product data via natural language. The assistant uses a complete RAG implementation to answer product-related questions, surface relevant information from reviews, and provide intelligent recommendations – demonstrating end-to-end AI engineering with vector databases and context-aware retrieval.

## Problem Statement

Online shoppers face information overload and often struggle to find the right product insights in massive marketplaces. Our AI assistant addresses this by retrieving and summarizing key data (e.g., reviews, ratings, product features) to support better and faster purchase decisions.

## Out-of-Scope (B2B Scope)

- contractual pricing
- account-specific catalogs
- procurement compliance
- multi-user workflows (approvers, requisitioners, etc.)
- Bulk ordering, BOM-style inputs, or quote-based negotiation are not captured
- ERP integration, punchout catalogs (OCI, cXML)
- product taxonomies (e.g. ETIM, UNSPSC)

## Target Users

- End consumers browsing for specific products
- Customer support agents
- E-commerce platform operators aiming to enhance product discoverability

## Data Source

### Public Amazon product datasets

- Metadata (title, brand, category, etc.)
- Reviews (text, rating, verified purchase flag, helpful votes)

### Subset focus

Selected product category (e.g., electronics, kitchenware) with a few thousand entries for initial development and indexing.

## Key Use Cases

- "What do people say about [Product X]?"
- "Is this product good for [specific use case]?"
- "What are the main complaints about [Product Y]?"
- "Compare [Product A] and [Product B]"
- "Recommend a budget-friendly alternative to [Product Z]"

## Core System Architecture (Implemented)

```text
[User Input]
     ↓
[Query Type Detection & Analysis]
     ↓
[ChromaDB Vector Search + Metadata Filtering]
     ↓
[Context-Aware Retrieval (Products + Reviews)]
     ↓
[Enhanced Prompt Generation with RAG Context]
     ↓
[Multi-Provider LLM (OpenAI/Groq/Google)]
     ↓
[Intelligent Response Generation]
```

### Components

- **Vector Database**: ChromaDB with persistent storage and semantic embeddings
- **Query Processor**: Intelligent query type detection and context-aware retrieval
- **Multi-Provider LLMs**: OpenAI GPT-4o, Groq Llama, Google Gemini 2.0
- **Frontend**: Enhanced Streamlit UI with RAG controls and example queries
- **Monitoring**: Weave tracing integration for performance analysis

## Sprint 0 Deliverables

### Task 1 – Project Setup ✅

- Local project environment running from base repo (Lesson 5)
- Dependencies installed and system operational

### Task 2 – Project Configuration ✅

- **Added config sliders in Streamlit sidebar:**
  - **Temperature**: 0.0-2.0 range (default: 0.7) - Controls response creativity/randomness
  - **Max Tokens**: 50-2000 range (default: 500) - Controls response length
  - **Top-p**: 0.0-1.0 range (default: 1.0) - Nucleus sampling for diversity control
  - **Top-k**: 1-100 range (default: 40) - Vocabulary limitation (Google only)
- **Multi-provider support**: Full configuration for Google, partial for OpenAI/Groq
- **Real-time updates**: Settings take effect immediately for new messages
- **Visual feedback**: Current configuration and provider compatibility displayed
- **Weave tracing integration**: Optional LLM call monitoring via W&B (WANDB_API_KEY)
- **Error handling**: Graceful fallback when tracing unavailable

### Task 3 – EDA on Amazon Reviews ✅

**Comprehensive dataset analysis completed with the following key findings:**

*Full analysis available in: [`notebooks/data_visualization_executed.ipynb`](notebooks/data_visualization_executed.ipynb)*

#### Dataset Overview
- **1,000 products** with **20,000 reviews** (20 reviews per product average)
- **20-year span** of review data (2003-2023)
- **High data quality**: 100% completeness for ratings, stores, categories; 77.2% for pricing

#### Review Distribution Analysis
- **Review count range**: 3,453 to 178,239 reviews per product (median: 5,332)
- **Rating distribution**: 4.29/5.0 average with strong positive skew
  - 68.2% five-star reviews, only 8.1% one-star reviews
- **Temporal patterns**: Tuesday most active day, January most active month
- **Recent activity**: 37.8% of reviews from 2020 onwards

#### Price Analysis
- **Price range**: $3.49 - $1,175.35 (median: $26.99)
- **Affordable focus**: 46.6% under $25, only 13.6% over $100
- **Market segments**: Clear budget ($10-25), mid-range ($25-100), premium ($100+) tiers

#### Category & Market Structure
- **312 unique categories** with 3.8 average hierarchy depth
- **Electronics dominance**: 84.8% pure electronics, heavy computer accessories focus
- **Store diversity**: 416 unique stores, Amazon leads with 107 products
- **Brand variety**: Multiple brands per major store indicating marketplace model

#### Content Characteristics
- **Product titles**: 133 characters average (detailed, descriptive)
- **Review text**: 274 characters average (substantial user feedback)
- **Category depth**: 0-7 levels showing sophisticated product taxonomy

### Task 4 – LLM Tracing & Monitoring ✅

- **Weave integration**: Implemented `@weave.op()` decorator on LLM calls
- **W&B authentication**: Automatic login with WANDB_API_KEY from environment
- **Project tracking**: All traces logged to "Bootcamp" project in W&B
- **Error resilience**: Graceful degradation when tracing unavailable
- **Docker compatibility**: Fixed TTY issues for containerized deployment
  - *Detailed documentation*: [docs/DOCKER_TTY_FIXES.md](docs/DOCKER_TTY_FIXES.md)
- **Performance monitoring**: Tracks call latency, token usage, and configuration
- **Configuration tracking**: Logs temperature, max_tokens, top_p, top_k values
  - *Complete tracing guide*: [docs/WEAVE_TRACING_GUIDE.md](docs/WEAVE_TRACING_GUIDE.md)

### Task 5 – RAG System Implementation ✅

**Complete RAG system implemented with vector database and intelligent query processing:**

#### Vector Database Setup
- **ChromaDB Integration**: Persistent vector storage with 2,000 documents (1,000 products + 1,000 reviews)
- **Automatic Ingestion**: Electronics RAG documents with structured metadata and embeddings
- **Performance**: Sub-second retrieval with semantic search and metadata filtering

#### Query Processing Intelligence  
- **6 Query Types**: Product info, reviews, complaints, comparisons, recommendations, use cases
- **Context-Aware Retrieval**: Customizable limits for products (1-10) and reviews (1-8)
- **Enhanced Prompts**: Structured context injection with product details and customer feedback

#### Streamlit Integration
- **RAG Controls**: Toggle, retrieval limits, real-time search status
- **Example Queries**: Interactive buttons for common use cases
- **Visual Feedback**: Search results and query type detection displayed

#### System Performance
- **100% Success Rate**: All query types working correctly
- **0.16s Average Response**: Fast semantic search and context building  
- **Comprehensive Testing**: Automated test suite with performance metrics

*Complete implementation files: `src/rag/vector_db.py`, `src/rag/query_processor.py`, `test_rag_system.py`*

## Success Criteria ✅

- **✅ Product Q&A**: System answers product-related questions using actual review data
- **✅ Contextual Responses**: Users receive helpful, contextual, and accurate responses via RAG
- **✅ Modular Architecture**: Clean separation of concerns with extensible RAG components
- **✅ Query Intelligence**: Handles 6 different query types with appropriate context retrieval
- **✅ Performance**: Sub-second response times with 100% success rate across test cases
- **✅ Production Ready**: Complete testing suite, documentation, and deployment configuration

## Repository

[GitHub repo](https://github.com/HendrikReh/AI-Powered-Amazon-Product-Assistant)
