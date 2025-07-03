# Project Canvas - "AI-Powered Amazon Product Assistant" (B2C only)

* **Prepared by:** [Hendrik Reh](hendrik.reh@gmail.com)
* **Prepared for:** Aurimas Griciūnas (Instructor) | [End-to-End AI Engineering Bootcamp](https://maven.com/swirl-ai/end-to-end-ai-engineering)
* **Date:** 2025/07/03
* **Version:** 0.9.0 (Enhanced Tracing v2.0)

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

- **Vector Database**: ChromaDB with GTE-large embeddings (1024-dimensional) and persistent storage
- **Embedding Model**: sentence-transformers with thenlper/gte-large for superior e-commerce understanding
- **Query Processor**: Intelligent query type detection and context-aware retrieval
- **Multi-Provider LLMs**: OpenAI GPT-4o, Groq Llama, Google Gemini 2.0, Ollama (local LLMs)
- **Frontend**: Enhanced Streamlit UI with professional UX design and real-time performance monitoring
- **Enhanced Tracing v2.0**: Three-tier observability with context propagation, vector performance monitoring, and business intelligence

## Sprint 0 Deliverables (Foundation Phase - June 28, 2025)

### Task 1 – Project Setup ✅

- Local project environment running from base repo 
- Dependencies installed and system operational with uv package manager
- Modern development infrastructure established

### Task 2 – Project Configuration ✅

- **Basic Streamlit interface**: Simple chat interface with LLM provider selection
- **Multi-provider support**: OpenAI, Groq, and Google LLM integration
- **Environment setup**: API key configuration and basic error handling
- **Initial monitoring**: Basic Weave tracing integration setup

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

### Task 4 – Data Processing Pipeline ✅

- **Data cleaning and preparation**: 25GB+ Amazon dataset processing
- **RAG document creation**: `electronics_rag_documents.jsonl` formatted for vector database
- **Quality validation**: Processed 1,000 products and 20,000 reviews successfully
- **Documentation**: Comprehensive data processing guides and troubleshooting

### Task 5 – Architecture Planning ✅

- **System design**: RAG architecture planning and component specification
- **Technology selection**: ChromaDB, sentence-transformers, Streamlit stack
- **Development roadmap**: Sprint planning and milestone definition
- **Documentation foundation**: Initial project documentation and guides

## Sprint 1 Deliverables (RAG Implementation - June 29 - July 6, 2025)

### Core Course Requirements (Tasks 1-4) ✅

#### Task 1 – Vector Database Setup ✅ (Lesson 3)
- **ChromaDB Integration**: Persistent vector storage with 2,000 documents (1,000 products + 1,000 reviews)
- **Embedding Upgrade**: Migrated from all-MiniLM-L6-v2 (384D) to GTE-large (1024D) for improved quality
- **Collection**: New `electronics_products_reviews_gte` collection with enhanced embeddings
- **Automatic Ingestion**: Electronics RAG documents with structured metadata and GTE embeddings
- **Performance**: Sub-second retrieval with superior semantic search and metadata filtering

#### Task 2 – Basic RAG Pipeline ✅ (Lesson 4)
- **Query Processing Intelligence**: 6 Query Types (Product info, reviews, complaints, comparisons, recommendations, use cases)
- **Context-Aware Retrieval**: Customizable limits for products (1-10) and reviews (1-8)
- **Enhanced Prompts**: Structured context injection with product details and customer feedback
- **Streamlit Integration**: RAG controls, example queries, and visual feedback
- **System Performance**: 100% success rate, 0.29s average response time

#### Task 3 – Enhanced Tracing v2.0 and Observability ✅ (Lesson 5) - SIGNIFICANTLY ENHANCED
- **Three-Tier Architecture**: Context propagation (Tier 1), vector performance monitoring (Tier 2), business intelligence (Tier 3)
- **53+ Weave Operations**: Comprehensive instrumentation across 8 Python files with granular tracking
- **Context Propagation**: Session-based trace contexts with unique IDs propagated across all operations
- **Vector Performance Monitoring**: Embedding generation timing, search performance, cache hit rates, and result quality metrics
- **Business Intelligence Tracing**: User journey tracking, intent classification, satisfaction prediction, and conversion analysis
- **User Type Classification**: ML-powered classification (Researcher, Buyer, Casual, Troubleshooter) and journey stage tracking
- **Real-Time Analytics**: Live performance metrics, business KPIs, and user behavior insights in Streamlit dashboard
- **Production-Ready Fallbacks**: Zero single points of failure with comprehensive graceful degradation mechanisms
- **W&B Integration**: Advanced session-based traces with user journey analytics logged to "Bootcamp" project

#### Task 4 – Evaluations with W&B ✅ (Lesson 6)
- **Evaluation Framework**: Complete system in `src/evaluation/` with 5 core metrics
- **Weave Integration**: Native W&B support for experiment tracking and analytics
- **Comprehensive Dataset**: 14 evaluation examples across 6 query types with ground truth
- **Performance Benchmarks**: Automated scoring with detailed breakdowns
- **Command-Line Tools**: Production-ready evaluation runner with multiple modes

### Enhanced Tracing v2.0 - Observability System ✅

#### Three-Tier Advanced Observability (Far Exceeding Requirements)
- **Tier 1: Context Propagation & Session Management**: Session-based trace contexts with unique IDs, cross-operation linking, user journey persistence
- **Tier 2: Vector Performance Monitoring**: Embedding performance tracking, search quality analytics, cache optimization, real-time vector metrics
- **Tier 3: Business Intelligence & User Analytics**: User type classification, intent analysis, business KPIs, journey stage tracking, feature usage analytics

#### Production-Ready Architecture
- **Zero Single Points of Failure**: Comprehensive fallback mechanisms across all components
- **Import Resilience**: Graceful degradation when tracing modules unavailable
- **Performance Monitoring**: Optional tracking that degrades gracefully without affecting core functionality
- **Business Intelligence Isolation**: Isolated failure handling preserving core chat functionality

#### Enhanced Streamlit Dashboard Integration
- **Real-Time Business Intelligence**: Live user journey insights and conversion tracking
- **Provider & Model Performance Comparison**: Historical performance tracking and comparison across all LLM providers
- **Session Analytics**: User behavior patterns with business recommendations and trend analysis
- **System Health Monitoring**: Component status with fallback indicators and comprehensive metrics

### Advanced Features (Beyond Course Scope) ✅

#### Enhanced UI Development ✅
- **Professional Tab-Based Interface**: Configuration, Query, Monitoring tabs with modern dark theme
- **Professional UX Typography**: Complete visual hierarchy with light text on dark backgrounds for optimal readability
- **Smart Query Features**: Auto-suggestions, query history, intelligent filters
- **Real-Time Monitoring**: Performance metrics, RAG analytics, system health dashboard
- **Provider Performance Comparison**: Comprehensive tracking and comparison of all LLM providers/models throughout session uptime
- **Enhanced Response Display**: Context cards, structured information, query analysis
- **Improved Accessibility**: High contrast ratios, responsive design, mobile-friendly layouts

#### Advanced Configuration System ✅
- **Multi-Provider LLM Support**: OpenAI, Groq, Google Gemini, Ollama (4 providers) with unified interface
- **Google Gemini Integration**: Complete message formatting compatibility with proper role conversion and content validation
- **Dynamic Parameter Controls**: Temperature, max tokens, top-p, top-k with provider-specific support
- **Visual Feedback**: Current configuration and provider compatibility displayed
- **Real-Time Updates**: Settings take effect immediately for new messages
- **Error Handling**: Graceful fallback when services unavailable

#### Production Optimizations ✅
- **Dual-Architecture Implementation**: Local development (GTE-large) vs Docker production (optimized)
- **Automatic Environment Detection**: Via `CHROMA_HOST` variable
- **Container Optimization**: 670MB space savings through optimized Docker implementation
- **Security Best Practices**: Non-root user configuration and secure deployment

#### Weave Tracing Optimization ✅
- **Issue Resolution**: Eliminated multiple/redundant Weave trace calls
- **Session State Management**: Single-session initialization preventing repeated trace creation
- **Performance Improvements**: Zero duplicate traces, optimized startup, maintained functionality
- **Clean Analytics**: Meaningful trace data in W&B without noise or multiplication

#### Advanced Evaluation Systems ✅
- **Synthetic Data Generation**: Template-based queries with variation techniques
- **Quality Validation**: Automated analysis of uniqueness, length distribution, topic coverage
- **Mixed Dataset Testing**: Combine original and synthetic data for robust validation
- **Production Testing**: Automated test case generation for systematic evaluation

#### Comprehensive Documentation ✅
- **Enhanced Tracing Guides**: `docs/WEAVE_TRACING_GUIDE.md`, `docs/GEMINI_MESSAGE_HANDLING.md`
- **Technical Guides**: `docs/EVALUATIONS.md`, `docs/SYNTHETIC_DATA.md`
- **Deployment Documentation**: `docs/LOCAL_VS_DOCKER.md`, `docs/DOCKER_TTY_FIXES.md`
- **Sprint Documentation**: `docs/SPRINT_0.md`, `docs/SPRINT_1.md`
- **Architecture Documentation**: Complete `docs/CHROMA.md` integration guide

## Success Criteria ✅

- **✅ Product Q&A**: System answers product-related questions using actual review data
- **✅ Contextual Responses**: Users receive helpful, contextual, and accurate responses via RAG
- **✅ Modular Architecture**: Clean separation of concerns with extensible RAG components
- **✅ Query Intelligence**: Handles 6 different query types with appropriate context retrieval
- **✅ Performance**: Sub-second response times with 100% success rate and enhanced embedding quality
- **✅ Enhanced Tracing v2.0**: Three-tier observability system with context propagation, vector performance monitoring, and business intelligence
- **✅ Multi-Provider Support**: Complete integration of OpenAI, Groq, Google Gemini, and Ollama with unified monitoring
- **✅ Professional UX**: Modern dark theme with optimized typography and comprehensive performance comparison dashboard
- **✅ Evaluation Framework**: Comprehensive RAG evaluation system with 5 metrics, 14 test examples, and Weave integration
- **✅ Synthetic Test Data**: Advanced synthetic data generation with template-based queries, variation techniques, and quality analysis
- **✅ Enterprise Ready**: Zero single points of failure, comprehensive fallback mechanisms, production-grade monitoring, and complete documentation

## Sprint Documentation

### [docs/SPRINT_0.md](docs/SPRINT_0.md) - Foundation Phase
**Sprint Duration**: Project Start - June 28, 2025  
**Version**: 0.1.0  
**Scope**: Foundational components and infrastructure setup

**Core Deliverables**:
- Project setup and environment configuration
- Data processing pipeline (25GB+ Amazon dataset)
- Multi-provider LLM configuration (OpenAI, Groq, Google)
- Basic Weave tracing integration
- Architecture planning for RAG implementation

**Key Achievements**:
- 1,000 electronics products, 20,000 reviews processed
- Modern development infrastructure with uv package manager
- Comprehensive documentation and troubleshooting guides
- Foundation established for advanced RAG capabilities

### [docs/SPRINT_1.md](docs/SPRINT_1.md) - RAG Implementation & Enhanced Tracing v2.0
**Sprint Duration**: June 29, 2025 - July 3, 2025  
**Version**: 0.9.0 (Enhanced Tracing v2.0)  
**Scope**: Complete RAG system with observability enhancements (three tier)

**Course Requirements Completed** (Tasks 1-4):
- ✅ **Task 1**: Vector database setup on local machine (Lesson 3)
- ✅ **Task 2**: Basic RAG pipeline implementation (Lesson 4)
- ✅ **Task 3**: Enhanced Tracing v2.0 - SIGNIFICANTLY ENHANCED (Lesson 5)
- ✅ **Task 4**: First batch of evaluations with W&B integration (Lesson 6)

**Enhanced Tracing v2.0 - Achievement**:
- **Three-Tier Architecture**: Context propagation, vector performance monitoring, business intelligence
- **53+ Weave Operations**: Comprehensive instrumentation across 8 Python files
- **Zero Single Points of Failure**: Production-ready fallback mechanisms across all components
- **Business Intelligence**: User journey tracking, intent classification, conversion analysis, satisfaction prediction
- **Real-Time Analytics**: Live performance metrics and business KPIs in enhanced Streamlit dashboard
- **Provider Performance Comparison**: Historical tracking and comparison across all LLM providers/models

**Additional Major Achievements**:
- **Multi-Provider LLM Support**: OpenAI, Groq, Google Gemini, Ollama with unified monitoring
- **Professional UX Design**: Modern dark theme with optimized typography and accessibility
- **Google Gemini Integration**: Complete message formatting compatibility and error handling
- **Comprehensive Documentation**: `docs/WEAVE_TRACING_GUIDE.md`, `docs/GEMINI_MESSAGE_HANDLING.md`
- **Enterprise Deployment**: Container optimization, security hardening, comprehensive testing

**W&B Integration**: Advanced session-based traces with user journey analytics and business intelligence logged to wandb.ai dashboard

## Repository

[GitHub repo](https://github.com/HendrikReh/AI-Powered-Amazon-Product-Assistant)
