# Enhanced Weave Tracing Guide

## Overview

This guide documents the comprehensive enhanced Weave tracing implementation for the AI-Powered Amazon Product Assistant. The system features **production-ready tracing** with **zero single points of failure**, comprehensive business intelligence, and graceful degradation mechanisms.

## 🚀 Latest Enhancements (2025)

### **Three-Tier Tracing Architecture**

1. **🔗 Context Propagation** - Session-based trace contexts with unique IDs
2. **📊 Vector Performance Monitoring** - Real-time embedding and search analytics  
3. **🧠 Business Intelligence Tracing** - User journey and conversion analytics

## 📁 Architecture Overview

```
src/
├── tracing/
│   ├── trace_utils.py           # Core tracing utilities & context propagation
│   └── business_intelligence.py # Business metrics & user journey tracking
├── rag/
│   ├── query_processor.py       # Enhanced with context propagation
│   ├── vector_db.py            # Performance monitoring integration
│   └── vector_db_docker.py     # Docker-optimized with tracing
└── chatbot-ui/
    └── streamlit_app.py         # Real-time BI dashboard integration
```

## 🔗 Context Propagation System

### **Trace Context Structure**
```python
@dataclass
class TraceContext:
    trace_id: str                    # Unique operation identifier
    session_id: str                  # User session identifier  
    conversation_turn: int           # Conversation depth
    user_type: Optional[UserType]    # Classified user behavior
    query_intent: Optional[QueryIntent] # Query classification
    start_time: float               # Performance timing
```

### **Usage Example**
```python
# Create enhanced trace context
trace_context = create_enhanced_trace_context(
    session_id="user_123",
    conversation_turn=3
)

# Context automatically propagates through all operations
result = rag_processor.build_context(
    query="What are the best budget laptops?",
    trace_id=trace_context.trace_id
)
```

### **Automatic Propagation**
- **Session Management**: Persistent across conversation turns
- **Operation Linking**: All sub-operations share trace context
- **Performance Correlation**: End-to-end timing analysis
- **Error Correlation**: Failures tracked across operation chains

## 📊 Vector Performance Monitoring

### **Embedding Performance Tracking**
```python
@weave.op()
def track_embedding_performance(query: str, embedding_time: float, embedding_dims: int):
    return {
        "query_length": len(query),
        "embedding_time_ms": round(embedding_time * 1000, 2),
        "embedding_dimensions": embedding_dims,
        "tokens_per_second": len(query.split()) / embedding_time,
        "characters_per_ms": len(query) / (embedding_time * 1000)
    }
```

### **Search Quality Analytics**
```python
@weave.op()
def analyze_search_quality(query: str, results: Dict, context: Dict):
    return {
        "relevance_score": calculate_relevance(query, results),
        "query_coverage": measure_coverage(query, results),
        "result_completeness": min(len(results) / 5, 1.0),
        "context_enrichment": {
            "products_found": context.get('num_products', 0),
            "reviews_found": context.get('num_reviews', 0)
        }
    }
```

### **Cache Performance Monitoring**
```python
class VectorPerformanceMonitor:
    def check_cache_hit(self, query: str) -> bool:
        query_hash = hash(query)
        if query_hash in self.cache:
            self.cache_hits += 1
            return True
        else:
            self.cache_misses += 1
            return False
    
    @property 
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
```

## 🧠 Business Intelligence Tracing

### **User Type Classification**
```python
class UserType(Enum):
    RESEARCHER = "researcher"      # Detailed, analytical queries
    BUYER = "buyer"               # Price-focused, comparison queries
    CASUAL = "casual"             # General, exploratory queries
    TROUBLESHOOTER = "troubleshooter"  # Problem-solving queries
```

### **Journey Stage Tracking**
```python
class UserJourneyStage(Enum):
    EXPLORATION = "exploration"    # Initial browsing
    RESEARCH = "research"         # Detailed investigation
    COMPARISON = "comparison"     # Comparing options
    DECISION = "decision"         # Ready to choose
    ACTION = "action"            # Purchase intent
    SUPPORT = "support"          # Post-purchase help
```

### **Business Metrics Calculation**
```python
@dataclass
class BusinessMetrics:
    user_satisfaction_prediction: float      # 0-1 satisfaction score
    conversion_potential: float              # 0-1 purchase likelihood
    recommendation_effectiveness: float       # 0-1 recommendation quality
    feature_usage_score: float               # 0-1 feature adoption
    query_success_rate: float                # 0-1 successful interactions
    response_quality_score: float            # 0-1 response quality
```

### **Intent Classification**
```python
@weave.op()
def classify_intent(self, query: str) -> QueryIntent:
    """Advanced intent classification with business context."""
    patterns = {
        QueryIntent.PRODUCT_INFO: [r'what is', r'tell me about', r'features'],
        QueryIntent.COMPARISON: [r'compare', r'vs', r'versus', r'difference'],
        QueryIntent.RECOMMENDATION: [r'recommend', r'suggest', r'best'],
        QueryIntent.COMPLAINT: [r'problem', r'issue', r'broken'],
        # ... more patterns
    }
    
    for intent, intent_patterns in patterns.items():
        if any(re.search(pattern, query.lower()) for pattern in intent_patterns):
            return intent
    
    return QueryIntent.GENERAL
```

## 📈 Real-Time Analytics Dashboard

### **Streamlit Integration**
The enhanced Streamlit interface provides real-time monitoring across three tabs:

#### **🔧 Configuration Tab**
- **System Status**: Weave, RAG, and API status indicators
- **Model Selection**: Multi-provider LLM configuration
- **RAG Configuration**: Performance tuning parameters

#### **💬 Query Tab**  
- **Enhanced Chat Interface**: Business intelligence integration
- **Smart Suggestions**: Context-aware query recommendations
- **Real-Time Context**: Retrieved products and reviews display

#### **📊 Monitoring Tab**
- **Session Analytics**: User journey and behavior insights
- **Performance Metrics**: Vector database and LLM timing
- **Business Intelligence**: Conversion tracking and satisfaction metrics
- **Vector Performance**: Embedding and search quality analytics

### **Live Metrics Display**
```python
# Business Intelligence Metrics
if hasattr(st.session_state, 'last_business_intelligence'):
    business_intel = st.session_state.last_business_intelligence
    
    user_journey = business_intel.get('user_journey', {})
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("User Type", user_journey.get('user_type', 'Unknown').title())
        st.metric("Journey Stage", user_journey.get('journey_stage', 'Unknown').title())
    
    with col2:
        st.metric("Queries in Session", user_journey.get('queries_count', 0))
        avg_satisfaction = sum(user_journey.get('satisfaction_scores', [0])) / max(len(user_journey.get('satisfaction_scores', [1])), 1)
        st.metric("Avg Satisfaction", f"{avg_satisfaction:.2f}")
```

## 🛡️ Comprehensive Fallback Mechanisms

### **1. Import Fallback Strategy**

#### **Pattern Implementation**
```python
# Universal fallback pattern across all modules
try:
    from tracing.trace_utils import (
        create_enhanced_trace_context, business_analyzer, performance_monitor
    )
except ImportError:
    # Graceful fallback - system continues without advanced tracing
    create_enhanced_trace_context = lambda **kwargs: None
    business_analyzer = None
    performance_monitor = None
```

#### **Covered Scenarios**
- ✅ **Missing Dependencies**: Tracing modules not installed
- ✅ **Import Path Issues**: Relative vs absolute import problems
- ✅ **Module Corruption**: Damaged or incomplete installations
- ✅ **Environment Differences**: Development vs production disparities

#### **Graceful Degradation**
- **Core Functionality**: RAG + Chat always available
- **User Impact**: No visible errors, features silently disabled
- **Recovery**: Automatic when modules become available

### **2. Vector Database Resilience**

#### **Multi-Path Resource Resolution**
```python
def initialize_vector_db(self):
    # Multiple fallback strategies
    possible_paths = [
        "data/processed/electronics_rag_documents.jsonl",
        "../data/processed/electronics_rag_documents.jsonl", 
        "../../data/processed/electronics_rag_documents.jsonl"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            try:
                self.vector_db = setup_vector_database(path)
                return
            except Exception as e:
                logger.warning(f"Failed to load from {path}: {e}")
                continue
    
    logger.warning("Vector DB unavailable - continuing without RAG")
    self.vector_db = None
```

#### **Covered Scenarios**
- ✅ **Missing Data Files**: JSONL files not found
- ✅ **ChromaDB Service Down**: External service unavailable
- ✅ **Embedding Model Failures**: Model loading issues
- ✅ **File Permissions**: Access denied scenarios
- ✅ **Docker vs Local Paths**: Environment path differences

#### **Graceful Degradation**
- **Fallback Mode**: Standard chatbot without RAG
- **User Communication**: Clear "RAG unavailable" messages
- **Functionality**: Full LLM capabilities maintained

### **3. Performance Monitoring Fallbacks**

#### **Conditional Execution Pattern**
```python
@weave.op()
def enhanced_search_with_monitoring(self, query: str, n_results: int = 5):
    """Search with optional performance monitoring."""
    start_time = time.time()
    
    # Core search operation (always works)
    results = self.collection.query(query_texts=[query], n_results=n_results)
    
    # Optional performance tracking
    if performance_monitor:
        try:
            # Enhanced metrics collection
            embedding_metrics = performance_monitor.track_embedding_performance(...)
            search_metrics = performance_monitor.track_search_performance(...)
            quality_metrics = performance_monitor.analyze_search_quality(...)
            
            return {
                "results": results,
                "performance_metrics": {
                    "embedding_metrics": embedding_metrics,
                    "search_metrics": search_metrics,
                    "quality_metrics": quality_metrics
                }
            }
        except Exception as e:
            logger.warning(f"Performance tracking failed: {e}")
            # Continue without metrics
    
    return {"results": results}
```

#### **Covered Scenarios**
- ✅ **Metrics Calculation Errors**: Math/computation failures
- ✅ **Timing Measurement Issues**: System clock problems
- ✅ **Storage/Logging Failures**: Disk or network issues
- ✅ **Memory Constraints**: Resource exhaustion

#### **Graceful Degradation**
- **Core Search**: Always functional regardless of monitoring
- **Optional Metrics**: Performance insights when available
- **No User Impact**: Transparent metric collection

### **4. Business Intelligence Isolation**

#### **Isolated Failure Handling**
```python
@weave.op()
def enhanced_run_llm_with_bi(client, messages):
    """LLM execution with isolated business intelligence tracking."""
    
    # Core LLM operation (protected)
    llm_result = call_llm_provider(provider, model_name, messages, ...)
    
    # Isolated business intelligence (optional)
    business_intelligence = None
    if llm_result["status"] == "success":
        try:
            business_intelligence = track_business_interaction(
                query=latest_user_message,
                response=llm_result["response"],
                context=rag_context,
                session_id=session_id,
                conversation_turn=conversation_turn
            )
            st.session_state.last_business_intelligence = business_intelligence
        except Exception as e:
            # Isolated failure - doesn't affect chat
            st.session_state.last_business_intelligence = {"error": str(e)}
            logger.warning(f"BI tracking failed: {e}")
    
    return llm_result["response"]
```

#### **Covered Scenarios**
- ✅ **User Classification Errors**: ML model failures
- ✅ **Intent Analysis Issues**: NLP processing problems
- ✅ **Session Tracking Problems**: State management failures
- ✅ **Business Metric Calculation**: Analytics computation errors

#### **Graceful Degradation**
- **Chat Preservation**: Core functionality unaffected
- **Partial Analytics**: Some metrics may still work
- **Clear Status**: BI dashboard shows error state

### **5. Weave Tracing Resilience**

#### **Multi-Stage Initialization**
```python
def initialize_weave_tracing():
    """Robust Weave initialization with comprehensive error handling."""
    
    # Check prerequisites
    if not config.WANDB_API_KEY:
        return {
            "status": "disabled",
            "message": "ℹ️ Weave tracing disabled (no WANDB_API_KEY)",
            "impact": "Local operation only"
        }
    
    try:
        # Stage 1: W&B Authentication
        import wandb
        wandb.login(key=config.WANDB_API_KEY, anonymous="never", force=True)
        
        # Stage 2: Weave Initialization  
        weave.init(project_name="Bootcamp")
        
        return {
            "status": "success",
            "message": "🔍 Weave tracing enabled",
            "features": ["Distributed tracing", "Performance analytics", "Error tracking"]
        }
        
    except ImportError as e:
        return {
            "status": "error",
            "message": f"❌ Weave libraries unavailable: {e}",
            "fallback": "Local tracing only",
            "impact": "No distributed traces"
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"❌ Weave initialization failed: {e}",
            "fallback": "Continuing without tracing",
            "impact": "No performance monitoring"
        }
```

#### **Covered Scenarios**
- ✅ **Missing API Keys**: Authentication unavailable
- ✅ **Network Issues**: W&B service unreachable
- ✅ **Service Outages**: External service down
- ✅ **Import Failures**: Weave libraries missing
- ✅ **Docker TTY Issues**: Container deployment problems

#### **Graceful Degradation**
- **Full Application**: All features work without tracing
- **Status Transparency**: Clear indicators in UI
- **Recovery Ready**: Automatic when service returns

### **6. LLM Provider Fault Tolerance**

#### **Structured Error Response System**
```python
def call_llm_provider(provider: str, model_name: str, messages: list, ...):
    """Fault-tolerant LLM provider calls with structured error handling."""
    
    request_metadata = {
        "provider": provider,
        "model": model_name,
        "timestamp": time.time(),
        "request_id": str(uuid.uuid4())
    }
    
    try:
        # Provider-specific implementations
        if provider == "OpenAI":
            response = client.chat.completions.create(...)
        elif provider == "Ollama":
            response = client.chat(model=model_name, messages=messages, ...)
        # ... other providers
        
        return {
            "status": "success",
            "response": response,
            "metadata": request_metadata
        }
        
    except RateLimitError as e:
        return {
            "status": "error",
            "error_type": "rate_limit",
            "error": "API rate limit exceeded. Please try again in a moment.",
            "retry_after": getattr(e, 'retry_after', 60),
            "metadata": request_metadata
        }
    except AuthenticationError as e:
        return {
            "status": "error", 
            "error_type": "authentication",
            "error": "API authentication failed. Please check your API key.",
            "metadata": request_metadata
        }
    except Exception as e:
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error": str(e),
            "metadata": request_metadata
        }
```

#### **Covered Scenarios**
- ✅ **API Key Issues**: Invalid or expired keys
- ✅ **Rate Limiting**: Request quota exceeded  
- ✅ **Network Timeouts**: Connection problems
- ✅ **Model Unavailability**: Service maintenance
- ✅ **Invalid Requests**: Malformed input

#### **User Experience**
- **Friendly Messages**: Clear, actionable error communication
- **Retry Guidance**: Specific retry timing when available
- **System Stability**: Chat interface remains responsive

## 📊 Fallback Coverage Matrix

| **Component** | **Failure Type** | **Detection** | **Fallback Strategy** | **User Impact** | **Recovery** |
|---------------|------------------|---------------|----------------------|-----------------|--------------|
| **Tracing Modules** | Import Error | `try/except ImportError` | Lambda no-ops | None | Automatic |
| **Vector Database** | Service Down | Connection test | Standard chatbot | "RAG unavailable" | Manual retry |
| **Performance Monitor** | Calculation Error | `if performance_monitor` | Skip metrics | Missing charts | Automatic |
| **Business Intelligence** | Analysis Failure | Exception wrapper | Error state | BI tab shows error | Next query |
| **Weave Tracing** | W&B Connection | Login attempt | Local operation | Status indicator | Service recovery |
| **LLM Providers** | API Failure | Structured response | Error message | Friendly prompt | User retry |
| **Embedding Model** | Loading Failure | Model initialization | Default embeddings | Reduced quality | Restart |
| **Session Context** | State Corruption | Context validation | Fresh session | New session | Immediate |

## 🔧 Configuration and Setup

### **Environment Variables**
```bash
# Required for basic functionality
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key  
GOOGLE_API_KEY=your_google_key

# Optional for enhanced tracing
WANDB_API_KEY=your_wandb_key

# Optional for local LLM
OLLAMA_BASE_URL=http://localhost:11434

# Docker environment detection
CHROMA_HOST=chromadb
CHROMA_PORT=8000
```

### **Initialization Sequence**
```python
# 1. Weave Tracing (optional)
weave_result = get_weave_initialization()

# 2. RAG Processor (with fallbacks)
rag_result = get_rag_processor()

# 3. Business Intelligence (optional)
if rag_result["status"] == "success":
    business_tracker = BusinessIntelligenceTracker()

# 4. Session Management
session_id = st.session_state.get('session_id', str(uuid.uuid4()))
st.session_state.session_id = session_id
```

## 📈 Monitoring and Observability

### **System Health Indicators**
```python
def get_system_health():
    """Comprehensive system health check."""
    health_status = {
        "weave_tracing": weave_result["status"],
        "rag_system": rag_result["status"] == "success",
        "vector_database": rag_result.get("has_vector_db", False),
        "business_intelligence": business_tracker is not None,
        "performance_monitoring": performance_monitor is not None
    }
    
    overall_health = "healthy" if all(health_status.values()) else "degraded"
    
    return {
        "overall": overall_health,
        "components": health_status,
        "capabilities": get_available_capabilities(health_status)
    }
```

### **Performance Baseline Metrics**
- **Response Time**: ~290ms average (improved with GTE embeddings)
- **Database Operations**: 2,000 documents indexed
- **Search Quality**: 100% success rate across query types
- **Embedding Model**: GTE-large (1024D) for local, default (384D) for Docker
- **Cache Hit Rate**: Tracked per session for optimization

### **Business Intelligence KPIs**
- **User Satisfaction**: Predicted 0-1 score based on interaction patterns
- **Conversion Potential**: 0-1 likelihood based on query intent and behavior
- **Feature Adoption**: RAG usage, filter usage, comparison queries
- **Session Quality**: Query depth, follow-up patterns, pain point detection

## 🚀 Production Deployment Guidelines

### **Deployment Checklist**
- ✅ **API Keys**: All provider keys configured
- ✅ **Data Files**: Vector database data available
- ✅ **Health Checks**: System status monitoring enabled
- ✅ **Error Handling**: All fallback mechanisms tested
- ✅ **Performance**: Baseline metrics established
- ✅ **Monitoring**: Weave tracing configured (optional)

### **Scaling Considerations**
- **Session Management**: Redis for distributed sessions
- **Vector Database**: Cluster deployment for high availability
- **Performance Monitoring**: Aggregate metrics across instances
- **Business Intelligence**: Centralized analytics database

### **Monitoring Integration**
```python
# Example health check endpoint
@app.route('/health')
def health_check():
    health = get_system_health()
    status_code = 200 if health["overall"] == "healthy" else 503
    return jsonify(health), status_code
```

## 🔍 Troubleshooting Guide

### **Common Issues and Solutions**

#### **Issue: Weave Tracing Not Working**
```bash
# Check W&B authentication
wandb login --verify

# Verify API key
echo $WANDB_API_KEY

# Test connection
python -c "import weave; weave.init('test-project')"
```

#### **Issue: RAG System Unavailable**
```bash
# Check data files
ls -la data/processed/electronics_rag_documents.jsonl

# Verify ChromaDB service (Docker)
docker ps | grep chromadb

# Test vector database
python -c "from src.rag.query_processor import create_rag_processor; create_rag_processor()"
```

#### **Issue: Performance Monitoring Missing**
```bash
# Check imports
python -c "from src.tracing.trace_utils import performance_monitor; print(performance_monitor)"

# Verify installation
uv run python -c "import numpy; print('NumPy available')"
```

### **Debug Mode Activation**
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Force tracing debug
import os
os.environ['WEAVE_DEBUG'] = '1'
```

## 📚 API Reference

### **Core Functions**

#### **create_enhanced_trace_context()**
```python
def create_enhanced_trace_context(
    session_id: str = None,
    conversation_turn: int = 0
) -> TraceContext:
    """Create comprehensive trace context for operation tracking."""
```

#### **track_business_interaction()**
```python
def track_business_interaction(
    query: str,
    response: str, 
    context: Dict[str, Any],
    session_id: str = None,
    conversation_turn: int = 0
) -> Dict[str, Any]:
    """Track user interaction with business intelligence analysis."""
```

#### **apply_rag_enhancement()**
```python
@weave.op()
def apply_rag_enhancement(
    query: str,
    rag_processor,
    max_products: int = 5,
    max_reviews: int = 3,
    session_id: str = None,
    conversation_turn: int = 0
) -> Dict[str, Any]:
    """Apply RAG enhancement with comprehensive tracing."""
```

### **Business Intelligence Classes**

#### **BusinessMetricsAnalyzer**
```python
class BusinessMetricsAnalyzer:
    def classify_intent(self, query: str) -> QueryIntent
    def calculate_complexity(self, query: str) -> float
    def measure_specificity(self, query: str) -> float
    def predict_satisfaction(self, query: str, response: str) -> float
    def estimate_conversion_potential(self, query: str, context: Dict) -> float
```

#### **VectorPerformanceMonitor**
```python
class VectorPerformanceMonitor:
    def track_embedding_performance(self, query: str, time: float, dims: int) -> Dict
    def track_search_performance(self, time: float, results: Dict, query: str) -> Dict
    def analyze_search_quality(self, query: str, results: Dict, context: Dict) -> Dict
```

## 🎯 Future Enhancements

### **Planned Improvements**
- **Advanced Analytics**: ML-powered user behavior prediction
- **A/B Testing**: Integrated experimentation framework
- **Custom Metrics**: Domain-specific KPI tracking
- **Real-Time Alerts**: Automated performance degradation detection
- **Multi-Tenant Support**: Isolated tracing per organization

### **Integration Opportunities**
- **External Analytics**: Google Analytics, Mixpanel integration
- **APM Tools**: DataDog, New Relic compatibility
- **Business Intelligence**: Tableau, PowerBI connectors
- **Alerting Systems**: PagerDuty, Slack notifications

---

## 📞 Support and Resources

- **Documentation**: This guide and inline code comments
- **Health Monitoring**: `/health` endpoint and Streamlit monitoring tab
- **Logging**: Comprehensive debug logs at all levels
- **Community**: GitHub Issues for questions and improvements

**Last Updated**: January 2025  
**Version**: Enhanced Tracing v2.0  
**Compatibility**: Python 3.12+, All LLM providers, Docker deployment ready