# Dashboard Metrics Documentation

## Overview

This document provides comprehensive documentation of all metrics displayed in the Streamlit monitoring dashboard, including their implementation details, interpretation guidelines, and business value.

## 📊 Session Statistics

### Message Tracking Metrics

#### User Queries
```python
user_messages = len([m for m in all_messages if m['role'] == 'user'])
```

**Purpose**: Counts the total number of user messages in the current session  
**Implementation**: Filters all messages for role='user' and counts them  
**Interpretation**: 
- Indicates user engagement level
- Higher values suggest active conversation participation
- Used as denominator for conversation balance calculations

#### Assistant Responses
```python
assistant_messages = len([m for m in all_messages if m['role'] == 'assistant'])
```

**Purpose**: Counts the total number of assistant messages in the current session  
**Implementation**: Filters all messages for role='assistant' and counts them  
**Interpretation**:
- Shows system responsiveness and message generation
- Should generally match or slightly exceed user queries
- Includes welcome messages, error responses, and follow-ups

#### Total Conversation
```python
total_conversation_messages = user_messages + assistant_messages
```

**Purpose**: Provides overall conversation volume indicator  
**Implementation**: Simple sum of user and assistant messages  
**Interpretation**:
- Measures total conversation activity
- Useful for session depth analysis
- Excludes system messages and internal operations

### Conversation Balance Analysis

#### Balance Ratio Calculation
```python
balance_ratio = assistant_messages / user_messages
```

**Purpose**: Quantifies the relationship between user inputs and assistant outputs  
**Mathematical Definition**: Ratio of assistant responses to user queries  
**Range**: 0.0 to ∞ (theoretical), typically 0.5 to 3.0 in practice

#### Balance Status Logic
```python
if balance_ratio < 0.8:  # More user messages than assistant (assistant behind)
    balance_status = "🟡 Pending"
elif balance_ratio <= 2.0:  # Normal range (1:1 to 1:2 ratio, accounts for welcome messages)
    balance_status = "🟢 Balanced"
else:  # Significantly more assistant messages (unusual, >2:1 ratio)
    balance_status = "🔴 Unbalanced"
```

**Status Definitions**:

##### 🟢 Balanced (Ratio: 0.8 - 2.0)
- **Ideal State**: Normal conversation flow
- **Scenarios**:
  - 1:1 ratio (perfect responsiveness)
  - 1:2 ratio (includes welcome messages, clarifications)
  - Accounts for system-initiated messages
- **Business Value**: Indicates healthy user experience

##### 🟡 Pending (Ratio: < 0.8)
- **Condition**: More user queries than assistant responses
- **Scenarios**:
  - Processing delays or errors
  - User sent multiple messages quickly
  - System temporarily unavailable
- **Action Required**: Review system performance

##### 🔴 Unbalanced (Ratio: > 2.0)
- **Condition**: Significantly more assistant messages
- **Scenarios**:
  - Multiple error messages
  - System malfunction
  - Repeated automated responses
- **Action Required**: Investigate system behavior

#### Query History Tracking
```python
len(st.session_state.get('query_history', []))
```

**Purpose**: Tracks cumulative query patterns across sessions  
**Implementation**: Persistent storage in session state  
**Interpretation**: 
- Shows user familiarity with system
- Useful for personalization features
- Indicates return user engagement

## ⚙️ Current Configuration

### System Status Indicators

#### Provider Selection
```python
current_provider = st.session_state.get('provider', 'Not Set')
```

**Purpose**: Displays active LLM provider  
**Values**: OpenAI, Groq, Google, Ollama, Not Set  
**Business Impact**: 
- Performance characteristics vary by provider
- Cost implications differ significantly
- Feature availability varies (e.g., top_k parameter)

#### Model Configuration
```python
current_model = st.session_state.get('model_name', 'Not Set')
```

**Purpose**: Shows selected model within provider  
**Examples**:
- OpenAI: gpt-4o, gpt-4o-mini
- Groq: llama-3.3-70b-versatile
- Google: gemini-2.0-flash-exp
- Ollama: gemma3n:e4b, llama3.2

#### RAG Status
```python
rag_enabled = st.session_state.get('use_rag', False)
rag_status = '🟢 Enabled' if rag_enabled else '🔴 Disabled'
```

**Purpose**: Indicates whether RAG enhancement is active  
**Implementation**: Boolean flag with visual indicator  
**Business Impact**:
- Enabled: Enhanced responses with product/review context
- Disabled: Standard LLM responses without retrieval

#### Parameter Monitoring
```python
temperature = st.session_state.get('temperature', 0.7)
max_tokens = st.session_state.get('max_tokens', 500)
```

**Purpose**: Shows current model parameters affecting response generation  
**Interpretation**:
- **Temperature** (0.0-2.0): Controls creativity vs consistency
- **Max Tokens**: Response length limitation
- **Top-p/Top-k**: Advanced sampling parameters (provider-dependent)

## 🔍 Recent Activity

### RAG Performance Metrics

#### Enhanced RAG Performance Display
```python
if hasattr(st.session_state, 'last_rag_result'):
    rag_res = st.session_state.last_rag_result
    context = rag_res["context"]
    st.success(f"Last Query: Found {context['num_products']} products, {context['num_reviews']} reviews")
```

**Purpose**: Shows retrieval effectiveness for last query  
**Metrics**:
- **Products Found**: Number of relevant products retrieved
- **Reviews Found**: Number of relevant review summaries retrieved
- **Query Type**: Classified intent (product_info, comparison, etc.)
- **Processing Time**: RAG pipeline execution duration

#### Vector Database Performance
```python
perf_metrics = context.get('performance_metrics', {})
if perf_metrics:
    for search_type, metrics in perf_metrics.items():
        emb_metrics = metrics['embedding_metrics']
        search_metrics = metrics['search_metrics']
        quality_metrics = metrics['quality_metrics']
```

**Detailed Metrics**:
- **Embedding Time**: Vector generation duration
- **Search Time**: Database query execution time
- **Relevance Score**: Quality of retrieved results (0.0-1.0)

## ⚡ Latest Query Performance

### Performance Tracking System

#### Provider-Specific Display
```python
llm_provider = perf.get('llm_provider', st.session_state.get('provider', 'Unknown'))
llm_model = perf.get('llm_model', st.session_state.get('model_name', 'Unknown'))
provider_emoji = {
    'OpenAI': '🔥',
    'Groq': '⚡',
    'Google': '🧠',
    'Ollama': '🏠'
}
```

**Purpose**: Visual identification of current LLM provider with performance context  
**Implementation**: Emoji mapping for quick recognition

#### Core Performance Metrics
```python
{
    "total_time_ms": final_trace['total_time_ms'],
    "rag_time_ms": rag_result['processing_time_ms'] if rag_result else 0,
    "llm_time_ms": llm_result['response_time_ms'],
    "llm_provider": provider,
    "llm_model": model_name
}
```

**Metrics Breakdown**:
- **Total Time**: End-to-end request processing duration
- **RAG Time**: Vector search and context generation time
- **LLM Time**: Language model processing and response generation time

#### Performance Breakdown Analysis
```python
if perf['rag_time_ms'] > 0:
    rag_percentage = (perf['rag_time_ms'] / perf['total_time_ms']) * 100
    llm_percentage = (perf['llm_time_ms'] / perf['total_time_ms']) * 100
```

**Purpose**: Identifies performance bottlenecks  
**Interpretation**:
- High RAG %: Vector database or embedding performance issues
- High LLM %: Language model latency or complex queries
- Balanced split: Normal operation

#### Provider-Specific Performance Insights
```python
if llm_provider == "Groq":
    if perf['llm_time_ms'] > 1000:
        st.warning("⚡ Groq response slower than expected (usually <500ms)")
elif llm_provider == "OpenAI":
    if perf['llm_time_ms'] > 5000:
        st.warning("🔥 OpenAI response time above average")
```

**Provider Baselines**:
- **Groq**: Expected <500ms (speed-optimized)
- **OpenAI**: Normal <5000ms (quality-focused)
- **Google**: Variable based on model complexity
- **Ollama**: Expected <10000ms (local processing)

## 🏆 Provider & Model Performance Comparison

### Historical Performance Tracking

#### Provider/Model Statistics Structure
```python
provider_model_key = f"{provider}::{model_name}"
stats = {
    "provider": provider,
    "model": model_name,
    "total_queries": 0,
    "total_time_ms": 0,
    "total_rag_time_ms": 0,
    "total_llm_time_ms": 0,
    "rag_queries": 0,
    "non_rag_queries": 0,
    "min_llm_time_ms": float('inf'),
    "max_llm_time_ms": 0,
    "recent_performances": []
}
```

**Purpose**: Comprehensive performance tracking across providers and models  
**Persistence**: Session-based accumulation for comparison analysis

#### Calculated Performance Metrics
```python
avg_total_time = stats["total_time_ms"] / stats["total_queries"]
avg_llm_time = stats["total_llm_time_ms"] / stats["total_queries"]
avg_rag_time = stats["total_rag_time_ms"] / stats["rag_queries"] if stats["rag_queries"] > 0 else 0
```

**Derived Metrics**:
- **Average Total Time**: Mean end-to-end processing time
- **Average LLM Time**: Mean language model processing time
- **Average RAG Time**: Mean retrieval augmentation time
- **Min/Max LLM Time**: Performance range indicators

### Performance Insights Dashboard

#### Speed Comparison
```python
comparison_data.sort(key=lambda x: x["Avg LLM Time (ms)"])
fastest = comparison_data[0]
slowest = comparison_data[-1]
speed_diff = slowest['Avg LLM Time (ms)'] - fastest['Avg LLM Time (ms)']
```

**Analysis Features**:
- **Fastest Provider**: Best performing provider/model combination
- **Slowest Provider**: Least performing combination
- **Speed Difference**: Performance gap quantification

#### Trend Analysis
```python
recent_times = [p["llm_time_ms"] for p in recent_perfs[-10:]]
avg_recent_3 = sum(recent_times[-3:]) / 3
avg_older_3 = sum(recent_times[-6:-3]) / 3
trend_change = avg_recent_3 - avg_older_3
```

**Trend Indicators**:
- **🔺 Slower**: Performance degrading (>50ms increase)
- **🔻 Faster**: Performance improving (>50ms decrease)
- **➡️ Stable**: Consistent performance (±50ms range)

## 📊 Business Performance Metrics

### Business Intelligence Integration
```python
business_metrics = perf.get('business_metrics', {})
{
    "user_satisfaction_prediction": float,      # 0-1 satisfaction score
    "conversion_potential": float,              # 0-1 purchase likelihood
    "recommendation_effectiveness": float,       # 0-1 recommendation quality
    "feature_usage_score": float,               # 0-1 feature adoption
    "query_success_rate": float,                # 0-1 successful interactions
    "response_quality_score": float             # 0-1 response quality
}
```

**Purpose**: Business value quantification beyond technical metrics  
**Implementation**: ML-powered analysis of user interactions  
**Applications**:
- User experience optimization
- Feature effectiveness measurement
- Conversion funnel analysis
- Product recommendation improvement

### User Journey Analytics

#### User Type Classification
```python
user_type = user_journey.get('user_type', 'Unknown')  # UserType enum
journey_stage = user_journey.get('journey_stage', 'Unknown')  # UserJourneyStage enum
```

**User Types**:
- **Researcher**: Detailed, analytical queries
- **Buyer**: Price-focused, comparison queries
- **Casual**: General, exploratory queries
- **Troubleshooter**: Problem-solving queries

**Journey Stages**:
- **Exploration**: Initial browsing
- **Research**: Detailed investigation
- **Comparison**: Comparing options
- **Decision**: Ready to choose
- **Action**: Purchase intent
- **Support**: Post-purchase help

#### Session Analytics
```python
queries_count = user_journey.get('queries_count', 0)
satisfaction_scores = user_journey.get('satisfaction_scores', [0])
avg_satisfaction = sum(satisfaction_scores) / max(len(satisfaction_scores), 1)
```

**Metrics**:
- **Queries in Session**: Conversation depth indicator
- **Average Satisfaction**: User experience quality measure
- **Journey Progression**: Stage advancement tracking

## 📈 Weave Tracing Integration

### Distributed Tracing Status
```python
weave_result = {
    "status": "success" | "error" | "disabled",
    "message": "Status description",
    "features": ["Distributed tracing", "Performance analytics", "Error tracking"]
}
```

**Status Indicators**:
- **✅ Active**: Full tracing operational with W&B integration
- **⚠️ Degraded**: Partial functionality or connection issues
- **❌ Disabled**: No tracing (missing API key or service unavailable)

### Trace Data Collection
- **RAG Query Processing**: Vector search and retrieval metrics
- **LLM Provider Performance**: Response timing and error tracking
- **Response Timing Analysis**: End-to-end request flow
- **Error Tracking**: Comprehensive failure analysis

## 🔧 Implementation Details

### Metric Calculation Frequency
- **Real-Time**: Performance metrics updated per query
- **Session-Based**: Statistics accumulated during session
- **Historical**: Provider comparison data persists across queries

### Data Storage
- **Session State**: Temporary metrics storage
- **Performance History**: Rolling window of recent performances
- **Export Capability**: JSON download for external analysis

### Error Handling
- **Graceful Degradation**: Metrics display even with partial data
- **Fallback Values**: Default displays when data unavailable
- **Error State Indication**: Clear messaging for metric calculation failures

## 📊 Metric Interpretation Guidelines

### Performance Thresholds

#### Response Time Benchmarks
- **Excellent**: <1000ms total time
- **Good**: 1000-3000ms total time
- **Acceptable**: 3000-10000ms total time
- **Poor**: >10000ms total time

#### RAG Efficiency
- **Excellent**: <200ms RAG time, >0.8 relevance
- **Good**: 200-500ms RAG time, >0.6 relevance
- **Acceptable**: 500-1000ms RAG time, >0.4 relevance
- **Poor**: >1000ms RAG time, <0.4 relevance

#### Conversation Health
- **Healthy**: 🟢 Balanced status, increasing user queries
- **Attention Needed**: 🟡 Pending status, declining satisfaction
- **Critical**: 🔴 Unbalanced status, error patterns

### Business Value Indicators

#### User Engagement
- **High**: >5 queries per session, >0.8 satisfaction
- **Medium**: 2-5 queries per session, 0.6-0.8 satisfaction
- **Low**: <2 queries per session, <0.6 satisfaction

#### System Effectiveness
- **High**: >0.8 success rate, balanced provider performance
- **Medium**: 0.6-0.8 success rate, some provider variation
- **Low**: <0.6 success rate, significant performance gaps

---

**Last Updated**: January 2025  
**Version**: Enhanced Tracing v2.0  
**Compatibility**: Streamlit Dashboard v0.9.0