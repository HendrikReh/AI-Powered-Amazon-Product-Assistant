# Weave Tracing & Configuration Logging Guide

This document explains how the AI-Powered Amazon Product Assistant implements comprehensive LLM call tracing and configuration logging using Weave (Weights & Biases).

## Overview

The application uses Weave to automatically track all LLM interactions, including:
- Model configuration parameters (temperature, max_tokens, top_p, top_k)
- Provider and model selection
- Input messages and conversation context
- Generated responses and metadata
- Performance metrics (latency, token usage)

## Implementation Architecture

### 1. Weave Initialization

```python
# Initialize Weave for tracing
if config.WANDB_API_KEY:
    try:
        import wandb
        # Login to W&B with API key before initializing Weave
        wandb.login(key=config.WANDB_API_KEY, anonymous="never", force=True)
        weave.init(project_name="Bootcamp")
        st.sidebar.success("🔍 Weave tracing enabled")
    except Exception as e:
        st.sidebar.error(f"❌ Weave initialization failed: {str(e)}")
        st.sidebar.info("Continuing without tracing...")
else:
    st.sidebar.info("ℹ️ Weave tracing disabled (no WANDB_API_KEY)")
```

**Key Features**:
- **Conditional initialization**: Only activates when `WANDB_API_KEY` is provided
- **Non-interactive login**: Uses `force=True` to prevent TTY issues in containers
- **Error resilience**: Application continues functioning even if tracing fails
- **Visual feedback**: Clear status indicators in the Streamlit sidebar

### 2. Function Decoration with @weave.op()

```python
@weave.op()
def run_llm(client, messages):
    # Get configuration from session state
    temperature = st.session_state.get('temperature', 0.7)
    max_tokens = st.session_state.get('max_tokens', 500)
    top_p = st.session_state.get('top_p', 1.0)
    top_k = st.session_state.get('top_k', 40)
    
    if st.session_state.provider == "Google":
        return client.models.generate_content(
            model=st.session_state.model_name,
            contents=[message["content"] for message in messages],
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k
            }
        ).text
    else:
        # OpenAI and Groq support top_p but not top_k
        return client.chat.completions.create(
            model=st.session_state.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        ).choices[0].message.content
```

**Automatic Capture**:
- **Function inputs**: All parameters passed to the function
- **Configuration values**: Retrieved from Streamlit session state
- **Function outputs**: Generated text responses
- **Execution metadata**: Timestamp, duration, success/failure status

## Configuration Parameter Tracking

### 3. Parameter Sources and Flow

| Parameter | UI Control | Range | Default | Provider Support |
|-----------|------------|-------|---------|------------------|
| `temperature` | Slider | 0.0-2.0 | 0.7 | All providers |
| `max_tokens` | Slider | 50-2000 | 500 | All providers |
| `top_p` | Slider | 0.0-1.0 | 1.0 | All providers |
| `top_k` | Slider | 1-100 | 40 | Google only |

### 4. Real-time Configuration Display

```python
# Display current configuration
st.divider()
st.caption(f"🎛️ Config: Temp={temperature} | Tokens={max_tokens} | Top-p={top_p} | Top-k={top_k}")

# Parameter support info
if provider == "Google":
    st.caption("✅ All parameters supported")
else:
    st.caption("⚠️ Top-k not supported by OpenAI/Groq")
```

**Visual Features**:
- **Live updates**: Configuration display updates as sliders change
- **Provider compatibility**: Clear indicators of parameter support
- **Parameter values**: Current settings displayed in compact format

## Setup & Configuration

### 5. Environment Setup

1. **Get W&B API Key**:
   ```bash
   # Sign up at https://wandb.ai
   # Get API key from https://wandb.ai/authorize
   ```

2. **Configure Environment**:
   ```bash
   # Add to your .env file
   echo "WANDB_API_KEY=your_wandb_api_key" >> .env
   ```

3. **Verify Configuration**:
   ```bash
   # Run the application
   streamlit run src/chatbot-ui/streamlit_app.py
   # Check sidebar for "🔍 Weave tracing enabled" message
   ```

### 6. Docker Compatibility

The implementation includes specific fixes for containerized environments:

```python
# Non-interactive authentication prevents TTY issues
wandb.login(key=config.WANDB_API_KEY, anonymous="never", force=True)
```

**Benefits**:
- **Container-friendly**: No interactive prompts that hang containers
- **CI/CD compatible**: Works in automated deployment pipelines
- **Error handling**: Graceful degradation when tracing unavailable

## Data Tracked

### 7. Comprehensive Logging

Each LLM call captures:

```json
{
  "inputs": {
    "client": "OpenAI/Groq/Google client object",
    "messages": [
      {"role": "user", "content": "User message"},
      {"role": "assistant", "content": "Previous response"}
    ],
    "temperature": 0.7,
    "max_tokens": 500,
    "top_p": 1.0,
    "top_k": 40,
    "provider": "OpenAI",
    "model_name": "gpt-4o-mini"
  },
  "output": "Generated response text",
  "metadata": {
    "start_time": "2024-01-01T12:00:00Z",
    "end_time": "2024-01-01T12:00:02Z",
    "duration_ms": 2000,
    "status": "success"
  }
}
```

### 8. Performance Metrics

Automatically tracked metrics include:
- **Latency**: Response time for each LLM call
- **Token usage**: Input and output token counts (where available)
- **Success rate**: Percentage of successful vs failed calls
- **Configuration impact**: Performance correlation with parameter settings

## Viewing and Analyzing Traces

### 9. W&B Dashboard Access

1. **Navigate to W&B**:
   - Visit [wandb.ai](https://wandb.ai)
   - Log in with your account

2. **Find Your Project**:
   - Look for the "Bootcamp" project
   - All traces are organized by timestamp

3. **Explore Traces**:
   - Click individual traces to see detailed logs
   - Compare different configuration settings
   - Analyze performance patterns

### 10. Trace Analysis Features

**Configuration Comparison**:
- Compare response quality across temperature settings
- Analyze token usage patterns with different max_tokens values
- A/B test provider performance with identical configurations

**Performance Analysis**:
- Identify optimal configuration combinations
- Track response times across providers
- Monitor error rates and failure patterns

**Conversation Flow**:
- View complete conversation context
- Understand how configuration changes affect responses
- Debug specific interaction issues

## Benefits & Use Cases

### 11. Development Benefits

- **Debugging**: Identify configuration issues and optimize settings
- **Performance Optimization**: Find the best parameter combinations
- **Cost Management**: Track token usage and optimize for efficiency
- **Quality Assurance**: Monitor response quality across different settings

### 12. Production Benefits

- **Monitoring**: Real-time visibility into LLM performance
- **Analytics**: Data-driven insights for system optimization
- **Troubleshooting**: Detailed logs for issue investigation
- **Compliance**: Audit trail for LLM interactions

## Advanced Features

### 13. Provider-Specific Handling

```python
if st.session_state.provider == "Google":
    # Google supports all parameters including top_k
    config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "top_p": top_p,
        "top_k": top_k
    }
else:
    # OpenAI and Groq don't support top_k
    # Parameter is tracked but not passed to API
    pass
```

**Smart Parameter Handling**:
- Tracks all parameters regardless of provider support
- Only passes supported parameters to each API
- Maintains consistent logging across all providers

### 14. Error Handling & Resilience

```python
try:
    weave.init(project_name="Bootcamp")
    st.sidebar.success("🔍 Weave tracing enabled")
except Exception as e:
    st.sidebar.error(f"❌ Weave initialization failed: {str(e)}")
    st.sidebar.info("Continuing without tracing...")
```

**Graceful Degradation**:
- Application continues working even if tracing fails
- Clear error messages for troubleshooting
- No impact on core functionality

## Troubleshooting

### 15. Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Missing API key | "Weave tracing disabled" message | Add `WANDB_API_KEY` to `.env` file |
| Authentication failure | "Weave initialization failed" error | Verify API key validity at wandb.ai |
| Container hanging | Application doesn't start | Ensure `force=True` in wandb.login() |
| No traces visible | Empty dashboard | Check project name and API key permissions |

### 16. Verification Steps

1. **Check Environment**:
   ```bash
   # Verify API key is set
   echo $WANDB_API_KEY
   ```

2. **Test Authentication**:
   ```python
   import wandb
   wandb.login(key="your_api_key", force=True)
   ```

3. **Verify Traces**:
   - Run the Streamlit app
   - Send a test message
   - Check W&B dashboard for new traces

This comprehensive tracing system provides full visibility into LLM interactions while maintaining robust error handling and container compatibility. 