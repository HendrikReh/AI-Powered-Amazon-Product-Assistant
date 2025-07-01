import streamlit as st
import weave
from openai import OpenAI
from groq import Groq
from google import genai
import sys
import time
from pathlib import Path

# Add the parent directory to sys.path to import from rag module
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from core.config import config
from rag.query_processor import create_rag_processor

def initialize_rag_processor():
    """Initialize RAG processor (cached wrapper handles tracing)."""
    try:
        processor = create_rag_processor()
        return {
            "processor": processor,
            "status": "success",
            "has_vector_db": processor.vector_db is not None if processor else False
        }
    except Exception as e:
        return {
            "processor": None,
            "status": "error", 
            "error": str(e),
            "has_vector_db": False
        }

# Initialize RAG processor with caching and single trace
@st.cache_resource
@weave.op()
def get_rag_processor():
    """Cached RAG processor initialization with tracing."""
    result = initialize_rag_processor()
    return result

def initialize_weave_tracing():
    """Initialize Weave tracing (cached wrapper handles tracing)."""
    if config.WANDB_API_KEY:
        try:
            import wandb
            # Login to W&B with API key before initializing Weave
            wandb.login(key=config.WANDB_API_KEY, anonymous="never", force=True)
            weave.init(project_name="Bootcamp")
            return {
                "status": "success",
                "message": "🔍 Weave tracing enabled"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"❌ Weave initialization failed: {str(e)}",
                "fallback": "Continuing without tracing..."
            }
    else:
        return {
            "status": "disabled",
            "message": "ℹ️ Weave tracing disabled (no WANDB_API_KEY)"
        }

def apply_rag_enhancement(query: str, rag_processor, max_products: int, max_reviews: int):
    """Apply RAG enhancement to user query with detailed tracing."""
    start_time = time.time()
    
    try:
        # Build context with custom limits
        context = rag_processor.build_context(
            query,
            max_products=max_products,
            max_reviews=max_reviews
        )
        
        rag_applied = bool(context.products or context.reviews)
        enhanced_prompt = None
        
        if rag_applied:
            enhanced_prompt = rag_processor.generate_rag_prompt(context)
        
        processing_time = time.time() - start_time
        
        return {
            "status": "success",
            "rag_applied": rag_applied,
            "context": {
                "query_type": context.query_type,
                "num_products": len(context.products),
                "num_reviews": len(context.reviews),
                "extracted_terms": context.metadata.get("extracted_terms", [])
            },
            "enhanced_prompt": enhanced_prompt,
            "processing_time_ms": round(processing_time * 1000, 2),
            "original_query": query
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        return {
            "status": "error",
            "error": str(e),
            "processing_time_ms": round(processing_time * 1000, 2),
            "original_query": query,
            "fallback": "Using original query"
        }

def call_llm_provider(provider: str, model_name: str, messages: list, temperature: float, 
                     max_tokens: int, top_p: float, top_k: int, client):
    """Call specific LLM provider with comprehensive tracing."""
    start_time = time.time()
    
    # Prepare request metadata
    request_metadata = {
        "provider": provider,
        "model": model_name,
        "parameters": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k if provider == "Google" else None
        },
        "message_count": len(messages),
        "total_chars": sum(len(msg.get("content", "")) for msg in messages)
    }
    
    try:
        if provider == "Google":
            response = client.models.generate_content(
                model=model_name,
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
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            response = completion.choices[0].message.content
        
        response_time = time.time() - start_time
        
        return {
            "status": "success",
            "response": response,
            "response_length": len(response),
            "response_time_ms": round(response_time * 1000, 2),
            "request_metadata": request_metadata
        }
        
    except Exception as e:
        response_time = time.time() - start_time
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "response_time_ms": round(response_time * 1000, 2),
            "request_metadata": request_metadata
        }

@weave.op()
def run_llm(client, messages):
    """Enhanced run_llm function with comprehensive tracing and error handling."""
    overall_start_time = time.time()
    
    # Get configuration from session state
    provider = st.session_state.get('provider', 'OpenAI')
    model_name = st.session_state.get('model_name', 'gpt-4o-mini')
    temperature = st.session_state.get('temperature', 0.7)
    max_tokens = st.session_state.get('max_tokens', 500)
    top_p = st.session_state.get('top_p', 1.0)
    top_k = st.session_state.get('top_k', 40)
    use_rag = st.session_state.get('use_rag', False)
    
    # Initialize response structure
    trace_data = {
        "provider": provider,
        "model": model_name,
        "use_rag": use_rag,
        "message_count": len(messages)
    }
    
    # Prepare messages for LLM
    llm_messages = messages.copy()
    rag_result = None
    
    # Apply RAG if enabled and user message exists
    if use_rag and st.session_state.rag_processor and len(messages) > 0:
        latest_message = messages[-1]
        if latest_message["role"] == "user":
            # Get RAG configuration
            max_products = st.session_state.get('max_products', 5)
            max_reviews = st.session_state.get('max_reviews', 3)
            
            # Apply RAG enhancement with tracing
            rag_result = apply_rag_enhancement(
                latest_message["content"],
                st.session_state.rag_processor,
                max_products,
                max_reviews
            )
            
            trace_data["rag_result"] = rag_result
            
            # Update message if RAG enhancement was successful
            if rag_result["status"] == "success" and rag_result["rag_applied"]:
                llm_messages[-1] = {
                    "role": "user",
                    "content": rag_result["enhanced_prompt"]
                }
                
                # Show RAG context info in sidebar
                with st.sidebar:
                    context = rag_result["context"]
                    st.success(f"🔍 RAG: Found {context['num_products']} products, {context['num_reviews']} reviews")
                    st.caption(f"Query type: {context['query_type']}")
                    st.caption(f"Processing: {rag_result['processing_time_ms']}ms")
            
            elif rag_result["status"] == "error":
                # Show RAG error in sidebar
                st.sidebar.error(f"RAG error: {rag_result['error']}")
    
    # Call LLM provider with tracing
    llm_result = call_llm_provider(
        provider, model_name, llm_messages, 
        temperature, max_tokens, top_p, top_k, client
    )
    
    total_time = time.time() - overall_start_time
    
    # Combine all trace data
    final_trace = {
        **trace_data,
        "llm_result": llm_result,
        "total_time_ms": round(total_time * 1000, 2),
        "success": llm_result["status"] == "success"
    }
    
    # Log trace summary to sidebar if successful
    if llm_result["status"] == "success":
        with st.sidebar:
            st.caption(f"⚡ Total time: {final_trace['total_time_ms']}ms")
            if rag_result and rag_result["status"] == "success":
                st.caption(f"📊 RAG: {rag_result['processing_time_ms']}ms | LLM: {llm_result['response_time_ms']}ms")
    
    # Return response or handle error
    if llm_result["status"] == "success":
        return llm_result["response"]
    else:
        st.error(f"LLM Error ({llm_result['error_type']}): {llm_result['error']}")
        return "I apologize, but I encountered an error processing your request. Please try again."

# Initialize Weave tracing with caching and single trace
@st.cache_resource  
@weave.op()
def get_weave_initialization():
    """Cached Weave initialization with tracing."""
    return initialize_weave_tracing()

# Initialize once per session using session state
if "initialization_complete" not in st.session_state:
    # Initialize Weave tracing
    weave_result = get_weave_initialization()
    st.session_state.weave_result = weave_result
    
    # Initialize RAG processor
    rag_init_result = get_rag_processor()
    st.session_state.rag_init_result = rag_init_result
    st.session_state.rag_processor = rag_init_result["processor"]
    
    st.session_state.initialization_complete = True
else:
    # Use cached results
    weave_result = st.session_state.weave_result
    rag_init_result = st.session_state.rag_init_result
    rag_processor = st.session_state.rag_processor

# Display initialization status in sidebar
if weave_result["status"] == "success":
    st.sidebar.success(weave_result["message"])
elif weave_result["status"] == "error":
    st.sidebar.error(weave_result["message"])
    st.sidebar.info(weave_result["fallback"])
else:
    st.sidebar.info(weave_result["message"])

if rag_init_result["status"] == "success" and rag_init_result["has_vector_db"]:
    st.sidebar.success("🔍 RAG system initialized")
elif rag_init_result["status"] == "success":
    st.sidebar.warning("⚠️ RAG processor created but vector DB unavailable")
else:
    st.sidebar.error(f"❌ RAG initialization failed: {rag_init_result.get('error', 'Unknown error')}")

## Lets create a sidebar with a dropdown for the model list and providers
with st.sidebar:
    st.title("Settings")

    #Dropdown for model
    provider = st.selectbox("Provider", ["OpenAI", "Groq", "Google"])
    if provider == "OpenAI":
        model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])
    elif provider == "Groq":
        model_name = st.selectbox("Model", ["llama-3.3-70b-versatile"])
    else:
        model_name = st.selectbox("Model", ["gemini-2.0-flash"])

    # Save provider and model to session state
    st.session_state.provider = provider
    st.session_state.model_name = model_name
    
    st.divider()
    
    # Configuration sliders
    st.subheader("Model Configuration")
    
    # Temperature slider
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Controls randomness: 0 = focused, 2 = very creative"
    )
    st.session_state.temperature = temperature
    
    # Max tokens slider
    max_tokens = st.slider(
        "Max Tokens",
        min_value=50,
        max_value=2000,
        value=500,
        step=50,
        help="Maximum length of the response"
    )
    st.session_state.max_tokens = max_tokens
    
    # Top-p slider
    top_p = st.slider(
        "Top-p (Nucleus Sampling)",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
        help="Controls diversity: lower values = more focused, higher values = more diverse"
    )
    st.session_state.top_p = top_p
    
    # Top-k slider
    top_k = st.slider(
        "Top-k",
        min_value=1,
        max_value=100,
        value=40,
        step=1,
        help="Limits vocabulary to top k tokens (Google only)"
    )
    st.session_state.top_k = top_k
    
    # Display current configuration
    st.divider()
    st.caption(f"🎛️ Config: Temp={temperature} | Tokens={max_tokens} | Top-p={top_p} | Top-k={top_k}")
    
    # Parameter support info
    if provider == "Google":
        st.caption("✅ All parameters supported")
    else:
        st.caption("⚠️ Top-k not supported by OpenAI/Groq")
    
    # RAG Settings
    st.divider()
    st.subheader("RAG Settings")
    
    # RAG toggle
    use_rag = st.checkbox(
        "Enable RAG (Product Search)",
        value=True,
        help="Use product database for enhanced responses about electronics"
    )
    st.session_state.use_rag = use_rag
    
    if use_rag and st.session_state.get('rag_processor'):
        # RAG configuration options
        max_products = st.slider(
            "Max Products",
            min_value=1,
            max_value=10,
            value=5,
            help="Maximum number of products to retrieve"
        )
        st.session_state.max_products = max_products
        
        max_reviews = st.slider(
            "Max Reviews",
            min_value=1,
            max_value=8,
            value=3,
            help="Maximum number of review summaries to retrieve"
        )
        st.session_state.max_reviews = max_reviews
        
        st.caption("🛍️ RAG will search electronics database for relevant context")
    elif use_rag and not st.session_state.get('rag_processor'):
        st.error("RAG system not available - check data files")
    else:
        st.caption("🤖 Standard chatbot mode")


if st.session_state.provider == "OpenAI":
    client = OpenAI(api_key=config.OPENAI_API_KEY)
elif st.session_state.provider == "Groq":
    client = Groq(api_key=config.GROQ_API_KEY)
else:
    client = genai.Client(api_key=config.GOOGLE_API_KEY)

# Set up page title and description
st.title("🛍️ Amazon Electronics Assistant")
if st.session_state.get('rag_processor'):
    st.caption("Ask me about electronics products, reviews, comparisons, and recommendations!")
else:
    st.caption("General AI assistant (RAG not available)")

# Add some example queries for RAG
if st.session_state.get('rag_processor') and st.session_state.get('use_rag', False):
    with st.expander("💡 Try these example queries"):
        example_queries = [
            "What do people say about iPhone charger cables?",
            "Compare Fire TV and regular tablets",
            "Is the Ethernet cable good for gaming?",
            "What are the main complaints about laptop backpacks?",
            "Recommend a budget-friendly tablet under $100"
        ]
        
        cols = st.columns(2)
        for i, query in enumerate(example_queries):
            with cols[i % 2]:
                if st.button(query, key=f"example_{i}"):
                    st.session_state.prefilled_query = query
                    st.rerun()

if "messages" not in st.session_state:
    welcome_msg = "Hello! I'm your Amazon Electronics Assistant. "
    if st.session_state.get('rag_processor'):
        welcome_msg += "Ask me about electronics products, reviews, comparisons, and recommendations from our database of 1,000 products and 20,000 reviews!"
    else:
        welcome_msg += "How can I assist you today?"
    
    st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize query input state
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

# Handle prefilled query from example buttons
if "prefilled_query" in st.session_state:
    st.session_state.query_input = st.session_state.prefilled_query
    del st.session_state.prefilled_query  # Clear after using

# Create a form for the query input
with st.form(key="query_form", clear_on_submit=True):
    query_input = st.text_input(
        "Ask about electronics products, reviews, or comparisons...",
        value=st.session_state.query_input,
        placeholder="Type your question here or click an example above",
        key="query_text_input"
    )
    submit_button = st.form_submit_button("Send")
    
    if submit_button and query_input.strip():
        # Clear the session state input
        st.session_state.query_input = ""
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query_input})
        with st.chat_message("user"):
            st.markdown(query_input)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..." if not st.session_state.get('use_rag', False) else "Searching products and reviews..."):
                output = run_llm(client, st.session_state.messages)
                st.write(output)
        st.session_state.messages.append({"role": "assistant", "content": output})
        st.rerun()