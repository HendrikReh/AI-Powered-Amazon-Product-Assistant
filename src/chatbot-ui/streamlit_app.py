import streamlit as st
import weave
from openai import OpenAI
from groq import Groq
from google import genai
import ollama
import sys
import time
from pathlib import Path
from typing import List

# Add the parent directory to sys.path to import from rag module
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from core.config import config
from rag.query_processor import create_rag_processor

# Import enhanced tracing utilities
try:
    from tracing.trace_utils import (
        create_enhanced_trace_context, get_current_trace_context,
        update_trace_context, business_analyzer
    )
    from tracing.business_intelligence import (
        track_business_interaction, get_business_session_summary,
        business_tracker
    )
except ImportError:
    # Fallback for cases where tracing utils are not available
    create_enhanced_trace_context = lambda **kwargs: None
    get_current_trace_context = lambda: None
    update_trace_context = lambda **kwargs: None
    business_analyzer = None
    track_business_interaction = lambda *args, **kwargs: {}
    get_business_session_summary = lambda *args: {}
    business_tracker = None

def get_llm_client():
    """Get or create LLM client based on current provider."""
    provider = st.session_state.get('provider', 'OpenAI')
    if provider == "OpenAI":
        return OpenAI(api_key=config.OPENAI_API_KEY)
    elif provider == "Groq":
        return Groq(api_key=config.GROQ_API_KEY)
    elif provider == "Ollama":
        return ollama.Client(host=config.OLLAMA_BASE_URL)
    else:
        return genai.Client(api_key=config.GOOGLE_API_KEY)

def get_query_suggestions(partial_query: str, rag_processor) -> List[str]:
    """Generate query suggestions based on partial input and product database."""
    if not rag_processor or not partial_query or len(partial_query) < 3:
        return []
    
    # Common product terms that might trigger suggestions
    product_terms = {
        "iphone": ["iPhone charger cables", "iPhone accessories", "iPhone cases"],
        "cable": ["USB cables", "Ethernet cables", "charging cables", "Lightning cables"],
        "headphone": ["wireless headphones", "noise-canceling headphones", "gaming headphones"],
        "tablet": ["budget tablets", "iPad alternatives", "Android tablets"],
        "laptop": ["laptop backpacks", "laptop accessories", "budget laptops"],
        "router": ["wireless routers", "gaming routers", "mesh routers"],
        "charger": ["phone chargers", "wireless chargers", "fast chargers"],
        "speaker": ["Bluetooth speakers", "smart speakers", "portable speakers"],
        "keyboard": ["mechanical keyboards", "gaming keyboards", "wireless keyboards"],
        "mouse": ["gaming mice", "wireless mice", "ergonomic mice"]
    }
    
    suggestions = []
    partial_lower = partial_query.lower()
    
    for term, product_suggestions in product_terms.items():
        if term in partial_lower:
            for suggestion in product_suggestions:
                if suggestion.lower() not in partial_lower:
                    suggestions.append(f"What do people say about {suggestion}?")
                    suggestions.append(f"Compare {suggestion} with alternatives")
                    if len(suggestions) >= 6:  # Limit suggestions
                        break
            break
    
    return suggestions[:6]  # Maximum 6 suggestions

def display_enhanced_response(response: str, rag_context=None):
    """Display response with enhanced formatting and context cards."""
    
    # Display the main response
    st.markdown(response)
    
    # If RAG context is available, show additional context
    if rag_context and hasattr(st.session_state, 'last_rag_result'):
        rag_result = st.session_state.last_rag_result
        context = rag_result["context"]
        
        if context["num_products"] > 0 or context["num_reviews"] > 0:
            with st.expander("🔍 Retrieved Context", expanded=False):
                
                # Create tabs for products and reviews
                if context["num_products"] > 0 and context["num_reviews"] > 0:
                    prod_tab, review_tab = st.tabs(["📦 Products", "⭐ Reviews"])
                elif context["num_products"] > 0:
                    prod_tab = st.container()
                    review_tab = None
                elif context["num_reviews"] > 0:
                    review_tab = st.container()
                    prod_tab = None
                
                # Display product cards
                if context["num_products"] > 0 and prod_tab is not None:
                    with prod_tab:
                        st.write(f"**Found {context['num_products']} relevant products:**")
                        
                        # Create a simple product card layout
                        for i in range(min(context["num_products"], 3)):  # Show max 3 products
                            with st.container():
                                st.write(f"**Product {i+1}**")
                                # Note: In a real implementation, you'd extract this from the RAG context
                                # For now, we'll show a placeholder since the actual product data 
                                # would need to be passed through the context
                                st.caption("📊 Product information retrieved from database")
                                st.divider()
                
                # Display review summaries
                if context["num_reviews"] > 0 and review_tab is not None:
                    with review_tab:
                        st.write(f"**Found {context['num_reviews']} relevant review summaries:**")
                        
                        # Create review summary cards
                        for i in range(min(context["num_reviews"], 3)):  # Show max 3 reviews
                            with st.container():
                                st.write(f"**Review Summary {i+1}**")
                                st.caption("💬 Customer feedback summary from database")
                                st.divider()
                
                # Show query analysis
                st.subheader("🎯 Query Analysis")
                analysis_col1, analysis_col2 = st.columns(2)
                
                with analysis_col1:
                    st.write(f"**Query Type:** {context['query_type']}")
                    st.write(f"**Processing Time:** {rag_result['processing_time_ms']}ms")
                
                with analysis_col2:
                    if context.get('extracted_terms'):
                        st.write(f"**Extracted Terms:** {', '.join(context['extracted_terms'])}")
                    else:
                        st.write("**Extracted Terms:** None detected")

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

@weave.op()
def apply_rag_enhancement(query: str, rag_processor, max_products: int, max_reviews: int, session_id: str = None, conversation_turn: int = 0):
    """Apply RAG enhancement to user query with comprehensive tracing."""
    start_time = time.time()
    
    # Create enhanced trace context
    trace_context = create_enhanced_trace_context(
        session_id=session_id or str(st.session_state.get('session_id', 'default')),
        conversation_turn=conversation_turn
    )
    
    try:
        # Build context with enhanced tracing
        context = rag_processor.build_context(
            query,
            max_products=max_products,
            max_reviews=max_reviews,
            trace_id=trace_context.trace_id if trace_context else None
        )
        
        rag_applied = bool(context.products or context.reviews)
        enhanced_prompt = None
        
        if rag_applied:
            enhanced_prompt = rag_processor.generate_rag_prompt(context)
        
        processing_time = time.time() - start_time
        
        # Enhanced result with comprehensive context
        result = {
            "status": "success",
            "rag_applied": rag_applied,
            "context": {
                "query_type": context.query_type,
                "num_products": len(context.products),
                "num_reviews": len(context.reviews),
                "extracted_terms": context.metadata.get("extracted_terms", []),
                "performance_metrics": context.metadata.get("performance_metrics", {})
            },
            "enhanced_prompt": enhanced_prompt,
            "processing_time_ms": round(processing_time * 1000, 2),
            "original_query": query,
            "trace_metadata": {
                "trace_id": trace_context.trace_id if trace_context else None,
                "session_id": trace_context.session_id if trace_context else session_id,
                "conversation_turn": conversation_turn
            }
        }
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        return {
            "status": "error",
            "error": str(e),
            "processing_time_ms": round(processing_time * 1000, 2),
            "original_query": query,
            "fallback": "Using original query",
            "trace_metadata": {
                "trace_id": trace_context.trace_id if trace_context else None,
                "session_id": trace_context.session_id if trace_context else session_id,
                "conversation_turn": conversation_turn
            }
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
        elif provider == "Ollama":
            # Ollama API format
            response = client.chat(
                model=model_name,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )["message"]["content"]
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
    """Enhanced run_llm function with comprehensive business intelligence tracing."""
    overall_start_time = time.time()
    
    # Get configuration from session state
    provider = st.session_state.get('provider', 'OpenAI')
    model_name = st.session_state.get('model_name', 'gpt-4o-mini')
    temperature = st.session_state.get('temperature', 0.7)
    max_tokens = st.session_state.get('max_tokens', 500)
    top_p = st.session_state.get('top_p', 1.0)
    top_k = st.session_state.get('top_k', 40)
    use_rag = st.session_state.get('use_rag', False)
    
    # Session management for enhanced tracing
    session_id = st.session_state.get('session_id', str(time.time()))
    if 'session_id' not in st.session_state:
        st.session_state.session_id = session_id
    
    conversation_turn = len([m for m in messages if m['role'] == 'user'])
    
    # Initialize response structure
    trace_data = {
        "provider": provider,
        "model": model_name,
        "use_rag": use_rag,
        "message_count": len(messages),
        "session_id": session_id,
        "conversation_turn": conversation_turn
    }
    
    # Prepare messages for LLM
    llm_messages = messages.copy()
    rag_result = None
    business_intelligence = None
    
    # Apply RAG if enabled and user message exists
    if use_rag and st.session_state.rag_processor and len(messages) > 0:
        latest_message = messages[-1]
        if latest_message["role"] == "user":
            # Get RAG configuration
            max_products = st.session_state.get('max_products', 5)
            max_reviews = st.session_state.get('max_reviews', 3)
            
            # Apply RAG enhancement with enhanced tracing
            rag_result = apply_rag_enhancement(
                latest_message["content"],
                st.session_state.rag_processor,
                max_products,
                max_reviews,
                session_id,
                conversation_turn
            )
            
            trace_data["rag_result"] = rag_result
            
            # Update message if RAG enhancement was successful
            if rag_result["status"] == "success" and rag_result["rag_applied"]:
                llm_messages[-1] = {
                    "role": "user",
                    "content": rag_result["enhanced_prompt"]
                }
                
                # Store RAG context info for monitoring tab
                st.session_state.last_rag_result = rag_result
            
            elif rag_result["status"] == "error":
                # Store RAG error for monitoring tab
                st.session_state.last_rag_error = rag_result["error"]
    
    # Call LLM provider with tracing
    llm_result = call_llm_provider(
        provider, model_name, llm_messages, 
        temperature, max_tokens, top_p, top_k, client
    )
    
    total_time = time.time() - overall_start_time
    
    # Business Intelligence Tracking
    if llm_result["status"] == "success" and len(messages) > 0:
        latest_user_message = messages[-1]["content"] if messages[-1]["role"] == "user" else ""
        response = llm_result["response"]
        context = rag_result.get("context", {}) if rag_result else {}
        
        # Track business intelligence
        try:
            business_intelligence = track_business_interaction(
                query=latest_user_message,
                response=response,
                context=context,
                session_id=session_id,
                conversation_turn=conversation_turn
            )
            
            # Store business intelligence for monitoring
            st.session_state.last_business_intelligence = business_intelligence
            
        except Exception as e:
            st.session_state.last_business_intelligence = {"error": str(e)}
    
    # Combine all trace data
    final_trace = {
        **trace_data,
        "llm_result": llm_result,
        "business_intelligence": business_intelligence,
        "total_time_ms": round(total_time * 1000, 2),
        "success": llm_result["status"] == "success"
    }
    
    # Store enhanced performance metrics for monitoring tab
    if llm_result["status"] == "success":
        st.session_state.last_performance = {
            "total_time_ms": final_trace['total_time_ms'],
            "rag_time_ms": rag_result['processing_time_ms'] if rag_result and rag_result["status"] == "success" else 0,
            "llm_time_ms": llm_result['response_time_ms'],
            "business_metrics": business_intelligence.get("business_metrics", {}) if business_intelligence else {}
        }
    
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

# Set up page title and description
st.title("🛍️ Amazon Electronics Assistant")
if st.session_state.get('rag_processor'):
    st.caption("Ask me about electronics products, reviews, comparisons, and recommendations!")
else:
    st.caption("General AI assistant (RAG not available)")

# Create main tab interface
tab_query, tab_config, tab_monitoring = st.tabs(["💬 Query", "🔧 Configuration", "📊 Monitoring"])

with tab_config:
    st.header("System Configuration")
    
    # System Status Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 System Status")
        
        # Weave Status
        if weave_result["status"] == "success":
            st.success(weave_result["message"])
        elif weave_result["status"] == "error":
            st.error(weave_result["message"])
            st.info(weave_result["fallback"])
        else:
            st.info(weave_result["message"])
        
        # RAG Status
        if rag_init_result["status"] == "success" and rag_init_result["has_vector_db"]:
            st.success("🔍 RAG system initialized")
        elif rag_init_result["status"] == "success":
            st.warning("⚠️ RAG processor created but vector DB unavailable")
        else:
            st.error(f"❌ RAG initialization failed: {rag_init_result.get('error', 'Unknown error')}")
    
    with col2:
        st.subheader("🤖 Model Selection")
        
        # Provider and model selection
        provider = st.selectbox("Provider", ["OpenAI", "Groq", "Google", "Ollama"])
        
        if provider == "OpenAI":
            model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])
        elif provider == "Groq":
            model_name = st.selectbox("Model", ["llama-3.3-70b-versatile"])
        elif provider == "Ollama":
            model_name = st.selectbox("Model", ["gemma3n:e4b", "llama3.2:latest"], help="Make sure the model is available in your Ollama installation")
        else:
            model_name = st.selectbox("Model", ["gemini-2.0-flash"])

        # Save provider and model to session state
        st.session_state.provider = provider
        st.session_state.model_name = model_name
        
        # Parameter support info
        if provider == "Google":
            st.success("✅ All parameters supported")
        elif provider == "Ollama":
            st.success("✅ Temperature supported (local model)")
        else:
            st.warning("⚠️ Top-k not supported by OpenAI/Groq")
    
    st.divider()
    
    # Model Parameters Section
    st.subheader("⚙️ Model Parameters")
    
    param_col1, param_col2 = st.columns(2)
    
    with param_col1:
        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.get('temperature', 0.7),
            step=0.1,
            help="Controls randomness: 0 = focused, 2 = very creative"
        )
        st.session_state.temperature = temperature
        
        # Max tokens slider
        max_tokens = st.slider(
            "Max Tokens",
            min_value=50,
            max_value=2000,
            value=st.session_state.get('max_tokens', 500),
            step=50,
            help="Maximum length of the response"
        )
        st.session_state.max_tokens = max_tokens
    
    with param_col2:
        # Top-p slider
        top_p = st.slider(
            "Top-p (Nucleus Sampling)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('top_p', 1.0),
            step=0.05,
            help="Controls diversity: lower values = more focused, higher values = more diverse"
        )
        st.session_state.top_p = top_p
        
        # Top-k slider
        top_k = st.slider(
            "Top-k",
            min_value=1,
            max_value=100,
            value=st.session_state.get('top_k', 40),
            step=1,
            help="Limits vocabulary to top k tokens (Google only)"
        )
        st.session_state.top_k = top_k
    
    # Display current configuration
    st.info(f"🎛️ Current Config: Temp={temperature} | Tokens={max_tokens} | Top-p={top_p} | Top-k={top_k}")
    
    st.divider()
    
    # RAG Configuration Section
    st.subheader("🔍 RAG Configuration")
    
    rag_col1, rag_col2 = st.columns(2)
    
    with rag_col1:
        # RAG toggle
        use_rag = st.checkbox(
            "Enable RAG (Product Search)",
            value=st.session_state.get('use_rag', True),
            help="Use product database for enhanced responses about electronics"
        )
        st.session_state.use_rag = use_rag
        
        if use_rag and st.session_state.get('rag_processor'):
            st.success("🛍️ RAG will search electronics database for relevant context")
        elif use_rag and not st.session_state.get('rag_processor'):
            st.error("RAG system not available - check data files")
        else:
            st.info("🤖 Standard chatbot mode")
    
    with rag_col2:
        if use_rag and st.session_state.get('rag_processor'):
            # RAG configuration options
            max_products = st.slider(
                "Max Products",
                min_value=1,
                max_value=10,
                value=st.session_state.get('max_products', 5),
                help="Maximum number of products to retrieve"
            )
            st.session_state.max_products = max_products
            
            max_reviews = st.slider(
                "Max Reviews",
                min_value=1,
                max_value=8,
                value=st.session_state.get('max_reviews', 3),
                help="Maximum number of review summaries to retrieve"
            )
            st.session_state.max_reviews = max_reviews

with tab_query:
    st.header("Chat Interface")
    
    # Enhanced example queries with categories
    if st.session_state.get('rag_processor') and st.session_state.get('use_rag', False):
        with st.expander("💡 Example Queries by Category", expanded=True):
            
            # Organize examples by category
            example_categories = {
                "🔍 Product Information": [
                    "What are the key features of iPhone charging cables?",
                    "Tell me about Fire TV Stick performance and capabilities"
                ],
                "⭐ Reviews & Feedback": [
                    "What do people say about iPhone charger cables?",
                    "What are customer experiences with Bluetooth earbuds?"
                ],
                "⚖️ Product Comparisons": [
                    "Compare Fire TV and regular tablets",
                    "Compare Ethernet cables vs USB cables for data transfer"
                ],
                "❗ Common Complaints": [
                    "What are the main complaints about laptop backpacks?",
                    "What problems do people have with wireless routers?"
                ],
                "💰 Budget Recommendations": [
                    "Recommend a budget-friendly tablet under $100",
                    "Suggest affordable alternatives to expensive noise-canceling headphones"
                ],
                "🎯 Use Case Evaluation": [
                    "Is the Ethernet cable good for gaming?",
                    "Can a smartwatch be used for fitness tracking effectively?"
                ]
            }
            
            for category, queries in example_categories.items():
                st.write(f"**{category}**")
                cols = st.columns(len(queries))
                for i, query in enumerate(queries):
                    with cols[i]:
                        if st.button(query, key=f"example_{category}_{i}", help="Click to use this query"):
                            st.session_state.prefilled_query = query
                            st.rerun()
                st.write("")  # Add some spacing
    
    # Query History
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    
    if st.session_state.query_history:
        with st.expander("🕐 Recent Queries"):
            selected_history = st.selectbox(
                "Select a previous query:",
                options=[""] + st.session_state.query_history[-10:],  # Last 10 queries
                index=0,
                key="history_selector"
            )
            if selected_history:
                st.session_state.prefilled_query = selected_history
                st.rerun()
    
    # Quick Filters (when RAG is enabled)
    if st.session_state.get('rag_processor') and st.session_state.get('use_rag', False):
        st.subheader("🎛️ Quick Filters")
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            query_type_filter = st.selectbox(
                "Query Type",
                ["Any", "Product Info", "Reviews", "Comparison", "Complaints", "Recommendations", "Use Case"],
                key="query_type_filter"
            )
        
        with filter_col2:
            category_filter = st.selectbox(
                "Category",
                ["Any", "Cables", "Audio", "Tablets", "Networking", "Gaming", "Accessories"],
                key="category_filter"
            )
        
        with filter_col3:
            price_filter = st.selectbox(
                "Price Range",
                ["Any", "Under $50", "$50-$100", "$100-$200", "Over $200"],
                key="price_filter"
            )
        
        # Apply filters button
        if st.button("🔍 Apply Filters to Next Query", help="Filters will be applied to your next question"):
            filter_context = []
            if query_type_filter != "Any":
                filter_context.append(f"Focus on {query_type_filter.lower()}")
            if category_filter != "Any":
                filter_context.append(f"in {category_filter.lower()} category")
            if price_filter != "Any":
                filter_context.append(f"with price range {price_filter.lower()}")
            
            if filter_context:
                st.session_state.active_filters = " ".join(filter_context)
                st.success(f"Filters applied: {st.session_state.active_filters}")
    
    st.divider()
    
    # Chat History Display
    if "messages" not in st.session_state:
        welcome_msg = "Hello! I'm your Amazon Electronics Assistant. "
        if st.session_state.get('rag_processor'):
            welcome_msg += "Ask me about electronics products, reviews, comparisons, and recommendations from our database of 1,000 products and 20,000 reviews!"
        else:
            welcome_msg += "How can I assist you today?"
        
        st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]

    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # Use enhanced display for assistant messages when RAG is enabled
            if (message["role"] == "assistant" and 
                i > 0 and  # Not the welcome message
                st.session_state.get('use_rag', False)):
                display_enhanced_response(message["content"], rag_context=False)
            else:
                st.markdown(message["content"])

    # Initialize query input state
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""

    # Handle prefilled query from example buttons
    if "prefilled_query" in st.session_state:
        st.session_state.query_input = st.session_state.prefilled_query
        del st.session_state.prefilled_query  # Clear after using

    # Build dynamic placeholder text
    placeholder_text = "Type your question here..."
    if st.session_state.get('rag_processor') and st.session_state.get('use_rag', False):
        placeholder_text = "Ask about electronics products, reviews, or comparisons..."
    
    # Add filter context to placeholder if active
    if hasattr(st.session_state, 'active_filters'):
        placeholder_text = f"{placeholder_text} (Filters: {st.session_state.active_filters})"
    
    # Query input (outside form to allow suggestions)
    query_input = st.text_input(
        "Your Question:",
        value=st.session_state.query_input,
        placeholder=placeholder_text,
        key="query_text_input",
        help="Ask about products, reviews, comparisons, or use the examples above"
    )
    
    # Show query suggestions if user is typing (outside form)
    if query_input and len(query_input) >= 3 and st.session_state.get('rag_processor'):
        suggestions = get_query_suggestions(query_input, st.session_state.rag_processor)
        if suggestions:
            st.write("💡 **Suggestions:**")
            suggestion_cols = st.columns(min(len(suggestions), 3))
            for i, suggestion in enumerate(suggestions):
                with suggestion_cols[i % 3]:
                    if st.button(suggestion, key=f"suggestion_{i}", help="Click to use this suggestion"):
                        st.session_state.query_input = suggestion
                        st.rerun()
    
    # Form for submit and clear buttons only
    with st.form(key="query_form", clear_on_submit=False):
        submit_col, clear_col = st.columns([3, 1])
        with submit_col:
            submit_button = st.form_submit_button("🚀 Send", use_container_width=True)
        with clear_col:
            clear_button = st.form_submit_button("🗑️ Clear", use_container_width=True)
        
    if clear_button:
        st.session_state.messages = [st.session_state.messages[0]]  # Keep welcome message
        st.session_state.query_input = ""  # Clear query input
        if hasattr(st.session_state, 'active_filters'):
            del st.session_state.active_filters
        st.rerun()
    
    if submit_button and query_input.strip():
        # Add active filters to query if they exist
        final_query = query_input
        if hasattr(st.session_state, 'active_filters'):
            final_query = f"{query_input} ({st.session_state.active_filters})"
            del st.session_state.active_filters  # Clear after use
        
        # Add to query history
        if query_input not in st.session_state.query_history:
            st.session_state.query_history.append(query_input)
        
        # Clear the session state input
        st.session_state.query_input = ""
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": final_query})
        with st.chat_message("user"):
            st.markdown(final_query)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..." if not st.session_state.get('use_rag', False) else "Searching products and reviews..."):
                client = get_llm_client()
                output = run_llm(client, st.session_state.messages)
                
                # Use enhanced display if RAG was used
                if st.session_state.get('use_rag', False):
                    display_enhanced_response(output, rag_context=True)
                else:
                    st.write(output)
        
        st.session_state.messages.append({"role": "assistant", "content": output})
        st.rerun()

with tab_monitoring:
    st.header("Performance Monitoring")
    
    # Performance Metrics Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Session Statistics")
        
        # Calculate session stats
        total_messages = len(st.session_state.get('messages', [])) - 1  # Exclude welcome message
        user_messages = len([m for m in st.session_state.get('messages', []) if m['role'] == 'user'])
        
        st.metric("Total Messages", total_messages)
        st.metric("User Queries", user_messages)
        st.metric("Query History", len(st.session_state.get('query_history', [])))
        
        # System Configuration Status
        st.subheader("⚙️ Current Configuration")
        current_provider = st.session_state.get('provider', 'Not Set')
        current_model = st.session_state.get('model_name', 'Not Set')
        rag_enabled = st.session_state.get('use_rag', False)
        
        st.write(f"**Provider:** {current_provider}")
        st.write(f"**Model:** {current_model}")
        st.write(f"**RAG Status:** {'🟢 Enabled' if rag_enabled else '🔴 Disabled'}")
        st.write(f"**Temperature:** {st.session_state.get('temperature', 0.7)}")
        st.write(f"**Max Tokens:** {st.session_state.get('max_tokens', 500)}")
    
    with col2:
        st.subheader("🔍 Recent Activity")
        
        # Show recent queries if any
        recent_queries = st.session_state.get('query_history', [])[-5:]  # Last 5 queries
        if recent_queries:
            for i, query in enumerate(reversed(recent_queries), 1):
                st.write(f"{i}. {query[:50]}{'...' if len(query) > 50 else ''}")
        else:
            st.write("No queries yet")
        
        # Enhanced RAG Performance (if available)
        if st.session_state.get('rag_processor') and st.session_state.get('use_rag', False):
            st.subheader("🛍️ Enhanced RAG Performance")
            st.write(f"**Max Products:** {st.session_state.get('max_products', 5)}")
            st.write(f"**Max Reviews:** {st.session_state.get('max_reviews', 3)}")
            st.write("**Database:** 1,000 products, 20,000 reviews")
            
            # Display last RAG result if available
            if hasattr(st.session_state, 'last_rag_result'):
                rag_res = st.session_state.last_rag_result
                context = rag_res["context"]
                st.success(f"Last Query: Found {context['num_products']} products, {context['num_reviews']} reviews")
                st.caption(f"Query type: {context['query_type']}")
                st.caption(f"Processing time: {rag_res['processing_time_ms']}ms")
                
                # Display performance metrics if available
                perf_metrics = context.get('performance_metrics', {})
                if perf_metrics:
                    with st.expander("🔍 Vector Database Performance", expanded=False):
                        for search_type, metrics in perf_metrics.items():
                            if isinstance(metrics, dict) and 'embedding_metrics' in metrics:
                                st.write(f"**{search_type.title()} Search:**")
                                emb_metrics = metrics['embedding_metrics']
                                search_metrics = metrics['search_metrics']
                                quality_metrics = metrics['quality_metrics']
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Embedding Time", f"{emb_metrics.get('embedding_time_ms', 0):.1f}ms")
                                with col2:
                                    st.metric("Search Time", f"{search_metrics.get('search_time_ms', 0):.1f}ms")
                                with col3:
                                    st.metric("Relevance Score", f"{quality_metrics.get('relevance_score', 0):.2f}")
            
            # Display RAG error if any
            if hasattr(st.session_state, 'last_rag_error'):
                st.error(f"Last RAG Error: {st.session_state.last_rag_error}")
        
        # Business Intelligence Metrics
        if hasattr(st.session_state, 'last_business_intelligence'):
            business_intel = st.session_state.last_business_intelligence
            
            if "error" not in business_intel:
                st.subheader("📊 Business Intelligence")
                
                # User Journey Insights
                user_journey = business_intel.get('user_journey', {})
                if user_journey:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("User Type", user_journey.get('user_type', 'Unknown').title())
                        st.metric("Journey Stage", user_journey.get('journey_stage', 'Unknown').title())
                    with col2:
                        st.metric("Queries in Session", user_journey.get('queries_count', 0))
                        avg_satisfaction = sum(user_journey.get('satisfaction_scores', [0])) / max(len(user_journey.get('satisfaction_scores', [1])), 1)
                        st.metric("Avg Satisfaction", f"{avg_satisfaction:.2f}")
                
                # Business Metrics
                business_metrics = business_intel.get('business_metrics', {})
                if business_metrics:
                    with st.expander("📈 Detailed Business Metrics", expanded=False):
                        metric_col1, metric_col2 = st.columns(2)
                        with metric_col1:
                            st.metric("Conversion Potential", f"{business_metrics.get('conversion_potential', 0):.2f}")
                            st.metric("Response Quality", f"{business_metrics.get('response_quality_score', 0):.2f}")
                        with metric_col2:
                            st.metric("Recommendation Effectiveness", f"{business_metrics.get('recommendation_effectiveness', 0):.2f}")
                            st.metric("Query Success Rate", f"{business_metrics.get('query_success_rate', 0):.2f}")
                
                # Feature Usage
                feature_usage = business_intel.get('feature_usage', {})
                if feature_usage:
                    with st.expander("🎯 Feature Usage Analytics", expanded=False):
                        for feature, count in feature_usage.items():
                            st.write(f"**{feature.replace('_', ' ').title()}:** {count}")
    
    # Enhanced Real-time Performance Metrics
    if hasattr(st.session_state, 'last_performance'):
        st.divider()
        st.subheader("⚡ Latest Query Performance")
        
        perf = st.session_state.last_performance
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        with perf_col1:
            st.metric("Total Time", f"{perf['total_time_ms']}ms")
        with perf_col2:
            st.metric("RAG Time", f"{perf['rag_time_ms']}ms")
        with perf_col3:
            st.metric("LLM Time", f"{perf['llm_time_ms']}ms")
        
        # Performance breakdown chart
        if perf['rag_time_ms'] > 0:
            rag_percentage = (perf['rag_time_ms'] / perf['total_time_ms']) * 100
            llm_percentage = (perf['llm_time_ms'] / perf['total_time_ms']) * 100
            st.caption(f"Breakdown: RAG {rag_percentage:.1f}% | LLM {llm_percentage:.1f}%")
        
        # Business performance metrics
        business_metrics = perf.get('business_metrics', {})
        if business_metrics:
            st.subheader("📊 Business Performance Metrics")
            biz_col1, biz_col2, biz_col3, biz_col4 = st.columns(4)
            
            with biz_col1:
                st.metric("User Satisfaction", f"{business_metrics.get('user_satisfaction_prediction', 0):.2f}")
            with biz_col2:
                st.metric("Conversion Potential", f"{business_metrics.get('conversion_potential', 0):.2f}")
            with biz_col3:
                st.metric("Response Quality", f"{business_metrics.get('response_quality_score', 0):.2f}")
            with biz_col4:
                st.metric("Success Rate", f"{business_metrics.get('query_success_rate', 0):.2f}")
    
    st.divider()
    
    # Weave Tracing Integration
    if weave_result["status"] == "success":
        st.subheader("📈 Weave Tracing Dashboard")
        st.success("Weave tracing is active! Visit your [W&B Dashboard](https://wandb.ai) to view detailed traces.")
        st.info("Navigate to the 'Bootcamp' project to see:")
        st.write("- RAG query processing metrics")
        st.write("- LLM provider performance")
        st.write("- Response timing analysis")
        st.write("- Error tracking and debugging")
    else:
        st.subheader("📈 Weave Tracing")
        st.warning("Weave tracing is not active. Add WANDB_API_KEY to enable detailed monitoring.")
    
    # Session Analytics
    if hasattr(st.session_state, 'session_id') and business_tracker:
        try:
            session_summary = get_business_session_summary(st.session_state.session_id)
            
            if "error" not in session_summary:
                st.subheader("👤 Session Analytics")
                
                session_overview = session_summary.get('session_overview', {})
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**User Type:** {session_overview.get('user_type', 'Unknown').title()}")
                    st.write(f"**Journey Stage:** {session_overview.get('journey_stage', 'Unknown').title()}")
                    st.write(f"**Total Queries:** {session_overview.get('total_queries', 0)}")
                
                with col2:
                    st.write(f"**Session Duration:** {session_overview.get('session_duration', 0):.1f}s")
                    st.write(f"**Avg Satisfaction:** {session_overview.get('avg_satisfaction', 0):.2f}")
                
                # Business recommendations
                recommendations = session_summary.get('recommendations', [])
                if recommendations:
                    with st.expander("💡 Business Recommendations", expanded=False):
                        for i, rec in enumerate(recommendations, 1):
                            st.write(f"{i}. {rec}")
        
        except Exception as e:
            st.warning(f"Session analytics unavailable: {str(e)}")
    
    # System Health Check
    st.subheader("🏥 System Health")
    health_checks = []
    
    # Check API configurations
    if config.OPENAI_API_KEY:
        health_checks.append("✅ OpenAI API configured")
    else:
        health_checks.append("❌ OpenAI API not configured")
    
    if config.GROQ_API_KEY:
        health_checks.append("✅ Groq API configured")
    else:
        health_checks.append("❌ Groq API not configured")
    
    if config.GOOGLE_API_KEY:
        health_checks.append("✅ Google API configured")
    else:
        health_checks.append("❌ Google API not configured")
    
    # Check RAG system
    if rag_init_result["status"] == "success" and rag_init_result["has_vector_db"]:
        health_checks.append("✅ RAG system operational")
    else:
        health_checks.append("❌ RAG system unavailable")
    
    for check in health_checks:
        st.write(check)

