import streamlit as st
import weave
from openai import OpenAI
from groq import Groq
from google import genai
import sys
from pathlib import Path

# Add the parent directory to sys.path to import from rag module
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from core.config import config
from rag.query_processor import create_rag_processor

# Initialize RAG processor
@st.cache_resource
def get_rag_processor():
    """Initialize and cache the RAG processor."""
    try:
        return create_rag_processor()
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return None

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

# Initialize RAG processor
rag_processor = get_rag_processor()
if rag_processor:
    st.sidebar.success("🔍 RAG system initialized")
else:
    st.sidebar.warning("⚠️ RAG system not available")

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
    
    if use_rag and rag_processor:
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
    elif use_rag and not rag_processor:
        st.error("RAG system not available - check data files")
    else:
        st.caption("🤖 Standard chatbot mode")


if st.session_state.provider == "OpenAI":
    client = OpenAI(api_key=config.OPENAI_API_KEY)
elif st.session_state.provider == "Groq":
    client = Groq(api_key=config.GROQ_API_KEY)
else:
    client = genai.Client(api_key=config.GOOGLE_API_KEY)

@weave.op()
def run_llm(client, messages):
    # Get configuration from session state
    temperature = st.session_state.get('temperature', 0.7)
    max_tokens = st.session_state.get('max_tokens', 500)
    top_p = st.session_state.get('top_p', 1.0)
    top_k = st.session_state.get('top_k', 40)
    use_rag = st.session_state.get('use_rag', False)
    
    # Prepare messages for LLM
    llm_messages = messages.copy()
    
    # Apply RAG if enabled and user message exists
    if use_rag and rag_processor and len(messages) > 0:
        latest_message = messages[-1]
        if latest_message["role"] == "user":
            try:
                # Get RAG configuration
                max_products = st.session_state.get('max_products', 5)
                max_reviews = st.session_state.get('max_reviews', 3)
                
                # Build context with custom limits
                context = rag_processor.build_context(
                    latest_message["content"],
                    max_products=max_products,
                    max_reviews=max_reviews
                )
                
                # Generate enhanced prompt if context found
                if context.products or context.reviews:
                    enhanced_prompt = rag_processor.generate_rag_prompt(context)
                    
                    # Replace the user's message with enhanced prompt
                    llm_messages[-1] = {
                        "role": "user",
                        "content": enhanced_prompt
                    }
                    
                    # Show RAG context info in sidebar
                    with st.sidebar:
                        st.success(f"🔍 RAG: Found {len(context.products)} products, {len(context.reviews)} reviews")
                        st.caption(f"Query type: {context.query_type}")
                
            except Exception as e:
                # Fallback to original query if RAG fails
                st.sidebar.error(f"RAG error: {str(e)}")
    
    if st.session_state.provider == "Google":
        return client.models.generate_content(
            model=st.session_state.model_name,
            contents=[message["content"] for message in llm_messages],
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
            messages=llm_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        ).choices[0].message.content



# Set up page title and description
st.title("🛍️ Amazon Electronics Assistant")
if rag_processor:
    st.caption("Ask me about electronics products, reviews, comparisons, and recommendations!")
else:
    st.caption("General AI assistant (RAG not available)")

# Add some example queries for RAG
if rag_processor and st.session_state.get('use_rag', False):
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
    if rag_processor:
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