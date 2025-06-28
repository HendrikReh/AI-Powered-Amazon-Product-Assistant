import streamlit as st
import weave
from openai import OpenAI
from groq import Groq
from google import genai

from core.config import config

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



if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hello! How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        output = run_llm(client, st.session_state.messages)
        st.write(output)
    st.session_state.messages.append({"role": "assistant", "content": output})