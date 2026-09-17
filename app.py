import os
import time
import json
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from groq import Groq, GroqError

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Groq AI Chatbot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# ---------------------------------------------------------
# Custom Styling (Modern Dark/Light-adaptive Glassmorphism)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 950px;
    }

    /* Custom Header Badge */
    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
        background: linear-gradient(135deg, #ff7a18 0%, #af002d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #71717a;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Suggestion Cards */
    .suggestion-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 12px;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }

    .suggestion-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 14px 16px;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
    }

    .suggestion-card:hover {
        border-color: #f97316;
        transform: translateY(-2px);
        background: rgba(249, 115, 22, 0.08);
    }

    .suggestion-title {
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .suggestion-desc {
        font-size: 0.8rem;
        color: #a1a1aa;
        line-height: 1.3;
    }

    /* Metric pill */
    .metric-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(249, 115, 22, 0.1);
        border: 1px solid rgba(249, 115, 22, 0.3);
        color: #f97316;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Chat Messages styling */
    [data-testid="stChatMessage"] {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
    }

    /* Footer styling */
    .footer-text {
        text-align: center;
        color: #71717a;
        font-size: 0.8rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_response_time" not in st.session_state:
    st.session_state.last_response_time = None

if "last_prompt_tokens" not in st.session_state:
    st.session_state.last_prompt_tokens = 0

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ Groq Settings")
    st.caption("Powered by Groq's Ultra-Fast LPU™ Inference Engine")
    st.divider()

    # API Key Configuration (Checks environment variable or Streamlit Cloud secrets)
    default_api_key = os.environ.get("GROQ_API_KEY", "")
    if not default_api_key:
        try:
            if "GROQ_API_KEY" in st.secrets:
                default_api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass

    api_key_input = st.text_input(
        "Groq API Key",
        value=default_api_key,
        type="password",
        help="Provide your Groq API key (starts with gsk_)",
        placeholder="gsk_..."
    )

    # Model Selection
    AVAILABLE_MODELS = {
        "llama-3.3-70b-versatile": "llama-3.3-70b-versatile (Flagship High Quality)",
        "llama-3.1-8b-instant": "llama-3.1-8b-instant (Ultra Fast Speed)",
        "openai/gpt-oss-120b": "gpt-oss-120b (High Reasoning & Capability)",
        "openai/gpt-oss-20b": "gpt-oss-20b (Balanced & High Speed)",
        "qwen/qwen3.6-27b": "qwen3.6-27b (Strong Coding & Multilingual)",
        "groq/compound": "groq/compound (Compound Reasoning System)",
        "groq/compound-mini": "groq/compound-mini (Ultra Fast Lightweight)",
    }

    selected_model_key = st.selectbox(
        "AI Model",
        options=list(AVAILABLE_MODELS.keys()),
        format_func=lambda x: AVAILABLE_MODELS.get(x, x),
        index=0
    )

    # System Prompts Presets
    SYSTEM_PRESETS = {
        "🤖 Helpful Assistant": "You are a helpful, knowledgeable, and polite AI assistant. Provide clear, well-structured, and accurate responses.",
        "💻 Expert Software Engineer": "You are a senior software engineer and architect. Provide clean, well-commented code, optimal architectures, best practices, and thorough debugging explanations.",
        "⚡ Ultra Concise": "You are a fast and direct assistant. Keep all answers brief, direct, and straight to the point without unnecessary filler.",
        "✍️ Creative Writer": "You are a creative writer and storyteller. Use rich, evocative language, engaging narratives, and expressive tone.",
        "🎓 Patient Tutor": "You are an encouraging and patient educator. Break down complex topics into intuitive, easy-to-understand explanations with examples and analogies.",
        "⚙️ Custom": ""
    }

    preset_choice = st.selectbox(
        "Persona / System Prompt Preset",
        options=list(SYSTEM_PRESETS.keys()),
        index=0
    )

    if preset_choice == "⚙️ Custom":
        system_prompt = st.text_area(
            "Custom System Prompt",
            value="You are a helpful AI assistant.",
            height=100
        )
    else:
        system_prompt = st.text_area(
            "System Prompt",
            value=SYSTEM_PRESETS[preset_choice],
            height=100
        )

    # Advanced Settings Accordion
    with st.expander("🛠️ Advanced Model Parameters", expanded=False):
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.05,
                                help="Higher values make output more random, lower values more focused and deterministic.")
        max_tokens = st.slider("Max Output Tokens", min_value=128, max_value=8192, value=4096, step=128,
                               help="Maximum number of tokens to generate in response.")
        top_p = st.slider("Top-p (Nucleus Sampling)", min_value=0.0, max_value=1.0, value=1.0, step=0.05)

    st.divider()

    # Chat Actions
    st.markdown("### 🗂️ Chat Management")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_response_time = None
            st.rerun()

    with col2:
        # Download Chat History as JSON
        if st.session_state.messages:
            chat_json = json.dumps(st.session_state.messages, indent=2)
            st.download_button(
                label="📥 Export JSON",
                data=chat_json,
                file_name=f"groq_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.button("📥 Export JSON", disabled=True, use_container_width=True)

    # Markdown Export
    if st.session_state.messages:
        md_text = f"# Groq Chat Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        md_text += f"**Model:** {selected_model_key}\n\n---\n\n"
        for msg in st.session_state.messages:
            role = "🧑 User" if msg["role"] == "user" else "⚡ Assistant"
            md_text += f"### {role}\n{msg['content']}\n\n"
        st.download_button(
            label="📝 Export Markdown",
            data=md_text,
            file_name=f"groq_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )

    # Session Stats
    st.divider()
    st.markdown("### 📊 Session Stats")
    st.markdown(f"- **Messages:** `{len(st.session_state.messages)}`")
    st.markdown(f"- **Active Model:** `{selected_model_key.split('/')[-1]}`")
    if st.session_state.last_response_time is not None:
        st.markdown(f"- **Last Latency:** `{st.session_state.last_response_time:.2f}s`")


# ---------------------------------------------------------
# Helper: Client Initializer
# ---------------------------------------------------------
def get_groq_client(api_key: str):
    if not api_key or not api_key.strip():
        return None
    return Groq(api_key=api_key.strip())


# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown('<div class="header-badge">⚡ Ultra Fast Inference</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Groq AI Assistant</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="hero-subtitle">High-speed conversational intelligence powered by <b>{selected_model_key}</b></p>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Validation: Check API Key
# ---------------------------------------------------------
client = get_groq_client(api_key_input)
if not client:
    st.warning("⚠️ **Groq API Key is required.** Please enter your API key in the sidebar to start chatting.", icon="🔑")
    st.info("💡 You can get a free API key from the [Groq Console](https://console.groq.com/keys).")

# ---------------------------------------------------------
# Empty State: Starter Suggestions
# ---------------------------------------------------------
if len(st.session_state.messages) == 0:
    st.markdown("##### 💡 Try one of these quick starters:")
    
    starter_prompts = [
        {"icon": "⚡", "title": "Explain Groq LPUs", "prompt": "Explain how Groq's Language Processing Unit (LPU) architecture achieves such high inference speeds compared to traditional GPUs."},
        {"icon": "🐍", "title": "Python Async Code", "prompt": "Write a clean Python script demonstrating asynchronous web scraping using `asyncio` and `aiohttp` with error handling."},
        {"icon": "🔬", "title": "Quantum Computing", "prompt": "Explain the core principles of quantum computing (superposition, entanglement, qubits) like I am 15 years old."},
        {"icon": "🚀", "title": "Startup Pitch Idea", "prompt": "Generate 3 innovative B2B SaaS startup ideas leveraging real-time low-latency AI inference, including target audience and monetization models."}
    ]

    cols = st.columns(len(starter_prompts))
    for idx, item in enumerate(starter_prompts):
        with cols[idx]:
            if st.button(f"{item['icon']} **{item['title']}**", key=f"starter_{idx}", use_container_width=True):
                if client:
                    st.session_state.messages.append({"role": "user", "content": item["prompt"]})
                    st.rerun()
                else:
                    st.error("Please enter a valid API key in the sidebar first.")

# ---------------------------------------------------------
# Display Chat History
# ---------------------------------------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="⚡"):
            st.markdown(msg["content"])

# ---------------------------------------------------------
# Handle User Input & Streaming
# ---------------------------------------------------------
user_prompt = st.chat_input("Ask anything... (Shift+Enter for new line)")

# Handle auto-trigger if a starter prompt was clicked
pending_trigger = False
if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
    if len(st.session_state.messages) == 1 or st.session_state.messages[-2]["role"] == "assistant":
        # Check if last user message needs an assistant reply
        pending_trigger = True

if user_prompt:
    if not client:
        st.error("❌ Please provide a valid Groq API Key in the sidebar before sending messages.")
    else:
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_prompt)
        pending_trigger = True

if pending_trigger and client:
    # Prepare messages payload including system prompt
    api_messages = []
    if system_prompt.strip():
        api_messages.append({"role": "system", "content": system_prompt.strip()})
    
    for m in st.session_state.messages:
        api_messages.append({"role": m["role"], "content": m["content"]})

    # Render Assistant Streaming Container
    with st.chat_message("assistant", avatar="⚡"):
        response_placeholder = st.empty()
        full_response = ""
        start_time = time.time()

        try:
            # Stream response from Groq
            stream = client.chat.completions.create(
                model=selected_model_key,
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=True
            )

            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        full_response += delta.content
                        response_placeholder.markdown(full_response + "▌")

            # Final render without cursor
            response_placeholder.markdown(full_response)
            elapsed_time = time.time() - start_time
            st.session_state.last_response_time = elapsed_time

            # Store in session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.rerun()

        except GroqError as ge:
            st.error(f"⚠️ **Groq API Error:** {str(ge)}")
        except Exception as e:
            st.error(f"⚠️ **Unexpected Error:** {str(e)}")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("""
<div class="footer-text">
    Built with 🧡 using <a href="https://streamlit.io" target="_blank" style="color: #f97316; text-decoration: none;">Streamlit</a> & <a href="https://groq.com" target="_blank" style="color: #f97316; text-decoration: none;">Groq</a> LPUs
</div>
""", unsafe_allow_html=True)
