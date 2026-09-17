import os
import time
import json
from datetime import datetime
from typing import Optional, List, Dict
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 950px;
    }

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

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stChatMessage"] {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
    }

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
    st.session_state["messages"] = []

if "last_response_time" not in st.session_state:
    st.session_state["last_response_time"] = None


# ---------------------------------------------------------
# Helper: Client Initializer
# ---------------------------------------------------------
def get_groq_client(api_key: str) -> Optional[Groq]:
    if not api_key or not api_key.strip():
        return None
    try:
        return Groq(api_key=api_key.strip())
    except Exception:
        return None


# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ Groq Settings")
    st.caption("Powered by Groq's Ultra-Fast LPU™ Inference Engine")
    st.divider()

    # API Key Configuration
    default_api_key = os.environ.get("GROQ_API_KEY", "")
    if not default_api_key:
        try:
            if "GROQ_API_KEY" in st.secrets:
                default_api_key = str(st.secrets["GROQ_API_KEY"])
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
    AVAILABLE_MODELS: Dict[str, str] = {
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
        format_func=lambda x: str(AVAILABLE_MODELS.get(str(x), str(x))),
        index=0
    )

    # System Prompts Presets
    SYSTEM_PRESETS: Dict[str, str] = {
        "🤖 Helpful Assistant": (
            "You are a helpful, knowledgeable, and polite AI assistant. "
            "Provide clear, well-structured, and accurate responses."
        ),
        "💻 Expert Software Engineer": (
            "You are a senior software engineer and architect. "
            "Provide clean, well-commented code, optimal architectures, "
            "best practices, and thorough debugging explanations."
        ),
        "⚡ Ultra Concise": (
            "You are a fast and direct assistant. "
            "Keep all answers brief, direct, and straight to the point."
        ),
        "✍️ Creative Writer": (
            "You are a creative writer and storyteller. "
            "Use rich, evocative language, engaging narratives, and expressive tone."
        ),
        "🎓 Patient Tutor": (
            "You are an encouraging and patient educator. "
            "Break down complex topics into intuitive, easy-to-understand explanations."
        ),
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
            value=SYSTEM_PRESETS.get(preset_choice, ""),
            height=100
        )

    # Advanced Settings Accordion
    with st.expander("🛠️ Advanced Model Parameters", expanded=False):
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.05,
            help="Higher values make output more random, lower values more focused."
        )
        max_tokens = st.slider(
            "Max Output Tokens",
            min_value=128,
            max_value=8192,
            value=4096,
            step=128,
            help="Maximum number of tokens to generate in response."
        )
        top_p = st.slider(
            "Top-p (Nucleus Sampling)",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05
        )

    st.divider()

    # Chat Actions
    st.markdown("### 🗂️ Chat Management")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["last_response_time"] = None
            st.rerun()

    with col2:
        messages_list = st.session_state.get("messages", [])
        if messages_list:
            chat_json = json.dumps(messages_list, indent=2)
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
    if messages_list:
        md_text = f"# Groq Chat Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        md_text += f"**Model:** {selected_model_key}\n\n---\n\n"
        for msg in messages_list:
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
    st.markdown(f"- **Messages:** `{len(messages_list)}`")
    st.markdown(f"- **Active Model:** `{selected_model_key.split('/')[-1]}`")
    last_resp = st.session_state.get("last_response_time")
    if last_resp is not None:
        st.markdown(f"- **Last Latency:** `{last_resp:.2f}s`")


# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown('<div class="header-badge">⚡ Ultra Fast Inference</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Groq AI Assistant</h1>', unsafe_allow_html=True)
st.markdown(
    f'<p class="hero-subtitle">High-speed conversational intelligence powered by <b>{selected_model_key}</b></p>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Check & Connect API Key
# ---------------------------------------------------------
active_key = api_key_input.strip() if api_key_input else ""
if not active_key and "custom_groq_key" in st.session_state:
    active_key = st.session_state["custom_groq_key"]

groq_client = get_groq_client(active_key)

if not groq_client:
    with st.container():
        st.info("👋 **Welcome to Groq AI Chatbot!** Please enter your Groq API key to start chatting:")
        k_col1, k_col2 = st.columns([3, 1])
        with k_col1:
            main_key_in = st.text_input(
                "Enter Groq API Key",
                type="password",
                placeholder="gsk_...",
                label_visibility="collapsed"
            )
        with k_col2:
            if st.button("🚀 Connect Key", use_container_width=True):
                if main_key_in and main_key_in.strip():
                    st.session_state["custom_groq_key"] = main_key_in.strip()
                    st.rerun()
                else:
                    st.error("Please paste your key first.")
        st.caption("💡 Don't have a key? Get a free API key in 10 seconds from [Groq Console](https://console.groq.com/keys)")
        st.divider()


# ---------------------------------------------------------
# Helper: Handle Chat Completion Flow
# ---------------------------------------------------------
def process_user_query(
    prompt_text: str,
    client: Optional[Groq],
    model_name: str,
    sys_prompt: str,
    temp: float,
    tokens: int,
    top_p_val: float
) -> None:
    if not client:
        st.error("❌ Please provide a valid Groq API Key in the sidebar before sending messages.", icon="🔑")
        return

    # Add user message to history and render
    st.session_state["messages"].append({"role": "user", "content": prompt_text})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt_text)

    # Build conversation payload
    api_messages: List[Dict[str, str]] = []
    if sys_prompt and sys_prompt.strip():
        api_messages.append({"role": "system", "content": sys_prompt.strip()})

    for m in st.session_state["messages"]:
        api_messages.append({"role": m["role"], "content": m["content"]})

    # Render streaming response container
    with st.chat_message("assistant", avatar="⚡"):
        response_placeholder = st.empty()
        full_response = ""
        start_time = time.time()

        try:
            stream = client.chat.completions.create(
                model=model_name,
                messages=api_messages,  # type: ignore
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p_val,
                stream=True
            )

            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        full_response += delta.content
                        response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)
            elapsed_time = time.time() - start_time
            st.session_state["last_response_time"] = elapsed_time
            st.session_state["messages"].append({"role": "assistant", "content": full_response})

        except GroqError as ge:
            if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "user":
                st.session_state["messages"].pop()
            response_placeholder.empty()
            err_msg = str(ge)
            if "401" in err_msg or "invalid_api_key" in err_msg.lower():
                st.error(
                    "🔑 **Invalid Groq API Key:** The key provided was rejected. "
                    "Please check or enter a valid API key in the sidebar.",
                    icon="⚠️"
                )
                st.info("💡 You can create a free API key at [Groq Console](https://console.groq.com/keys).")
            else:
                st.error(f"⚠️ **Groq API Error:** {err_msg}")
        except Exception as e:
            if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "user":
                st.session_state["messages"].pop()
            response_placeholder.empty()
            st.error(f"⚠️ **Unexpected Error:** {str(e)}")


# ---------------------------------------------------------
# Display Existing Chat Messages
# ---------------------------------------------------------
for msg in st.session_state.get("messages", []):
    if msg["role"] == "user":
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="⚡"):
            st.markdown(msg["content"])


# ---------------------------------------------------------
# Empty State: Starter Suggestions
# ---------------------------------------------------------
if len(st.session_state.get("messages", [])) == 0:
    st.markdown("##### 💡 Try one of these quick starters:")

    starter_prompts = [
        {
            "icon": "⚡",
            "title": "Explain Groq LPUs",
            "prompt": "Explain how Groq's Language Processing Unit (LPU) architecture achieves high inference speeds."
        },
        {
            "icon": "🐍",
            "title": "Python Async Code",
            "prompt": "Write a clean Python script demonstrating asynchronous HTTP requests with error handling."
        },
        {
            "icon": "🔬",
            "title": "Quantum Computing",
            "prompt": "Explain the core principles of quantum computing (superposition, qubits) simply."
        },
        {
            "icon": "🚀",
            "title": "Startup Pitch Idea",
            "prompt": "Generate 3 innovative B2B SaaS startup ideas leveraging real-time low-latency AI inference."
        }
    ]

    cols = st.columns(len(starter_prompts))
    for idx, item in enumerate(starter_prompts):
        with cols[idx]:
            if st.button(f"{item['icon']} **{item['title']}**", key=f"starter_{idx}", use_container_width=True):
                process_user_query(
                    prompt_text=item["prompt"],
                    client=groq_client,
                    model_name=selected_model_key,
                    sys_prompt=system_prompt,
                    temp=temperature,
                    tokens=max_tokens,
                    top_p_val=top_p
                )


# ---------------------------------------------------------
# Handle User Input
# ---------------------------------------------------------
user_prompt = st.chat_input("Ask anything... (Shift+Enter for new line)")
if user_prompt:
    process_user_query(
        prompt_text=user_prompt,
        client=groq_client,
        model_name=selected_model_key,
        sys_prompt=system_prompt,
        temp=temperature,
        tokens=max_tokens,
        top_p_val=top_p
    )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("""
<div class="footer-text">
    Built with 🧡 using <a href="https://streamlit.io" target="_blank" style="color: #f97316; text-decoration: none;">Streamlit</a> & <a href="https://groq.com" target="_blank" style="color: #f97316; text-decoration: none;">Groq</a> LPUs
</div>
""", unsafe_allow_html=True)
