# ⚡ Ultra-Fast Groq AI Chatbot with Streamlit

A modern, responsive, and feature-packed AI chatbot interface built using **Streamlit** and **Groq Cloud's ultra-low latency LPU™ inference engine**.

---

## 🌟 Key Features

- **⚡ Blazing Fast Streaming**: Real-time token streaming powered by Groq's high-speed inference.
- **🧠 Multiple SOTA Models**: Switch easily between models:
  - `openai/gpt-oss-120b` (Flagship reasoning & quality)
  - `openai/gpt-oss-20b` (Balanced speed & performance)
  - `qwen/qwen3.6-27b` (Exceptional coding & multilingual skills)
  - `groq/compound` & `groq/compound-mini` (Fast compound models)
- **🎨 Persona & System Prompt Presets**:
  - 🤖 Helpful General Assistant
  - 💻 Expert Software Engineer & Architect
  - ⚡ Ultra Concise & Direct
  - ✍️ Creative Writer
  - 🎓 Patient Tutor
  - ⚙️ Custom Persona
- **🎛️ Advanced Parameter Control**: Fine-tune Temperature, Max Tokens, and Top-p sampling in real-time.
- **📥 Chat Export**: Export conversation history to **JSON** or **Markdown** with a single click.
- **💡 Starter Prompts**: Interactive quick-start prompts for rapid testing.
- **📊 Real-time Stats**: Track message count and last response latency in seconds.

---

## 🚀 Quick Start

### 1. Requirements

Ensure you have Python 3.9+ installed.

Install the required packages:
```bash
python -m pip install -r requirements.txt
```

### 2. Configure API Key

Your Groq API key is already configured in `.env`:
```env
GROQ_API_KEY=gsk_...
```
*(You can also override or input a new key directly in the sidebar inside the app).*

### 3. Run the Streamlit Application

```bash
python -m streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
Chatbot Groq/
│
├── .env                # Stores your GROQ_API_KEY
├── .gitignore          # Ignores sensitive environment files & cache
├── app.py              # Main Streamlit Chatbot Application
├── requirements.txt    # Required Python dependencies
└── README.md           # Documentation & instructions
```
