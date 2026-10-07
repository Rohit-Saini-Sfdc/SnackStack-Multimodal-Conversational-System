# 🍔 SnackStack: Multimodal Multi-Agent Food Delivery Assistant

> **Layered Intelligence for Food** — A voice and text food delivery assistant powered by **LangGraph**, **LangChain**, **ChromaDB RAG**, and **OpenAI (GPT-4o, Whisper, TTS)**.

---

📌 **Assignment Brief & Specifications**: [InterviewKickstart Project Requirements (Google Doc)](https://docs.google.com/document/d/1TQPbLhmBqexVN-mVbzUMqIXJwXeZA2874TOWFQ5uEvc/edit?tab=t.0)  
🐙 **GitHub Repository**: [Rohit-Saini-Sfdc/SnackStack-Multimodal-Conversational-System](https://github.com/Rohit-Saini-Sfdc/SnackStack-Multimodal-Conversational-System)

---

## 📐 Architecture Overview

SnackStack routes natural language queries through an LLM-powered **Orchestrator** to specialist agents (**Menu Agent** and **Order Agent**), executing tools or triggering **Human-in-the-Loop (HITL)** interrupts when information is missing, before synthesizing a unified response.

```
                         ┌───────────────────────────┐
                         │    Voice / Text Input     │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │       Orchestrator        │ ◄── Structured Pydantic Routing
                         └─────────────┬─────────────┘
                                       │
             ┌─────────────────────────┴─────────────────────────┐
             │                                                   │
             ▼ (Parallel Dispatch via Send())                    ▼
┌───────────────────────────┐                       ┌───────────────────────────┐
│        Menu Agent         │                       │        Order Agent        │
│   (ChromaDB RAG Search)   │                       │  (Order DB & HITL Loop)   │
└────────────┬──────────────┘                       └────────────┬──────────────┘
             │                                                   │
             └─────────────────────────┬─────────────────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │        Synthesizer        │ ◄── Response Merging
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │    Voice / Text Output    │
                         └───────────────────────────┘
```

---

## ✨ Key Features

- **🎯 Multi-Agent Orchestration**: Dynamic query routing using `llm.with_structured_output(RouteDecision)` with Pydantic validation.
- **🥗 RAG-Powered Menu Catalog**: Semantic similarity search using **ChromaDB** vector store and `text-embedding-3-small` embeddings across 8 dishes with dietary tags (Veg, Vegan, GF), prices, and ratings.
- **📦 Order Tracking & HITL Interrupts**: Looks up order status by Order ID (`ORD-201`), Tracking ID (`SS201TRK`), or Email (`priya@example.com`). If an identifier is missing, it pauses graph execution using LangGraph's `interrupt()` and seamlessly resumes via `Command(resume=...)`.
- **⚡ Parallel Dispatch**: Executes `Menu Agent` and `Order Agent` concurrently using LangGraph `Send()` syntax when queries span both domains (e.g. *"Show pizza options and track my order ORD-203"*).
- **🎙️ Multimodal Voice I/O**:
  - **Speech-to-Text (STT)**: Voice input via `sounddevice` microphone recording and OpenAI Whisper API (`whisper-1`).
  - **Text-to-Speech (TTS)**: Natural voice responses generated via OpenAI TTS (`tts-1`) and played through speakers.
  - **Interactive Press-to-Talk**: Press `ENTER` to start recording, speak as long as needed, and press `ENTER` again to stop.
- **💾 Session Memory & Checkpointing**: Persists state across multi-turn interactions using `MemorySaver`.

---

## 📂 Project Structure

```text
snackstack/
├── snackstack/
│   ├── __init__.py
│   ├── config.py           # OpenAI client, GPT-4o LLM & Embeddings configuration
│   ├── logger.py           # Centralized logging setup
│   ├── state.py            # Shared StackState TypedDict schema
│   ├── graph.py            # StateGraph construction, edges, and checkpointer
│   ├── main.py             # CLI REPL entry point & HITL interrupt handler
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── prompts.py      # System prompts for all agents
│   │   ├── orchestrator.py # Pydantic structured output router
│   │   ├── menu_agent.py   # RAG menu search agent with tool loop
│   │   ├── order_agent.py  # Order tracking agent with HITL interrupt()
│   │   └── synthesizer.py  # Merges single/parallel agent responses
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── menu.py         # 8-dish menu catalog dataset
│   │   └── orders.py       # 5-order mock database
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── rag.py          # ChromaDB vector store initialization
│   │   ├── menu_tools.py   # search_menu_catalog @tool
│   │   └── order_tools.py  # get_order_status @tool
│   │
│   └── voice/              # Multimodal audio processing
│       ├── __init__.py
│       ├── recorder.py     # Microphone stream capture & Whisper STT
│       └── speaker.py      # OpenAI TTS & speaker playback
│
├── tests/
│   └── test_snackstack.py  # Comprehensive integration test suite
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- **Python 3.11+**
- **OpenAI API Key**

### 2. Virtual Environment Setup
```bash
git clone https://github.com/Rohit-Saini-Sfdc/SnackStack-Multimodal-Conversational-System.git
cd SnackStack-Multimodal-Conversational-System

python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Environment Variables
Create a `.env` file in the project root:
```bash
cp .env.example .env
```
Add your OpenAI API Key inside `.env`:
```env
OPENAI_API_KEY=sk-proj-...
```

---

## 🚀 How to Run

### 1. Standard Interactive Text Mode (Default)
```bash
python -m snackstack.main
```
*Type natural language queries directly into the CLI.*

### 2. Full Voice Mode (Voice Input + Voice Output)
```bash
python -m snackstack.main --voice
```
- Press `ENTER` to start speaking.
- Speak your query.
- Press `ENTER` again when done talking to send audio to Whisper.

### 3. Text Input with Voice Output
```bash
python -m snackstack.main --voice-out
```

### CLI Commands:
- `reset` — Resets conversation state for a new session.
- `quit` / `exit` — Exit the assistant.

---

## 🧪 Automated Testing

Run the test suite covering vector RAG searches, order lookups, HITL interrupts, and parallel routing:

```bash
python -m unittest tests/test_snackstack.py
```

### Test Coverage Matrix:
| Test Scenario | Input Query | Expected Route | Verification |
| :--- | :--- | :--- | :--- |
| **Menu Search** | *"Show me vegan dishes"* | `menu_agent` | Returns Vegan Buddha Bowl, Pasta Primavera & Aglio e Olio |
| **Order Lookup (ID)** | *"Track order ORD-201"* | `order_agent` | Direct lookup returning Butter Chicken status |
| **Order Lookup (Email)**| *"Check order by email priya@example.com"*| `order_agent` | Email lookup returning matching orders |
| **HITL Interrupt** | *"Where is my order?"* | `order_agent` | Triggers `interrupt()`, prompts for ID, resumes on input |
| **Parallel Dispatch** | *"Pizza options and track ORD-203"* | `menu_agent` + `order_agent` | Parallel execution via `Send()`, merged by `Synthesizer` |

---

## 📄 License

This project is open-source under the MIT License.
