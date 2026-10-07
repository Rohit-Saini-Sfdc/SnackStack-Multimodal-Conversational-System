# 🍔 SnackStack: Multimodal Multi-Agent Food Delivery Assistant

> **Layered Intelligence for Food** — A voice and text food delivery assistant powered by **LangGraph**, **LangChain**, **ChromaDB RAG**, and **OpenAI (GPT-4o, Whisper, TTS)**.

---

## 📋 System Requirements & Specification

SnackStack is a CLI-based assistant for a fictional food delivery platform. The system processes natural language queries (text or voice), routes them to specialist agents via an LLM orchestrator, executes tool-calling loops, supports **Human-in-the-Loop (HITL)** interrupts, and synthesizes unified responses.

### Key Capabilities & Functional Requirements:

1. **Orchestrator (Query Router)**:
   - Uses an LLM with **Pydantic structured output** (`RouteDecision`) to classify user queries.
   - Routes food/menu queries and general greetings to `menu_agent`.
   - Routes order tracking queries to `order_agent`.
   - Dispatches to **both agents in parallel** using LangGraph `Send()` syntax when queries span both domains.
   - Defaults to `menu_agent` when intent is ambiguous.

2. **Menu Agent (RAG-Powered)**:
   - Built on a **ChromaDB** vector store embedded with OpenAI `text-embedding-3-small`.
   - Binds the `search_menu_catalog` tool inside an internal tool-calling loop (capped at 5 iterations).
   - Handles dish recommendations, dietary filtering (Veg, Vegan, GF), pricing inquiries, and warm greetings.

3. **Order Agent (with Human-in-the-Loop)**:
   - Supports order lookup by **Order ID** (`ORD-201`), **Tracking ID** (`SS201TRK`), or **Email** (`priya@example.com`).
   - Uses regex extraction to detect identifiers in user input.
   - **HITL Interrupt**: If no identifier is present in the query, it triggers LangGraph's `interrupt()` to pause execution and request the missing ID.
   - Resumes seamlessly via `Command(resume=user_input)` once provided.

4. **Synthesizer**:
   - Merges parallel outputs from `menu_agent` and `order_agent` into a single, cohesive, friendly response.
   - Formats single-agent outputs directly for clean presentation.

5. **Multimodal Voice I/O**:
   - **Voice Input (STT)**: Records microphone audio via `sounddevice` with press-to-start / press-to-stop interaction, transcribing via OpenAI Whisper (`whisper-1`).
   - **Voice Output (TTS)**: Converts text responses into speech via OpenAI TTS (`tts-1`) and plays audio through speakers.

---

## 📊 Embedded Datasets

### 1. Menu Catalog (8 Dishes)

| ID | Dish | Cuisine | Price (INR) | Rating | Dietary Tags | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DISH-001** | Margherita Pizza | Italian | ₹299 | 4.7 | Veg | Classic thin crust with tomato, mozzarella, basil |
| **DISH-002** | Vegan Pasta Primavera | Italian | ₹349 | 4.5 | Vegan | Penne with seasonal vegetables, olive oil, garlic |
| **DISH-003** | Butter Chicken | Indian | ₹379 | 4.9 | GF | Creamy tomato curry with tender chicken and naan |
| **DISH-004** | Vegan Buddha Bowl | Fusion | ₹319 | 4.6 | Vegan, GF | Quinoa, chickpeas, avocado, greens, tahini |
| **DISH-005** | Classic Cheeseburger | American | ₹259 | 4.4 | None | Beef patty, cheddar, lettuce, tomato, brioche bun |
| **DISH-006** | Paneer Tikka | Indian | ₹199 | 4.8 | Veg, GF | Tandoor-grilled cottage cheese with peppers |
| **DISH-007** | Aglio e Olio | Italian | ₹279 | 4.5 | Vegan | Spaghetti with garlic, chilli, olive oil, parsley |
| **DISH-008** | Mango Lassi | Indian | ₹99 | 4.7 | Veg, GF | Blended yogurt with Alphonso mango, cardamom |

### 2. Order Database (5 Orders)

| Order ID | Item | Customer Name | Customer Email | Status | Price | Tracking ID | Est. Delivery |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ORD-201** | Butter Chicken | Priya Nair | priya@example.com | Out for Delivery | ₹379 | SS201TRK | 20 mins |
| **ORD-202** | Margherita Pizza | Arjun Mehta | arjun@example.com | Placed | ₹299 | SS202TRK | 45 mins |
| **ORD-203** | Classic Cheeseburger | Sneha Roy | sneha@example.com | Preparing | ₹259 | SS203TRK | 25 mins |
| **ORD-204** | Vegan Buddha Bowl | Rahul Das | rahul@example.com | Delivered | ₹319 | SS204TRK | Delivered |
| **ORD-205** | Paneer Tikka | Kavya Sharma | kavya@example.com | Placed | ₹199 | SS205TRK | 40 mins |

---

## 📐 Architecture Overview

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

## 📂 Project Structure

```text
snackstack/
├── snackstack/
│   ├── __init__.py
│   ├── config.py           # OpenAI client, GPT-4o LLM & Embeddings setup
│   ├── logger.py           # Centralized logging module
│   ├── state.py            # Shared StackState TypedDict schema
│   ├── graph.py            # StateGraph builder, conditional edges & checkpointer
│   ├── main.py             # CLI REPL entry point & HITL interrupt loop
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── prompts.py      # System prompts for all nodes
│   │   ├── orchestrator.py # Pydantic structured output router
│   │   ├── menu_agent.py   # RAG menu search agent with tool loop
│   │   ├── order_agent.py  # Order tracking agent with HITL interrupt()
│   │   └── synthesizer.py  # Merges single/parallel agent responses
│   │
│   ├── data/
│   │   ├── __init__.py     # Package exports for MENU_CATALOG & ORDER_DATABASE
│   │   ├── menu.py         # 8-dish menu catalog dataset
│   │   └── orders.py       # 5-order mock database
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── rag.py          # ChromaDB vector store loader
│   │   ├── menu_tools.py   # search_menu_catalog @tool
│   │   └── order_tools.py  # get_order_status @tool
│   │
│   └── voice/              # Multimodal audio processing
│       ├── __init__.py
│       ├── recorder.py     # Microphone stream capture & Whisper STT
│       └── speaker.py      # OpenAI TTS & speaker playback
│
├── tests/
│   └── test_snackstack.py  # Automated unit and integration test suite
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
- Press `ENTER` again when done talking to stop recording and send to Whisper STT.

### 3. Text Input with Voice Output
```bash
python -m snackstack.main --voice-out
```

### In-App CLI Commands:
- `reset` — Resets conversation state to start a fresh thread.
- `quit` / `exit` — Exit the assistant.

---

## 🧪 Testing Matrix & Verification

Run the automated test suite covering vector RAG searches, order lookups, HITL interrupts, and parallel routing:

```bash
python -m unittest tests/test_snackstack.py
```

### Validation Matrix:

| Query Scenario | Expected Route | System Behavior |
| :--- | :--- | :--- |
| *"hi"* / *"hello"* | `menu_agent` | Returns warm greeting and offers menu assistance |
| *"Show me vegan dishes"* | `menu_agent` | Executes `search_menu_catalog`, returns Vegan Buddha Bowl, Pasta Primavera, Aglio e Olio |
| *"Indian food under 300"* | `menu_agent` | Filters dishes, returns Paneer Tikka (₹199) and Mango Lassi (₹99) |
| *"Track order ORD-201"* | `order_agent` | Direct lookup returning Butter Chicken status ("Out for Delivery") |
| *"Where is my order?"* | `order_agent` | Triggers HITL `interrupt()`, prompts user for ID, resumes on input |
| *"Check order by email priya@example.com"* | `order_agent` | Email lookup returning matching ORD-201 details |
| *"Pizza options + track ORD-203"* | Both (`menu_agent` + `order_agent`) | Parallel dispatch via `Send()`, merged by `Synthesizer` node |

---

## 📄 License

This project is open-source under the MIT License.
