# Customer Support AI Analytics Assistant

An end-to-end AI-powered Customer Support Ticket Analytics System built with **FastAPI**, **Streamlit**, **LangChain**, **Groq LLM**, and **SQLite**.

---

## 🏗️ Architecture Overview

```text
                 STREAMLIT UI (Port 8501)
                            │
                            │ HTTP (via APIClient)
                            ▼
                 FASTAPI REST API (Port 8000)
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
    POST /ask      GET /analytics/summary   GET /anomalies
        │                   │                   │
        ▼                   ▼                   ▼
  LangChain Agent    Analytics Metrics    Anomaly Detectors
        │                   │                   │
        ▼                   └─────────┬─────────┘
    Groq LLM                          │
(llama-3.1-8b-instant)                │
        │                             │
        ▼                             │
  Deterministic Tools                 │
        │                             │
        └──────────────┬──────────────┘
                       ▼
              SQLite Store Layer
           (data/support_tickets.db)
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Dependencies

Ensure Python 3.10+ is installed. Install the required dependencies:

```bash
pip install fastapi uvicorn streamlit requests langchain langchain-groq pandas numpy pytest pydantic python-dotenv
```

### 2. Environment Configuration

Create or update a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
GROQ_TEMPERATURE=0
API_BASE_URL=http://localhost:8000
```

---

## 🏃 Running the Application

To run the complete system, start the **FastAPI backend** first, followed by the **Streamlit frontend**.

### Terminal 1: Start FastAPI Backend

```bash
uvicorn app.api.main:app --reload --port 8000
```

* API Docs (Swagger UI): [http://localhost:8000/docs](http://localhost:8000/docs)
* Health Check: [http://localhost:8000/health](http://localhost:8000/health)

---

### Terminal 2: Start Streamlit Frontend

```bash
streamlit run app/ui/streamlit_app.py
```

* Streamlit Dashboard: [http://localhost:8501](http://localhost:8501)

---

## 🧪 Running Automated Tests

Run the full pytest suite (103 unit and integration tests across all pipeline stages, query layer, analytics, anomalies, LLM tools, and API endpoints):

```bash
python -m pytest -v
```

---

## 💡 Example Queries to Try in the UI

You can ask natural-language questions directly in the **Streamlit UI** or test them via `POST /ask`:

1. `How many critical tickets are unresolved?`
2. `Which agent has the lowest average customer rating?`
3. `How many unresolved tickets are older than 24 hours?`
4. `Show high-priority unresolved tickets older than 24 hours.`
5. `What is the average resolution time by priority?`
6. `Are there any anomalies in resolution times?`

---

## 📂 Project Structure

```text
├── app/
│   ├── api/                  # FastAPI REST endpoints & Pydantic schemas
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── schemas.py
│   ├── ui/                   # Streamlit Frontend & HTTP Client
│   │   ├── __init__.py
│   │   ├── api_client.py
│   │   └── streamlit_app.py
│   ├── llm/                  # LangChain agent, Groq orchestrator & tools
│   ├── analytics/            # Deterministic aggregations & metrics
│   ├── anomalies/            # Rule-based & statistical IQR anomaly detectors
│   ├── database/             # SQLite connection, schema & query repository
│   └── pipeline/             # CSV parser, validator, cleaner & feature engineering
├── data/
│   └── support_tickets.db    # Populated SQLite database (500 tickets)
├── dataset/
│   └── support_tickets.csv   # Source dataset
├── tests/                    # 103 automated tests across all stages
├── .env.example
├── Readme.md
└── requirements.txt
```
