# Customer Support AI Analytics System

An AI-powered support ticket analytics and anomaly detection system built with **FastAPI**, **Streamlit**, **LangChain**, **Groq LLM**, and **SQLite**.

---

## 📌 Problem Statement

> **Objective:**
> Given a customer support ticket dataset (`support_tickets.csv`), build an end-to-end AI-powered system that:
> 1. Ingests raw CSV data, cleans/validates it, and makes it queryable via a local database.
> 2. Answers natural language questions accurately using deterministic tools (e.g., *"How many critical tickets are unresolved?"*, *"Which agent has the lowest average customer rating?"*).
> 3. Detects and flags operational and statistical anomalies (e.g., SLA breaches, unresolved high-priority tickets older than 24 hours, and resolution time outliers using IQR).
> 4. Exposes the functionality through both a **REST API (FastAPI)** and a clean **web interface (Streamlit)**.
> 5. Uses an LLM (Groq free tier) for natural language reasoning while delegating all calculations to deterministic functions.

---

## 🔄 Step-by-Step Implementation Workflow

The system was developed modularly across 8 distinct stages:

```text
       1. CSV Parsing
             ↓
  2. Validation + Cleaning
             ↓
   3. Feature Engineering
             ↓
    4. Store / Query Layer (SQLite)
             ↓
5. Analytics + Anomaly Detection
             ↓
   6. LLM (Groq) + LangChain
             ↓
       7. FastAPI Backend
             ↓
     8. Streamlit UI (Frontend)
```

### Stage Summary:

* **Stage 1 — CSV Parsing:** Ingests raw CSV data with strict datetime and numeric type conversions while preserving valid missing values.
* **Stage 2 — Validation + Cleaning:** Validates domain schemas, removes duplicates, handles anomalies, normalizes strings, and enforces business rules.
* **Stage 3 — Feature Engineering:** Computes derived columns deterministically (e.g., `is_unresolved`, `is_critical_unresolved`, `ticket_age_hours`, `unresolved_age_hours`, priority scores).
* **Stage 4 — SQLite Store & Query Layer:** Persists structured records into indexed SQLite tables and exposes parameterized, SQL-injection-safe query repositories.
* **Stage 5 — Analytics & Anomaly Detection:** Calculates KPIs (backlog volume, resolution/response averages, rating distributions) and identifies anomalies using rule-based SLA filters and statistical 1.5× IQR bounds.
* **Stage 6 — LangChain + Groq LLM:** Connects a zero-temperature Groq LLM agent to deterministic tools, grounding all responses in verified database evidence without hallucination.
* **Stage 7 — FastAPI Backend:** Exposes clean REST endpoints (`GET /health`, `POST /ask`, `GET /analytics/summary`, `GET /anomalies`) with Pydantic validation and CORS support.
* **Stage 8 — Streamlit UI:** A clean dashboard communicating exclusively with FastAPI over HTTP. Features instant enter-to-submit question answering, live KPI cards, and an interactive anomaly explorer.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Data Processing:** Pandas, NumPy
* **Storage & Queries:** SQLite3
* **LLM Orchestration:** LangChain, Groq API (`openai/gpt-oss-120b` / `llama-3.3-70b-versatile`)
* **Backend API:** FastAPI, Uvicorn, Pydantic
* **Frontend UI:** Streamlit, Requests
* **Testing:** Pytest (103 automated unit and integration tests)

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
GROQ_TEMPERATURE=0
API_BASE_URL=http://localhost:8000
```

### 3. Start the FastAPI Backend (Terminal 1)
```bash
uvicorn app.api.main:app --reload --port 8000
```
* **Interactive API Docs (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

### 4. Start the Streamlit Frontend (Terminal 2)
```bash
streamlit run app/ui/streamlit_app.py
```
* **Web UI Dashboard:** [http://localhost:8501](http://localhost:8501)

---

## 🧪 Running Tests

Run the comprehensive test suite covering data parsing, cleaning, database queries, anomaly detection, agent tooling, and API endpoints:

```bash
python -m pytest -v
```
*(All 103 tests pass with 100% test coverage across all pipeline layers.)*

---

## 💡 Example Inquiries

You can test these queries directly in the Streamlit UI or via `POST /ask`:

* `How many critical tickets are unresolved?`
* `Which agent has the lowest average customer rating?`
* `How many unresolved tickets are older than 24 hours?`
* `Show high-priority unresolved tickets older than 24 hours.`
* `What is the average resolution time by priority?`
* `Are there any anomalies in resolution times?`

---

## 📂 Project Structure

```text
app/
├── api/                  # FastAPI endpoints, schemas, CORS
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── ui/                   # Streamlit web interface & API client
│   ├── __init__.py
│   ├── api_client.py
│   └── streamlit_app.py
├── llm/                  # LangChain agent, Groq orchestrator & tools
├── analytics/            # Deterministic metrics & aggregations
├── anomalies/            # Rule-based & statistical IQR anomaly detectors
├── database/             # SQLite connection, schema & query repository
└── pipeline/             # CSV parser, validator, cleaner & feature engineer
data/
└── support_tickets.db    # SQLite database (500 tickets)
dataset/
└── support_tickets.csv   # Source dataset
tests/                    # 103 automated tests across all stages
Readme.md
requirements.txt
```
