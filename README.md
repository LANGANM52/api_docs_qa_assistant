# API Documentation Q&A Assistant

A production-ready Retrieval-Augmented Generation (RAG) system that helps developers understand and use APIs by answering questions about API documentation using LLMs and vector search.

## 🎯 Project Overview

This system enables intelligent question-answering over API documentation by:
1. Chunking and embedding documentation into a vector database
2. Retrieving relevant context based on user queries
3. Using LLMs to generate accurate, contextual answers
4. Providing source attribution for transparency

## 🚀 Features

- **Document Ingestion**: Upload and process API documentation with intelligent chunking
- **Semantic Search**: TF-IDF vector-based retrieval (easily upgradeable to neural embeddings)
- **LLM Integration**: Flexible architecture supporting OpenAI GPT-4 or mock services
- **Production-Ready**: Includes monitoring, logging, and health checks
- **RESTful API**: Clean FastAPI implementation with automatic OpenAPI docs
- **Observability**: Prometheus metrics and structured JSON logging
- **Source Attribution**: All answers include relevant source documents
- **Scalable Architecture**: Modular design supporting easy extensions

## 🛠️ Technology Stack

- **Framework**: FastAPI
- **Vector Search**: TF-IDF with scikit-learn (easily swappable for neural embeddings)
- **LLM Provider**: OpenAI (GPT-4) or Mock Service for testing
- **Monitoring**: Prometheus metrics
- **Logging**: Structured JSON logging
- **Testing**: pytest
- **Language**: Python 3.9+

## 📁 Project Structure

```
api-docs-qa-assistant/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application and middleware
│   ├── config.py            # Configuration management
│   ├── models.py            # Pydantic models
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py        # API endpoints
│   └── services/
│       ├── __init__.py
│       ├── embeddings.py    # Document processing
│       ├── llm_service.py   # LLM integration
│       └── vector_store.py  # Vector database management
├── tests/
│   ├── __init__.py
│   └── test_api.py          # API tests
├── sample_docs/
│   └── sample_api_doc.txt   # Sample documentation
├── requirements.txt
├── .env.example
└── README.md
```

## 🔧 Installation & Setup

### Prerequisites

- Python 3.9 or higher
- OpenAI API key
- pip or conda for package management

### Step 1: Clone/Create Project

```bash
# Create project directory
mkdir api-docs-qa-assistant
cd api-docs-qa-assistant
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

### Step 5: Run the Application

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

Once running, access:
- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Metrics**: http://localhost:8000/metrics

## 🎮 Usage Examples

### 1. Upload Documentation

```bash
curl -X POST "http://localhost:8000/api/v1/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Your API documentation content here...",
    "doc_id": "my-api-docs",
    "metadata": {"source": "official_docs", "version": "1.0"}
  }'
```

### 2. Ask Questions

```bash
curl -X POST "http://localhost:8000/api/v1/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I authenticate with the API?",
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

### 3. Check Health

```bash
curl "http://localhost:8000/api/v1/health"
```

### 4. Get Statistics

```bash
curl "http://localhost:8000/api/v1/stats"
```

