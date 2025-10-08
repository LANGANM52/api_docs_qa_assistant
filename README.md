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
- **Semantic Search**: ChromaDB vector database with OpenAI embeddings for accurate retrieval
- **LLM Integration**: OpenAI GPT models for generating contextual answers
- **Production-Ready**: Includes monitoring, logging, and health checks
- **RESTful API**: Clean FastAPI implementation with automatic OpenAPI docs
- **Observability**: Prometheus metrics and structured JSON logging
- **Source Attribution**: All answers include relevant source documents
- **Scalable Architecture**: Modular design supporting easy extensions

## 🛠️ Technology Stack

- **Framework**: FastAPI
- **Vector Database**: ChromaDB with OpenAI embeddings
- **LLM Provider**: OpenAI (GPT-4o-mini, GPT-4, or GPT-3.5-turbo)
- **Monitoring**: Prometheus metrics
- **Logging**: Structured JSON logging
- **Testing**: pytest
- **Language**: Python 3.9+

**Alternative (No API Credits):** The system can also run with TF-IDF (scikit-learn) and mock LLM responses for testing without OpenAI costs.

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
- OpenAI API key (required for full functionality)
- pip or conda for package management

**Note:** This project uses OpenAI's API for embeddings and language generation. You'll need an API key with available credits. If you don't have credits, see the "Testing Without OpenAI Credits" section below.

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

**Note for Windows users:** If ChromaDB installation fails due to missing C++ build tools, see the troubleshooting section below.

### Step 4: Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

Required environment variables:
```bash
OPENAI_API_KEY=sk-your-actual-key-here
OPENAI_MODEL=gpt-4o-mini  # or gpt-4, gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-3-small
```

### Step 5: Run the Application

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Testing Without OpenAI Credits

If you want to test the system without OpenAI API credits, you can use the mock services:

1. **Edit `app/api/routes.py`** (lines 7-13):

```python
# Comment out the real services:
# from app.services.vector_store import VectorStore
# from app.services.llm_service import LLMService

# Uncomment the mock services:
from app.services.simple_vector_store import SimpleVectorStore as VectorStore
from app.services.mock_llm_service import MockLLMService as LLMService
```

2. **Restart the server**

The system will now use TF-IDF for semantic search (no API calls) and generate mock responses. This is perfect for:
- Testing the architecture
- Demonstrating the RAG flow
- Development without API costs

### Troubleshooting

**ChromaDB Installation Issues (Windows):**

If you get an error about missing Microsoft Visual C++:

1. Install Microsoft C++ Build Tools from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. During installation, select "Desktop development with C++"
3. Restart your terminal and try `pip install -r requirements.txt` again

**Alternative:** Use the mock services (see "Testing Without OpenAI Credits" above) which don't require ChromaDB.

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

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_api.py -v
```

## 📊 Monitoring & Observability

### Prometheus Metrics

The system exposes the following metrics at `/metrics`:

- `api_requests_total`: Total API requests by method, endpoint, and status
- `api_request_duration_seconds`: Request duration histogram
- `questions_asked_total`: Total questions asked
- `documents_uploaded_total`: Total documents uploaded

### Structured Logging

All logs are in JSON format for easy parsing:

```json
{
  "asctime": "2025-10-07T10:30:00Z",
  "name": "app.services.llm_service",
  "levelname": "INFO",
  "message": "Generated answer using 450 tokens"
}
```

### Health Checks

The `/api/v1/health` endpoint monitors:
- Vector store connectivity
- LLM service availability
- Overall system status

