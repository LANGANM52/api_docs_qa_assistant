# Quick Start Guide

Get the API Documentation Q&A Assistant running in 5 minutes!

## Prerequisites

- Python 3.9+
- OpenAI API key (get one at https://platform.openai.com/api-keys)
- Git (optional)

## Setup Steps

### 1. Get the Code

```bash
# If cloning from GitHub
git clone <your-repo-url>
cd api-docs-qa-assistant

# Or if you received the files directly
cd api-docs-qa-assistant
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**If ChromaDB fails to install on Windows:**
- You need Microsoft C++ Build Tools
- Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Install "Desktop development with C++"
- Retry: `pip install -r requirements.txt`

### 4. Configure API Key

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-your-actual-key-here
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
DEBUG=True
LOG_LEVEL=INFO
```

**Important:** Replace `sk-your-actual-key-here` with your real OpenAI API key!

### 5. Start the Server

```bash
uvicorn app.main:app --reload
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### 6. Test It!

Open your browser to: **http://127.0.0.1:8000/docs**

You'll see the interactive Swagger UI. Try:

1. **Upload Documentation** - POST /api/v1/documents
   ```json
   {
     "content": "The TaskFlow API uses Bearer token authentication. Include your API token in the Authorization header. Rate limits are 100 requests per hour for free tier.",
     "doc_id": "test-doc"
   }
   ```

2. **Ask a Question** - POST /api/v1/ask
   ```json
   {
     "question": "How do I authenticate with the API?",
     "max_tokens": 300
   }
   ```

3. **Check Health** - GET /api/v1/health

## Testing Without OpenAI Credits

Don't have OpenAI credits? No problem!

1. **Edit `app/api/routes.py`** (lines 7-13)
2. **Comment out** the real services:
   ```python
   # from app.services.vector_store import VectorStore
   # from app.services.llm_service import LLMService
   ```

3. **Uncomment** the mock services:
   ```python
   from app.services.simple_vector_store import SimpleVectorStore as VectorStore
   from app.services.mock_llm_service import MockLLMService as LLMService
   ```

4. **Restart** the server

Now it uses TF-IDF (no API calls) and generates mock responses!

## Running the Test Suite

```bash
# Run all tests
python test_system.py

# Or use pytest
pytest tests/
```

## Common Issues

### "ModuleNotFoundError: No module named 'app'"

Make sure you're in the project root directory and your virtual environment is activated.

### "OpenAI API error: Invalid API key"

Check that your `.env` file has the correct `OPENAI_API_KEY` value.

### "Rate limit exceeded" 

You've used up your OpenAI credits. Either:
- Add credits at https://platform.openai.com/settings/organization/billing
- Use the mock services (see "Testing Without OpenAI Credits" above)

### ChromaDB installation fails

Install Microsoft C++ Build Tools or switch to mock services.

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Review [PROJECT_TALKING_POINTS.md](PROJECT_TALKING_POINTS.md) for interview prep
- Check [FINAL_CHECKLIST.md](FINAL_CHECKLIST.md) before presenting

## Architecture Overview

```
Request → FastAPI → Document Processor → ChromaDB (OpenAI embeddings)
                                       ↓
                                    Search
                                       ↓
                           Relevant chunks + Query
                                       ↓
                                OpenAI GPT-4
                                       ↓
                                 Answer + Sources
```

**Monitoring:** Check `/metrics` for Prometheus metrics

**API Docs:** Check `/docs` for interactive API documentation

---

**Having issues?** Check the full README.md for detailed troubleshooting.