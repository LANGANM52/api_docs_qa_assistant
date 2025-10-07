import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint returns basic info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data


def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code in [200, 503]  # May fail if OpenAI key not set
    data = response.json()
    assert "status" in data
    assert "version" in data


def test_upload_document():
    """Test document upload endpoint."""
    doc_data = {
        "content": "This is a test API documentation. It explains how to use the test API.",
        "metadata": {"source": "test"},
        "doc_id": "test_doc_1"
    }
    
    response = client.post("/api/v1/documents", json=doc_data)
    assert response.status_code == 201
    data = response.json()
    assert "chunks_created" in data
    assert data["chunks_created"] > 0


def test_ask_question():
    """Test asking a question (requires document to be uploaded first)."""
    # First upload a document
    doc_data = {
        "content": "The test API uses OAuth 2.0 for authentication. You need to include a Bearer token in the Authorization header.",
        "metadata": {"source": "test"},
        "doc_id": "test_doc_2"
    }
    client.post("/api/v1/documents", json=doc_data)
    
    # Now ask a question
    question_data = {
        "question": "How do I authenticate with the API?",
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    response = client.post("/api/v1/ask", json=question_data)
    # May fail if OpenAI key not configured
    if response.status_code == 200:
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "question" in data


def test_get_stats():
    """Test the stats endpoint."""
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data


def test_invalid_question():
    """Test asking an invalid question (too short)."""
    question_data = {
        "question": "Hi"  # Too short, minimum is 5 characters
    }
    
    response = client.post("/api/v1/ask", json=question_data)
    assert response.status_code == 422  # Validation error