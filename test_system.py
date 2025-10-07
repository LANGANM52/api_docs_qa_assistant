"""
Quick test script to verify the system is working.
Run this after starting the server to test basic functionality.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000/api/v1"


def test_health():
    """Test health endpoint."""
    print("\n1. Testing Health Check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def upload_sample_doc():
    """Upload sample documentation."""
    print("\n2. Uploading Sample Documentation...")
    
    # Use simpler inline content that works well with TF-IDF
    content = """
    TaskFlow API Documentation
    
    Authentication: The TaskFlow API uses Bearer token authentication. Include your API token in the Authorization header of all requests.
    
    Rate Limits: API requests are rate limited. Free tier allows 100 requests per hour, Pro tier allows 1,000 requests per hour.
    
    Projects Endpoint: GET /projects returns a list of all projects accessible to you. You can filter by status using the status parameter.
    
    POST /projects creates a new project. You need to provide a name and team_id in the request body.
    
    Tasks Endpoint: GET /projects/{project_id}/tasks returns all tasks within a project. You can filter by status, assignee, and priority.
    
    POST /projects/{project_id}/tasks creates a new task. Include title, assignee_id, and priority in the request body.
    
    Error Handling: All errors return appropriate HTTP status codes. 400 for bad requests, 401 for unauthorized, 404 for not found, 429 for rate limit exceeded.
    
    Webhooks: TaskFlow supports webhooks to notify your application of events like task.completed and task.updated.
    """

    payload = {
        "content": content,
        "doc_id": "taskflow_api_docs",
        "metadata": {
            "source": "test_script",
            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    response = requests.post(f"{BASE_URL}/documents", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 201


def ask_questions():
    """Ask sample questions."""
    print("\n3. Asking Questions...")

    questions = [
        "How do I authenticate with the TaskFlow API?",
        "What are the rate limits?",
        "How do I create a new task?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n  Question {i}: {question}")
        
        payload = {
            "question": question,
            "max_tokens": 300,
            "temperature": 0.7
        }
        
        response = requests.post(f"{BASE_URL}/ask", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"  Answer: {data['answer'][:200]}...")
            print(f"  Sources used: {len(data['sources'])}")
            print(f"  Tokens used: {data.get('tokens_used', 'N/A')}")
        else:
            print(f"  Error: {response.status_code}")
            print(f"  {response.text}")
        
        time.sleep(1)  # Be nice to the API


def get_stats():
    """Get system statistics."""
    print("\n4. Getting Statistics...")
    response = requests.get(f"{BASE_URL}/stats")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("API Documentation Q&A Assistant - System Test")
    print("=" * 60)
    
    try:
        # Test health
        if not test_health():
            print("\n❌ Health check failed. Is the server running?")
            return
        
        # Upload docs
        if not upload_sample_doc():
            print("\n⚠️  Document upload failed. Check logs.")
            return
        
        # Ask questions
        ask_questions()
        
        # Get stats
        get_stats()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to server.")
        print("Make sure the server is running: uvicorn app.main:app --reload")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")


if __name__ == "__main__":
    main()