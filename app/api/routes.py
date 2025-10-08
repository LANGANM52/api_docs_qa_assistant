from fastapi import APIRouter, HTTPException, status
from app.models import (
    DocumentUpload, QuestionRequest, Answer, HealthCheck, 
    ErrorResponse, Source
)
# Default: Use ChromaDB with OpenAI embeddings
from app.services.vector_store import VectorStore
# Alternative: Use TF-IDF without OpenAI
# from app.services.simple_vector_store import SimpleVectorStore as VectorStore

# Default: Use OpenAI LLM
from app.services.llm_service import LLMService
# Alternative: Use mock service for testing
# from app.services.mock_llm_service import MockLLMService as LLMService

from app.services.embeddings import DocumentProcessor
from app.config import settings
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
vector_store = VectorStore()
llm_service = LLMService()
doc_processor = DocumentProcessor()


@router.post("/documents", status_code=status.HTTP_201_CREATED)
async def upload_document(doc: DocumentUpload):
    """
    Upload and index API documentation.
    
    This endpoint accepts documentation content, chunks it, and stores it
    in the vector database for later retrieval.
    """
    try:
        # Preprocess the text
        processed_text = doc_processor.preprocess_text(doc.content)
        
        # Chunk the document
        chunks = doc_processor.chunk_text(processed_text)
        
        # Create metadata for each chunk
        metadatas = [
            doc_processor.create_metadata(
                chunk_index=i,
                total_chunks=len(chunks),
                doc_id=doc.doc_id,
                additional_metadata=doc.metadata
            )
            for i in range(len(chunks))
        ]
        
        # Generate IDs for chunks
        if doc.doc_id:
            chunk_ids = [f"{doc.doc_id}_chunk_{i}" for i in range(len(chunks))]
        else:
            chunk_ids = None
        
        # Add to vector store
        vector_store.add_documents(chunks, metadatas, chunk_ids)
        
        logger.info(f"Successfully uploaded document with {len(chunks)} chunks")
        
        return {
            "message": "Document uploaded successfully",
            "chunks_created": len(chunks),
            "doc_id": doc.doc_id
        }
        
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document: {str(e)}"
        )


@router.post("/ask", response_model=Answer)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about the API documentation.
    
    This endpoint retrieves relevant documentation chunks and uses an LLM
    to generate an accurate answer based on the context.
    """
    try:
        # Search for relevant documents
        documents, metadatas, distances = vector_store.search(
            query=request.question,
            top_k=settings.top_k_results
        )
        
        if not documents:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant documentation found. Please upload documents first."
            )
        
        # Generate answer using LLM
        answer_text, tokens_used = llm_service.generate_answer(
            question=request.question,
            context_docs=documents,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Prepare sources
        sources = [
            Source(
                content=doc[:200] + "..." if len(doc) > 200 else doc,
                metadata=meta,
                relevance_score=1.0 - distance  # Convert distance to similarity score
            )
            for doc, meta, distance in zip(documents, metadatas, distances)
        ]
        
        # Create response
        response = Answer(
            question=request.question,
            answer=answer_text,
            sources=sources,
            model_used=settings.openai_model,
            tokens_used=tokens_used
        )
        
        logger.info(f"Successfully answered question: {request.question[:50]}...")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to answer question: {str(e)}"
        )


@router.get("/health", response_model=HealthCheck)
async def health_check():
    """
    Check the health status of all services.
    
    Returns the status of the vector store and LLM service.
    """
    try:
        vector_store_healthy = vector_store.health_check()
        llm_healthy = llm_service.health_check()
        
        overall_status = "healthy" if (vector_store_healthy and llm_healthy) else "degraded"
        
        return HealthCheck(
            status=overall_status,
            version=settings.app_version,
            vector_store_status="healthy" if vector_store_healthy else "unhealthy",
            llm_status="healthy" if llm_healthy else "unhealthy"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service health check failed"
        )


@router.get("/stats")
async def get_stats():
    """
    Get statistics about the vector store.
    
    Returns information about the number of documents indexed.
    """
    try:
        stats = vector_store.get_collection_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )