import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Tuple
import logging
from app.config import settings
from openai import OpenAI

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector storage and retrieval using ChromaDB with OpenAI embeddings."""
    
    def __init__(self):
        """Initialize the vector store."""
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=settings.vector_store_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        try:
            self.collection = self.client.get_or_create_collection(
                name=settings.collection_name,
                metadata={"description": "API Documentation embeddings"}
            )
            logger.info(f"Vector store initialized with collection: {settings.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings from OpenAI.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            response = self.openai_client.embeddings.create(
                model=settings.embedding_model,
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise
    
    def add_documents(
        self, 
        texts: List[str], 
        metadatas: List[Dict],
        doc_ids: List[str] = None
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text chunks to add
            metadatas: List of metadata dictionaries for each chunk
            doc_ids: Optional list of document IDs
        """
        try:
            if doc_ids is None:
                doc_ids = [f"doc_{self.collection.count() + i}" for i in range(len(texts))]
            
            # Get embeddings from OpenAI
            embeddings = self._get_embeddings(texts)
            
            # Add to ChromaDB
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=doc_ids
            )
            
            logger.info(f"Added {len(texts)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def search(
        self, 
        query: str, 
        top_k: int = None
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Search for relevant documents.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            Tuple of (documents, metadatas, distances)
        """
        if top_k is None:
            top_k = settings.top_k_results
        
        if self.collection.count() == 0:
            logger.warning("No documents in collection")
            return [], [], []
        
        try:
            # Get query embedding
            query_embedding = self._get_embeddings([query])[0]
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count())
            )
            
            documents = results['documents'][0] if results['documents'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            distances = results['distances'][0] if results['distances'] else []
            
            logger.info(f"Search returned {len(documents)} results")
            return documents, metadatas, distances
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise
    
    def delete_document(self, doc_id: str) -> None:
        """Delete a document from the vector store."""
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted document: {doc_id}")
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": settings.collection_name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> bool:
        """Check if the vector store is healthy."""
        try:
            self.collection.count()
            return True
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return False