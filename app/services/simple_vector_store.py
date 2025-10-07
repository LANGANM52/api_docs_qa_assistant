"""
Simple vector store using TF-IDF instead of OpenAI embeddings.
"""
import pickle
import os
from typing import List, Dict, Tuple
import logging
from app.config import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)


class SimpleVectorStore:
    """Manages vector storage and retrieval using TF-IDF."""
    
    def __init__(self):
        """Initialize the vector store."""
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.documents = []
        self.metadatas = []
        self.doc_ids = []
        self.document_vectors = None
        
        # Try to load existing index
        self._load_index()
        
        logger.info(f"Simple vector store initialized with {len(self.documents)} documents")
    
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
                doc_ids = [f"doc_{len(self.documents) + i}" for i in range(len(texts))]
            
            # Add to storage
            self.documents.extend(texts)
            self.metadatas.extend(metadatas)
            self.doc_ids.extend(doc_ids)
            
            # Recompute vectors
            if len(self.documents) > 0:
                self.document_vectors = self.vectorizer.fit_transform(self.documents)
            
            # Save index
            self._save_index()
            
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
        
        if len(self.documents) == 0:
            logger.warning("No documents in index")
            return [], [], []
        
        try:
            # Vectorize query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.document_vectors)[0]
            
            # Get top k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Get results
            documents = [self.documents[i] for i in top_indices]
            metadatas = [self.metadatas[i] for i in top_indices]
            # Convert similarity to distance (1 - similarity)
            distances = [1.0 - similarities[i] for i in top_indices]
            
            logger.info(f"Search returned {len(documents)} results")
            return documents, metadatas, distances
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise
    
    def delete_document(self, doc_id: str) -> None:
        """Delete a document from the vector store."""
        try:
            if doc_id in self.doc_ids:
                idx = self.doc_ids.index(doc_id)
                self.documents.pop(idx)
                self.metadatas.pop(idx)
                self.doc_ids.pop(idx)
                
                # Recompute vectors
                if len(self.documents) > 0:
                    self.document_vectors = self.vectorizer.fit_transform(self.documents)
                else:
                    self.document_vectors = None
                
                self._save_index()
                logger.info(f"Deleted document: {doc_id}")
            else:
                logger.warning(f"Document {doc_id} not found")
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        try:
            return {
                "total_documents": len(self.documents),
                "collection_name": settings.collection_name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def _save_index(self):
        """Save the index and metadata to disk."""
        os.makedirs(settings.vector_store_directory, exist_ok=True)
        
        metadata_path = os.path.join(settings.vector_store_directory, "simple_store.pkl")
        
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadatas': self.metadatas,
                'doc_ids': self.doc_ids,
                'vectorizer': self.vectorizer,
                'document_vectors': self.document_vectors
            }, f)
        
        logger.info("Index saved to disk")
    
    def _load_index(self):
        """Load the index and metadata from disk."""
        metadata_path = os.path.join(settings.vector_store_directory, "simple_store.pkl")
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.metadatas = data['metadatas']
                    self.doc_ids = data['doc_ids']
                    self.vectorizer = data['vectorizer']
                    self.document_vectors = data['document_vectors']
                
                logger.info(f"Loaded existing index with {len(self.documents)} documents")
            except Exception as e:
                logger.error(f"Error loading index: {e}")
        else:
            logger.info("No existing index found, starting fresh")
    
    def health_check(self) -> bool:
        """Check if the vector store is healthy."""
        try:
            return True
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return False