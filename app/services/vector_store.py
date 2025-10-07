import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple
import logging
from app.config import settings
from openai import OpenAI

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector storage and retrieval using FAISS."""
    
    def __init__(self):
        """Initialize the vector store."""
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.dimension = 1536  # OpenAI embedding dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadatas = []
        self.doc_ids = []
        
        # Try to load existing index
        self._load_index()
        
        logger.info(f"Vector store initialized with {self.index.ntotal} documents")
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from OpenAI."""
        try:
            response = self.client.embeddings.create(
                model=settings.embedding_model,
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings, dtype=np.float32)
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
                doc_ids = [f"doc_{self.index.ntotal + i}" for i in range(len(texts))]
            
            # Get embeddings
            embeddings = self._get_embeddings(texts)
            
            # Add to FAISS index
            self.index.add(embeddings)
            
            # Store documents and metadata
            self.documents.extend(texts)
            self.metadatas.extend(metadatas)
            self.doc_ids.extend(doc_ids)
            
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
        
        if self.index.ntotal == 0:
            logger.warning("No documents in index")
            return [], [], []
        
        try:
            # Get query embedding
            query_embedding = self._get_embeddings([query])
            
            # Search
            distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
            
            # Get results
            documents = [self.documents[i] for i in indices[0]]
            metadatas = [self.metadatas[i] for i in indices[0]]
            distances_list = distances[0].tolist()
            
            logger.info(f"Search returned {len(documents)} results")
            return documents, metadatas, distances_list
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise
    
    def delete_document(self, doc_id: str) -> None:
        """Delete a document from the vector store."""
        try:
            if doc_id in self.doc_ids:
                idx = self.doc_ids.index(doc_id)
                # FAISS doesn't support deletion easily, so we rebuild
                self.documents.pop(idx)
                self.metadatas.pop(idx)
                self.doc_ids.pop(idx)
                self._rebuild_index()
                logger.info(f"Deleted document: {doc_id}")
            else:
                logger.warning(f"Document {doc_id} not found")
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise
    
    def _rebuild_index(self):
        """Rebuild the FAISS index from stored documents."""
        if not self.documents:
            self.index = faiss.IndexFlatL2(self.dimension)
            return
        
        embeddings = self._get_embeddings(self.documents)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        self._save_index()
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        try:
            return {
                "total_documents": self.index.ntotal,
                "collection_name": settings.collection_name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def _save_index(self):
        """Save the FAISS index and metadata to disk."""
        os.makedirs(settings.chroma_persist_directory, exist_ok=True)
        
        index_path = os.path.join(settings.chroma_persist_directory, "faiss.index")
        metadata_path = os.path.join(settings.chroma_persist_directory, "metadata.pkl")
        
        faiss.write_index(self.index, index_path)
        
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadatas': self.metadatas,
                'doc_ids': self.doc_ids
            }, f)
        
        logger.info("Index saved to disk")
    
    def _load_index(self):
        """Load the FAISS index and metadata from disk."""
        index_path = os.path.join(settings.chroma_persist_directory, "faiss.index")
        metadata_path = os.path.join(settings.chroma_persist_directory, "metadata.pkl")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                self.index = faiss.read_index(index_path)
                
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.metadatas = data['metadatas']
                    self.doc_ids = data['doc_ids']
                
                logger.info(f"Loaded existing index with {self.index.ntotal} documents")
            except Exception as e:
                logger.error(f"Error loading index: {e}")
                self.index = faiss.IndexFlatL2(self.dimension)
        else:
            logger.info("No existing index found, starting fresh")
    
    def health_check(self) -> bool:
        """Check if the vector store is healthy."""
        try:
            # Just check if index exists
            return self.index is not None
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return False