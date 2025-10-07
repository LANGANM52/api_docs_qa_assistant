from typing import List, Dict
import logging
from app.config import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document chunking and preprocessing."""
    
    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: The text to chunk
            chunk_size: Size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
            
        Returns:
            List of text chunks
        """
        if chunk_size is None:
            chunk_size = settings.chunk_size
        if chunk_overlap is None:
            chunk_overlap = settings.chunk_overlap
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    last_punct = text[start:end].rfind(punct)
                    if last_punct != -1:
                        end = start + last_punct + len(punct)
                        break
            
            chunks.append(text[start:end].strip())
            start = end - chunk_overlap
            
            # Prevent infinite loop
            if start <= 0 and len(chunks) > 0:
                break
        
        logger.info(f"Split document into {len(chunks)} chunks")
        return chunks
    
    @staticmethod
    def create_metadata(
        chunk_index: int,
        total_chunks: int,
        doc_id: str = None,
        additional_metadata: Dict = None
    ) -> Dict:
        """
        Create metadata for a document chunk.
        
        Args:
            chunk_index: Index of this chunk
            total_chunks: Total number of chunks
            doc_id: Document identifier
            additional_metadata: Additional metadata to include
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
        }
        
        if doc_id:
            metadata["doc_id"] = doc_id
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        return metadata
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocess text before chunking.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove multiple consecutive newlines
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')
        
        return text.strip()