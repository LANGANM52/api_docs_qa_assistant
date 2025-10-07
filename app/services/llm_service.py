from openai import OpenAI
from typing import List, Dict, Tuple
import logging
from app.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Manages interactions with the LLM."""
    
    def __init__(self):
        """Initialize the LLM service."""
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        logger.info(f"LLM service initialized with model: {self.model}")
    
    def generate_answer(
        self,
        question: str,
        context_docs: List[str],
        max_tokens: int = None,
        temperature: float = None
    ) -> Tuple[str, int]:
        """
        Generate an answer using the LLM with provided context.
        
        Args:
            question: The user's question
            context_docs: List of relevant document chunks
            max_tokens: Maximum tokens in response
            temperature: Temperature for response generation
            
        Returns:
            Tuple of (answer, tokens_used)
        """
        if max_tokens is None:
            max_tokens = settings.max_tokens
        if temperature is None:
            temperature = settings.temperature
        
        # Build context from retrieved documents
        context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(context_docs)])
        
        # Create the prompt
        system_prompt = """You are an expert API documentation assistant. Your role is to help developers understand and use APIs effectively.

When answering questions:
1. Be precise and technical when needed
2. Provide code examples when relevant
3. Reference the specific documentation sections you're using
4. If the documentation doesn't contain the answer, clearly state that
5. Be concise but thorough

Always base your answers on the provided documentation context."""

        user_prompt = f"""Based on the following API documentation, please answer the question.

Documentation Context:
{context}

Question: {question}

Please provide a clear, accurate answer based on the documentation provided."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            logger.info(f"Generated answer using {tokens_used} tokens")
            return answer, tokens_used
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if the LLM service is accessible."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return False