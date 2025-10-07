"""
Mock LLM service for testing without OpenAI API.
"""
from typing import List, Tuple
import logging
import re

logger = logging.getLogger(__name__)


class MockLLMService:
    """Mock LLM service that simulates intelligent responses."""

    def __init__(self):
        """Initialize the mock LLM service."""
        self.model = "mock-gpt-4"
        logger.info(f"Mock LLM service initialized")

    def generate_answer(
        self,
        question: str,
        context_docs: List[str],
        max_tokens: int = None,
        temperature: float = None
    ) -> Tuple[str, int]:
        """
        Generate a mock answer based on context.

        Args:
            question: The user's question
            context_docs: List of relevant document chunks
            max_tokens: Maximum tokens in response
            temperature: Temperature for response generation

        Returns:
            Tuple of (answer, tokens_used)
        """
        if not context_docs:
            answer = "I couldn't find relevant information in the documentation to answer your question. Please ensure documentation has been uploaded or try rephrasing your question."
            tokens_used = len(answer.split())
            return answer, tokens_used

        # Extract keywords from question
        keywords = self._extract_keywords(question)

        # Find most relevant sentences
        relevant_info = self._find_relevant_info(context_docs, keywords)

        # Determine question type and format response accordingly
        question_type = self._classify_question(question)

        if question_type == "how_to":
            answer = self._format_how_to_answer(question, relevant_info)
        elif question_type == "what_is":
            answer = self._format_definition_answer(question, relevant_info)
        elif question_type == "list":
            answer = self._format_list_answer(question, relevant_info)
        else:
            answer = self._format_general_answer(question, relevant_info)

        tokens_used = len(answer.split())
        logger.info(f"Generated mock answer with ~{tokens_used} tokens")
        return answer, tokens_used

    def _extract_keywords(self, question: str) -> List[str]:
        """Extract important keywords from the question."""
        # Remove common question words
        stop_words = {'how', 'what', 'when', 'where', 'why', 'who', 'which', 'do', 'does',
                      'is', 'are', 'can', 'the', 'a', 'an', 'i', 'to', 'with', 'for'}

        # Extract words
        words = re.findall(r'\b\w+\b', question.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords[:5]  # Top 5 keywords

    def _find_relevant_info(self, context_docs: List[str], keywords: List[str]) -> str:
        """Find the most relevant sentences from context based on keywords."""
        all_sentences = []

        for doc in context_docs[:3]:  # Check top 3 documents
            # Split into sentences
            sentences = re.split(r'[.!?]\s+', doc)
            for sentence in sentences:
                if len(sentence.strip()) > 20:  # Ignore very short fragments
                    # Count keyword matches
                    score = sum(1 for kw in keywords if kw.lower() in sentence.lower())
                    if score > 0:
                        all_sentences.append((score, sentence.strip()))

        # Sort by relevance and get top sentences
        all_sentences.sort(reverse=True, key=lambda x: x[0])
        top_sentences = [s[1] for s in all_sentences[:3]]

        if not top_sentences:
            # Fallback to first part of first document
            return context_docs[0][:400]

        return ' '.join(top_sentences)

    def _classify_question(self, question: str) -> str:
        """Classify the type of question being asked."""
        question_lower = question.lower()

        if any(word in question_lower for word in ['how do i', 'how to', 'how can i', 'how does']):
            return "how_to"
        elif any(word in question_lower for word in ['what is', 'what are', 'what does']):
            return "what_is"
        elif any(word in question_lower for word in ['list', 'show me', 'what are the']):
            return "list"
        else:
            return "general"

    def _format_how_to_answer(self, question: str, info: str) -> str:
        """Format a how-to answer."""
        return f"""To accomplish this, according to the documentation: {info} This provides the steps needed to answer: "{question}" Make sure to follow the authentication and rate limiting guidelines mentioned in the API documentation."""

    def _format_definition_answer(self, question: str, info: str) -> str:
        """Format a definition/explanation answer."""
        return f"""Based on the API documentation: {info} This explains the concept you asked about. For implementation details and examples, refer to the complete documentation."""

    def _format_list_answer(self, question: str, info: str) -> str:
        """Format a list-type answer."""
        return f"""According to the documentation, here are the relevant details: {info} These are the key points that address your question: "{question}" Check the full API reference for additional options and parameters."""

    def _format_general_answer(self, question: str, info: str) -> str:
        """Format a general answer."""
        return f"""Based on the API documentation: {info} This information directly addresses your question. For more detailed information or code examples, please consult the complete API documentation. Note: This is a demonstration of the RAG system's retrieval capabilities. In production with a full LLM, responses would be more detailed and contextual."""
    
    def health_check(self) -> bool:
        """Mock health check always returns True."""
        return True