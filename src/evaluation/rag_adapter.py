"""
RAG adapter to connect existing RAG system to ragas evaluation format.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rag.query_processor import create_rag_processor

logger = logging.getLogger(__name__)

@dataclass
class RagasEvaluationInput:
    """Input format for ragas evaluation."""
    question: str
    contexts: List[str]
    answer: str
    ground_truth: Optional[str] = None

class RAGAdapter:
    """Adapter to connect existing RAG system to ragas evaluation format."""
    
    def __init__(self, llm_client=None):
        """Initialize with RAG processor and LLM client."""
        self.rag_processor = create_rag_processor()
        self.llm_client = llm_client
        
        if not self.rag_processor.vector_db:
            raise ValueError("Failed to initialize vector database")
    
    def process_question(self, question: str) -> RagasEvaluationInput:
        """Process a question through the RAG system and format for ragas evaluation."""
        try:
            # Process query through RAG system
            rag_result = self.rag_processor.process_query(question)
            
            if not rag_result["success"]:
                raise ValueError(f"RAG processing failed: {rag_result.get('error', 'Unknown error')}")
            
            # Extract contexts from RAG result
            contexts = self._extract_contexts(rag_result)
            
            # Generate answer using LLM
            answer = self._generate_answer(rag_result, question)
            
            return RagasEvaluationInput(
                question=question,
                contexts=contexts,
                answer=answer
            )
            
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            raise
    
    def _extract_contexts(self, rag_result: Dict[str, Any]) -> List[str]:
        """Extract context strings from RAG result."""
        contexts = []
        
        # Get context from RAG result
        rag_context = rag_result.get("context")
        
        if rag_context:
            # Handle QueryContext dataclass
            if hasattr(rag_context, 'products'):
                # Extract product contexts
                for product in rag_context.products:
                    if isinstance(product, dict):
                        content = product.get("content", "")
                        if content:
                            contexts.append(content)
                
                # Extract review contexts
                for review in rag_context.reviews:
                    if isinstance(review, dict):
                        content = review.get("content", "")
                        if content:
                            contexts.append(content)
            
            # Handle dict-style context (fallback)
            elif isinstance(rag_context, dict):
                products = rag_context.get("products", [])
                for product in products:
                    if isinstance(product, dict):
                        content = product.get("content", "")
                        if content:
                            contexts.append(content)
                
                reviews = rag_context.get("reviews", [])
                for review in reviews:
                    if isinstance(review, dict):
                        content = review.get("content", "")
                        if content:
                            contexts.append(content)
        
        # If no structured context, try to get from enhanced prompt
        if not contexts:
            enhanced_prompt = rag_result.get("enhanced_prompt", "")
            if enhanced_prompt:
                # Extract context section from enhanced prompt
                context_start = enhanced_prompt.find("Context:")
                context_end = enhanced_prompt.find("Question:")
                
                if context_start != -1 and context_end != -1:
                    context_section = enhanced_prompt[context_start:context_end].replace("Context:", "").strip()
                    if context_section:
                        contexts.append(context_section)
                else:
                    # If no clear sections, use the first part of the prompt as context
                    context_lines = enhanced_prompt.split('\n')[:5]  # First 5 lines
                    context_text = '\n'.join(context_lines)
                    if len(context_text.strip()) > 20:
                        contexts.append(context_text)
        
        return contexts
    
    def _generate_answer(self, rag_result: Dict[str, Any], question: str) -> str:
        """Generate answer using LLM client or return mock answer."""
        if self.llm_client:
            try:
                enhanced_prompt = rag_result.get("enhanced_prompt", "")
                if hasattr(self.llm_client, 'generate'):
                    return self.llm_client.generate(enhanced_prompt)
                else:
                    return str(self.llm_client(enhanced_prompt))
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}")
                return self._generate_mock_answer(rag_result, question)
        else:
            return self._generate_mock_answer(rag_result, question)
    
    def _generate_mock_answer(self, rag_result: Dict[str, Any], question: str) -> str:
        """Generate a mock answer based on RAG context."""
        contexts = self._extract_contexts(rag_result)
        metadata = rag_result.get("metadata", {})
        
        # Extract key information from contexts
        key_info = []
        for context in contexts[:2]:  # Limit to first 2 contexts
            # Extract first sentence or first 100 characters
            sentences = context.split('. ')
            if sentences:
                key_info.append(sentences[0])
        
        # Create mock answer
        if key_info:
            answer = f"Based on the available information, {'. '.join(key_info)}."
        else:
            answer = "Based on the search results, I found relevant information about your query."
        
        # Add context about the search
        num_products = metadata.get('num_products', 0)
        num_reviews = metadata.get('num_reviews', 0)
        
        if num_products > 0 or num_reviews > 0:
            answer += f" This information is based on {num_products} products and {num_reviews} customer reviews."
        
        return answer
    
    def process_batch(self, questions: List[str]) -> List[RagasEvaluationInput]:
        """Process a batch of questions."""
        results = []
        
        for i, question in enumerate(questions):
            try:
                logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
                result = self.process_question(question)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process question {i+1}: {e}")
                # Add a fallback result
                results.append(RagasEvaluationInput(
                    question=question,
                    contexts=["No context available due to processing error."],
                    answer="Unable to generate answer due to processing error."
                ))
        
        return results
    
    def create_ragas_dataset(self, test_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert test examples to ragas dataset format."""
        ragas_dataset = []
        
        for example in test_examples:
            question = example.get('question', example.get('query', ''))
            ground_truth = example.get('ground_truth', example.get('expected_answer', ''))
            
            try:
                # Process through RAG system
                rag_input = self.process_question(question)
                
                # Create ragas format
                ragas_example = {
                    'question': rag_input.question,
                    'contexts': rag_input.contexts,
                    'answer': rag_input.answer,
                    'ground_truth': ground_truth,
                    'reference': ground_truth  # Some metrics require 'reference'
                }
                
                # Add metadata if available
                if 'metadata' in example:
                    ragas_example['metadata'] = example['metadata']
                
                ragas_dataset.append(ragas_example)
                
            except Exception as e:
                logger.error(f"Failed to process example: {e}")
                continue
        
        return ragas_dataset
    
    def validate_contexts(self, contexts: List[str]) -> bool:
        """Validate that contexts are meaningful."""
        if not contexts:
            return False
        
        for context in contexts:
            if not context or len(context.strip()) < 10:
                return False
        
        return True
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        try:
            stats = self.rag_processor.vector_db.get_collection_stats()
            return {
                'vector_db_stats': stats,
                'rag_processor_available': True,
                'llm_client_available': self.llm_client is not None
            }
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {
                'error': str(e),
                'rag_processor_available': False,
                'llm_client_available': self.llm_client is not None
            }

# Mock LLM client for testing
class MockLLMClient:
    """Mock LLM client for testing purposes."""
    
    def generate(self, prompt: str) -> str:
        """Generate a mock response based on prompt content."""
        # Extract key information from prompt
        if "iPhone" in prompt or "lightning" in prompt:
            return ("iPhone charging cables feature Lightning connectors, MFi certification, "
                   "various lengths from 3ft to 10ft, and support fast charging. Quality varies "
                   "by brand with Apple and certified third-party options being most reliable.")
        
        elif "Fire TV" in prompt:
            return ("Fire TV Stick provides 1080p streaming, supports Netflix, Prime Video, "
                   "has Alexa Voice Remote, and compact design. Performance is good for most "
                   "streaming but may have occasional buffering with high-demand content.")
        
        elif "backpack" in prompt:
            return ("Laptop backpack durability varies by brand. Higher-end models feature "
                   "reinforced stitching and water-resistant materials. Common issues include "
                   "strap wear and zipper failures after 6-12 months of use.")
        
        elif "earbuds" in prompt:
            return ("Bluetooth earbuds battery life ranges from 3-8 hours per charge. "
                   "Premium models like AirPods deliver 6-8 hours with 20-30 hours total via case. "
                   "Common complaints include battery degradation after 1-2 years.")
        
        else:
            return ("Based on the available product information and customer reviews, here's what I found. "
                   "The products mentioned generally receive positive ratings with some common concerns "
                   "around durability and performance that vary by brand and price point.")

if __name__ == "__main__":
    # Test the adapter
    logging.basicConfig(level=logging.INFO)
    
    try:
        adapter = RAGAdapter(llm_client=MockLLMClient())
        
        # Test questions
        test_questions = [
            "What are iPhone charger cable features?",
            "How is Fire TV performance?",
            "What do people say about laptop backpacks?"
        ]
        
        for question in test_questions:
            print(f"\nQuestion: {question}")
            result = adapter.process_question(question)
            print(f"Contexts: {len(result.contexts)}")
            print(f"Answer: {result.answer[:100]}...")
        
        # Get system stats
        stats = adapter.get_system_stats()
        print(f"\nSystem stats: {stats}")
        
    except Exception as e:
        print(f"Test failed: {e}")