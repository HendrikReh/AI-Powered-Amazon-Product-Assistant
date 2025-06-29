"""
RAG Query Processor for Amazon Electronics Assistant.
Handles query analysis, document retrieval, and context generation for LLM responses.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import os

# Check if running in Docker environment
is_docker = os.getenv("CHROMA_HOST") is not None

try:
    if is_docker:
        from .vector_db_docker import ElectronicsVectorDBDocker as ElectronicsVectorDB, setup_vector_database_docker as setup_vector_database
    else:
        from .vector_db import ElectronicsVectorDB, setup_vector_database
except ImportError:
    # For direct execution
    if is_docker:
        from vector_db_docker import ElectronicsVectorDBDocker as ElectronicsVectorDB, setup_vector_database_docker as setup_vector_database
    else:
        from vector_db import ElectronicsVectorDB, setup_vector_database

logger = logging.getLogger(__name__)


@dataclass
class QueryContext:
    """Structured context for RAG queries."""
    query: str
    query_type: str
    products: List[Dict[str, Any]]
    reviews: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class RAGQueryProcessor:
    """Processes user queries using RAG with Amazon Electronics data."""
    
    def __init__(self, vector_db: Optional[ElectronicsVectorDB] = None):
        """Initialize the RAG query processor."""
        self.vector_db = vector_db
        if not self.vector_db:
            # Initialize with default database
            # Try different possible paths
            possible_paths = [
                "data/processed/electronics_rag_documents.jsonl",
                "../data/processed/electronics_rag_documents.jsonl",
                "../../data/processed/electronics_rag_documents.jsonl"
            ]
            
            jsonl_path = None
            for path in possible_paths:
                if Path(path).exists():
                    jsonl_path = path
                    break
            
            if jsonl_path:
                self.vector_db = setup_vector_database(jsonl_path)
            else:
                logger.warning(f"JSONL file not found in any of these paths: {possible_paths}")
        
        # Query type patterns
        self.query_patterns = {
            "product_info": [
                r"what.*about\s+(.+)",
                r"tell me about\s+(.+)",
                r"describe\s+(.+)",
                r"information.*about\s+(.+)",
                r"details.*about\s+(.+)"
            ],
            "product_reviews": [
                r"reviews?\s+(?:for|of|about)\s+(.+)",
                r"what.*people.*say.*about\s+(.+)",
                r"feedback.*(?:for|about)\s+(.+)",
                r"opinions?.*(?:on|about)\s+(.+)",
                r"experiences?.*with\s+(.+)"
            ],
            "product_complaints": [
                r"(?:problems?|issues?|complaints?).*(?:with|about)\s+(.+)",
                r"what.*wrong.*with\s+(.+)",
                r"negative.*(?:reviews?|feedback).*(?:for|about)\s+(.+)",
                r"main.*complaints?.*about\s+(.+)",
                r"criticism.*(?:of|about)\s+(.+)"
            ],
            "product_comparison": [
                r"compare\s+(.+?)\s+(?:and|vs|versus)\s+(.+)",
                r"difference.*between\s+(.+?)\s+and\s+(.+)",
                r"(.+?)\s+vs\s+(.+)",
                r"which.*better.*(.+?)\s+or\s+(.+)",
                r"(.+?)\s+versus\s+(.+)"
            ],
            "product_recommendation": [
                r"recommend.*(?:budget|cheap|affordable).*(?:alternative|option).*(?:to|for)\s+(.+)",
                r"(?:budget|cheap|affordable).*(?:alternative|option).*(?:to|for)\s+(.+)",
                r"similar.*(?:to|like)\s+(.+).*(?:but|under|less than).*(?:budget|cheap|affordable|\$[\d,]+)",
                r"suggest.*(?:budget|cheap|affordable).*(.+)",
                r"alternative.*(?:to|for)\s+(.+)"
            ],
            "use_case": [
                r"is\s+(.+?)\s+good\s+for\s+(.+)",
                r"can\s+(.+?)\s+be\s+used\s+for\s+(.+)",
                r"suitable.*(.+?).*for\s+(.+)",
                r"(.+?)\s+for\s+(.+)",
                r"best.*(.+?).*for\s+(.+)"
            ],
            "general_search": [
                r"(.+)"  # Catch-all pattern
            ]
        }
    
    def analyze_query(self, query: str) -> Tuple[str, List[str]]:
        """Analyze query to determine type and extract key terms."""
        query_lower = query.lower().strip()
        
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    return query_type, list(match.groups())
        
        return "general_search", [query_lower]
    
    def extract_product_names(self, query: str) -> List[str]:
        """Extract potential product names from query."""
        # Common product name indicators
        product_indicators = [
            r"iphone\s+\d+",
            r"samsung\s+galaxy",
            r"macbook",
            r"fire\s+tv",
            r"kindle",
            r"echo\s+dot",
            r"airpods",
            r"cat\s*\d+\s+cable",
            r"ethernet\s+cable",
            r"usb\s+cable",
            r"charger",
            r"laptop",
            r"tablet",
            r"headphones",
            r"speaker",
            r"router",
            r"backpack"
        ]
        
        products = []
        query_lower = query.lower()
        
        for indicator in product_indicators:
            matches = re.findall(indicator, query_lower)
            products.extend(matches)
        
        return products
    
    def build_context(self, query: str, max_products: int = 5, max_reviews: int = 3) -> QueryContext:
        """Build context for RAG query."""
        if not self.vector_db:
            logger.error("Vector database not initialized")
            return QueryContext(query, "error", [], [], {"error": "Database not available"})
        
        query_type, extracted_terms = self.analyze_query(query)
        logger.info(f"Query type: {query_type}, Terms: {extracted_terms}")
        
        products = []
        reviews = []
        metadata = {
            "query_type": query_type,
            "extracted_terms": extracted_terms,
            "search_strategy": "semantic"
        }
        
        try:
            if query_type == "product_comparison" and len(extracted_terms) >= 2:
                # Search for each product separately
                for term in extracted_terms[:2]:
                    product_results = self.vector_db.search_products(
                        term.strip(), 
                        n_results=2
                    )
                    if "error" not in product_results:
                        products.extend(self._format_search_results(product_results))
                
                # Get reviews for comparison
                review_results = self.vector_db.search_reviews(
                    " ".join(extracted_terms), 
                    n_results=max_reviews
                )
                if "error" not in review_results:
                    reviews.extend(self._format_search_results(review_results))
            
            elif query_type == "product_complaints":
                # Focus on negative sentiment in reviews
                negative_query = f"{extracted_terms[0]} problems issues complaints negative"
                review_results = self.vector_db.search_reviews(
                    negative_query, 
                    n_results=max_reviews * 2
                )
                if "error" not in review_results:
                    reviews.extend(self._format_search_results(review_results))
                
                # Also get product info
                product_results = self.vector_db.search_products(
                    extracted_terms[0], 
                    n_results=max_products // 2
                )
                if "error" not in product_results:
                    products.extend(self._format_search_results(product_results))
            
            elif query_type == "product_recommendation":
                # Search for similar products, potentially with price filtering
                budget_terms = ["budget", "cheap", "affordable", "under"]
                is_budget_query = any(term in query.lower() for term in budget_terms)
                
                if is_budget_query:
                    # Search with price filter
                    product_results = self.vector_db.search_products(
                        extracted_terms[0], 
                        n_results=max_products,
                        price_range=(0, 100)  # Budget range
                    )
                else:
                    product_results = self.vector_db.search_products(
                        extracted_terms[0], 
                        n_results=max_products
                    )
                
                if "error" not in product_results:
                    products.extend(self._format_search_results(product_results))
                
                metadata["search_strategy"] = "recommendation"
                metadata["budget_query"] = is_budget_query
            
            elif query_type == "use_case" and len(extracted_terms) >= 2:
                # Search for products suitable for specific use case
                use_case_query = f"{extracted_terms[0]} {extracted_terms[1]} suitable for"
                product_results = self.vector_db.search_products(
                    use_case_query, 
                    n_results=max_products
                )
                if "error" not in product_results:
                    products.extend(self._format_search_results(product_results))
                
                # Get reviews mentioning the use case
                review_results = self.vector_db.search_reviews(
                    extracted_terms[1], 
                    n_results=max_reviews
                )
                if "error" not in review_results:
                    reviews.extend(self._format_search_results(review_results))
            
            else:
                # General hybrid search
                hybrid_results = self.vector_db.hybrid_search(
                    query, 
                    n_results=max_products + max_reviews,
                    include_products=True,
                    include_reviews=True
                )
                
                if "error" not in hybrid_results:
                    formatted_results = self._format_search_results(hybrid_results)
                    for result in formatted_results:
                        if result["type"] == "product":
                            products.append(result)
                        elif result["type"] == "review_summary":
                            reviews.append(result)
                
                # Limit results
                products = products[:max_products]
                reviews = reviews[:max_reviews]
        
        except Exception as e:
            logger.error(f"Error building context: {e}")
            metadata["error"] = str(e)
        
        return QueryContext(
            query=query,
            query_type=query_type,
            products=products,
            reviews=reviews,
            metadata=metadata
        )
    
    def _format_search_results(self, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format search results into a consistent structure."""
        formatted = []
        
        if "results" not in search_results or "error" in search_results:
            return formatted
        
        results = search_results["results"]
        
        for i, doc_id in enumerate(results["ids"][0]):
            try:
                formatted.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None,
                    "type": results["metadatas"][0][i].get("type", "unknown")
                })
            except (IndexError, KeyError) as e:
                logger.warning(f"Error formatting result {i}: {e}")
                continue
        
        return formatted
    
    def generate_rag_prompt(self, context: QueryContext) -> str:
        """Generate enhanced prompt with RAG context."""
        prompt_parts = [
            f"You are an AI assistant helping users with Amazon Electronics products. "
            f"Answer the user's question using the provided product information and customer reviews.\n",
            f"User Question: {context.query}\n",
            f"Query Type: {context.query_type}\n"
        ]
        
        if context.products:
            prompt_parts.append("=== PRODUCT INFORMATION ===")
            for i, product in enumerate(context.products, 1):
                metadata = product["metadata"]
                prompt_parts.append(f"\nProduct {i}:")
                prompt_parts.append(f"Title: {metadata.get('title', 'N/A')}")
                prompt_parts.append(f"Price: ${metadata.get('price', 'N/A')}")
                prompt_parts.append(f"Rating: {metadata.get('average_rating', 'N/A')}/5 ({metadata.get('rating_number', 'N/A')} ratings)")
                prompt_parts.append(f"Store: {metadata.get('store', 'N/A')}")
                prompt_parts.append(f"Content: {product['content'][:500]}...")
                prompt_parts.append("")
        
        if context.reviews:
            prompt_parts.append("=== CUSTOMER REVIEWS SUMMARY ===")
            for i, review in enumerate(context.reviews, 1):
                metadata = review["metadata"]
                prompt_parts.append(f"\nReview Summary {i}:")
                prompt_parts.append(f"Product: {metadata.get('product_title', 'N/A')}")
                prompt_parts.append(f"Total Reviews: {metadata.get('total_reviews', 'N/A')}")
                prompt_parts.append(f"Positive: {metadata.get('positive_reviews', 'N/A')}, Negative: {metadata.get('negative_reviews', 'N/A')}")
                prompt_parts.append(f"Summary: {review['content'][:400]}...")
                prompt_parts.append("")
        
        prompt_parts.append("=== INSTRUCTIONS ===")
        
        if context.query_type == "product_comparison":
            prompt_parts.append("Compare the products based on features, price, ratings, and customer feedback. Highlight key differences and similarities.")
        elif context.query_type == "product_complaints":
            prompt_parts.append("Focus on the negative aspects and common complaints mentioned in the reviews. Be balanced and mention both positives and negatives.")
        elif context.query_type == "product_recommendation":
            prompt_parts.append("Recommend products based on the user's needs. Consider price, ratings, features, and customer satisfaction.")
        elif context.query_type == "use_case":
            prompt_parts.append("Evaluate whether the products are suitable for the specific use case mentioned. Use product features and customer reviews.")
        else:
            prompt_parts.append("Provide a comprehensive answer using the product information and customer reviews.")
        
        prompt_parts.append("\nPlease provide a helpful, accurate, and well-structured response based on the information above.")
        
        return "\n".join(prompt_parts)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a complete RAG query."""
        try:
            # Build context
            context = self.build_context(query)
            
            # Generate enhanced prompt
            rag_prompt = self.generate_rag_prompt(context)
            
            return {
                "success": True,
                "query": query,
                "context": context,
                "enhanced_prompt": rag_prompt,
                "metadata": {
                    "num_products": len(context.products),
                    "num_reviews": len(context.reviews),
                    "query_type": context.query_type
                }
            }
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }


def create_rag_processor(jsonl_path: str = "data/processed/electronics_rag_documents.jsonl") -> RAGQueryProcessor:
    """Create and initialize RAG query processor."""
    return RAGQueryProcessor()


if __name__ == "__main__":
    # Test the RAG processor
    logging.basicConfig(level=logging.INFO)
    
    processor = create_rag_processor()
    
    # Test queries
    test_queries = [
        "What do people say about iPhone charger cables?",
        "Is the Fire TV good for streaming?",
        "Compare ethernet cables and USB cables",
        "What are the main complaints about laptop backpacks?",
        "Recommend a budget-friendly alternative to expensive tablets"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Testing: {query}")
        print('='*50)
        
        result = processor.process_query(query)
        if result["success"]:
            print(f"Query Type: {result['metadata']['query_type']}")
            print(f"Products Found: {result['metadata']['num_products']}")
            print(f"Reviews Found: {result['metadata']['num_reviews']}")
            print(f"\nEnhanced Prompt Preview:")
            print(result["enhanced_prompt"][:300] + "...")
        else:
            print(f"Error: {result['error']}")