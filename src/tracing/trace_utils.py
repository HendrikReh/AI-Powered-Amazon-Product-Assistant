"""
Enhanced tracing utilities for Weave instrumentation.
Provides context propagation, performance monitoring, and business metrics.
"""

import time
import uuid
import re
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import weave


class QueryIntent(Enum):
    """Classification of user query intents."""
    PRODUCT_INFO = "product_info"
    PRODUCT_REVIEW = "product_review"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    COMPLAINT = "complaint"
    USE_CASE = "use_case"
    GENERAL = "general"


class UserType(Enum):
    """Classification of user types based on behavior."""
    RESEARCHER = "researcher"  # Detailed, analytical queries
    BUYER = "buyer"  # Price-focused, comparison queries
    CASUAL = "casual"  # General, exploratory queries
    TROUBLESHOOTER = "troubleshooter"  # Problem-solving queries


@dataclass
class TraceContext:
    """Context object for trace propagation."""
    trace_id: str
    session_id: str
    conversation_turn: int
    user_type: Optional[UserType] = None
    query_intent: Optional[QueryIntent] = None
    start_time: float = 0.0
    
    def __post_init__(self):
        if self.start_time == 0.0:
            self.start_time = time.time()


class TraceContextManager:
    """Manages trace context propagation across operations."""
    
    def __init__(self):
        self._current_context: Optional[TraceContext] = None
    
    def create_context(self, session_id: str = None, conversation_turn: int = 0) -> TraceContext:
        """Create a new trace context."""
        context = TraceContext(
            trace_id=str(uuid.uuid4()),
            session_id=session_id or str(uuid.uuid4()),
            conversation_turn=conversation_turn
        )
        self._current_context = context
        return context
    
    def get_current_context(self) -> Optional[TraceContext]:
        """Get the current trace context."""
        return self._current_context
    
    def update_context(self, **kwargs) -> None:
        """Update the current trace context."""
        if self._current_context:
            for key, value in kwargs.items():
                if hasattr(self._current_context, key):
                    setattr(self._current_context, key, value)


class BusinessMetricsAnalyzer:
    """Analyzes business-level metrics from user interactions."""
    
    def __init__(self):
        self.product_keywords = {
            'phone', 'iphone', 'android', 'smartphone', 'mobile',
            'laptop', 'computer', 'pc', 'macbook', 'tablet', 'ipad',
            'headphones', 'earbuds', 'speakers', 'audio', 'bluetooth',
            'charger', 'cable', 'usb', 'lightning', 'power',
            'router', 'wifi', 'internet', 'network', 'ethernet',
            'camera', 'video', 'streaming', 'tv', 'monitor'
        }
        
        self.intent_patterns = {
            QueryIntent.PRODUCT_INFO: [
                r'what is', r'tell me about', r'features of', r'specs', r'specifications'
            ],
            QueryIntent.PRODUCT_REVIEW: [
                r'reviews?', r'what do people say', r'opinions?', r'feedback', r'experience'
            ],
            QueryIntent.COMPARISON: [
                r'compare', r'vs', r'versus', r'difference', r'better', r'best'
            ],
            QueryIntent.RECOMMENDATION: [
                r'recommend', r'suggest', r'best', r'should i', r'good for'
            ],
            QueryIntent.COMPLAINT: [
                r'problem', r'issue', r'complaint', r'broken', r'doesn\'t work', r'bad'
            ],
            QueryIntent.USE_CASE: [
                r'good for', r'use for', r'suitable', r'work with', r'compatible'
            ]
        }
    
    @weave.op()
    def classify_intent(self, query: str) -> QueryIntent:
        """Classify the intent of a user query."""
        query_lower = query.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        return QueryIntent.GENERAL
    
    @weave.op()
    def calculate_complexity(self, query: str) -> float:
        """Calculate query complexity score (0-1)."""
        factors = {
            'word_count': min(len(query.split()) / 20, 1.0),  # Normalize to 20 words
            'question_words': len(re.findall(r'\b(?:what|how|why|when|where|which|who)\b', query.lower())) * 0.1,
            'technical_terms': len([word for word in query.lower().split() if word in self.product_keywords]) * 0.05,
            'punctuation': len(re.findall(r'[?!]', query)) * 0.05
        }
        
        return min(sum(factors.values()), 1.0)
    
    @weave.op()
    def measure_specificity(self, query: str) -> float:
        """Measure query specificity (0-1)."""
        specificity_indicators = {
            'product_mentions': len([word for word in query.lower().split() if word in self.product_keywords]) * 0.2,
            'numbers': len(re.findall(r'\d+', query)) * 0.1,
            'brand_names': len(re.findall(r'\b(?:apple|samsung|sony|lg|hp|dell|lenovo|asus)\b', query.lower())) * 0.2,
            'model_indicators': len(re.findall(r'model|version|generation|series', query.lower())) * 0.1
        }
        
        return min(sum(specificity_indicators.values()), 1.0)
    
    @weave.op()
    def extract_product_focus(self, query: str) -> List[str]:
        """Extract product categories mentioned in query."""
        query_lower = query.lower()
        mentioned_products = []
        
        for keyword in self.product_keywords:
            if keyword in query_lower:
                mentioned_products.append(keyword)
        
        return mentioned_products
    
    @weave.op()
    def categorize_response_length(self, response: str) -> str:
        """Categorize response length."""
        word_count = len(response.split())
        
        if word_count < 50:
            return "short"
        elif word_count < 150:
            return "medium"
        elif word_count < 300:
            return "long"
        else:
            return "very_long"
    
    @weave.op()
    def measure_response_specificity(self, response: str, query: str) -> float:
        """Measure how specifically the response addresses the query."""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Calculate word overlap
        overlap = len(query_words.intersection(response_words))
        total_query_words = len(query_words)
        
        if total_query_words == 0:
            return 0.0
        
        return overlap / total_query_words
    
    @weave.op()
    def count_product_mentions(self, response: str) -> int:
        """Count product mentions in response."""
        response_lower = response.lower()
        return len([word for word in response_lower.split() if word in self.product_keywords])
    
    @weave.op()
    def detect_actionable_content(self, response: str) -> bool:
        """Detect if response contains actionable advice."""
        actionable_patterns = [
            r'should', r'recommend', r'suggest', r'try', r'consider',
            r'look for', r'check', r'compare', r'avoid', r'choose'
        ]
        
        response_lower = response.lower()
        return any(re.search(pattern, response_lower) for pattern in actionable_patterns)
    
    @weave.op()
    def predict_follow_up(self, query: str, response: str) -> float:
        """Predict likelihood of follow-up question (0-1)."""
        factors = {
            'question_complexity': self.calculate_complexity(query) * 0.3,
            'response_length': 0.5 if self.categorize_response_length(response) == "short" else 0.2,
            'comparison_mentioned': 0.3 if 'compare' in response.lower() else 0.0,
            'multiple_options': 0.4 if len(re.findall(r'option|choice|alternative', response.lower())) > 1 else 0.0
        }
        
        return min(sum(factors.values()), 1.0)
    
    @weave.op()
    def estimate_conversion_potential(self, query: str, context: Dict[str, Any]) -> float:
        """Estimate potential for user to make a purchase (0-1)."""
        intent = self.classify_intent(query)
        
        conversion_weights = {
            QueryIntent.RECOMMENDATION: 0.8,
            QueryIntent.COMPARISON: 0.7,
            QueryIntent.PRODUCT_INFO: 0.5,
            QueryIntent.USE_CASE: 0.6,
            QueryIntent.PRODUCT_REVIEW: 0.4,
            QueryIntent.COMPLAINT: 0.2,
            QueryIntent.GENERAL: 0.1
        }
        
        base_score = conversion_weights.get(intent, 0.1)
        
        # Adjust based on context
        if context.get('num_products', 0) > 0:
            base_score *= 1.2
        if context.get('num_reviews', 0) > 0:
            base_score *= 1.1
        
        return min(base_score, 1.0)
    
    @weave.op()
    def predict_satisfaction(self, query: str, response: str) -> float:
        """Predict user satisfaction with response (0-1)."""
        factors = {
            'specificity_match': self.measure_response_specificity(response, query) * 0.3,
            'response_length_appropriate': 0.3 if self.categorize_response_length(response) in ["medium", "long"] else 0.1,
            'actionable_content': 0.2 if self.detect_actionable_content(response) else 0.0,
            'product_mentions': min(self.count_product_mentions(response) * 0.05, 0.2)
        }
        
        return min(sum(factors.values()), 1.0)
    
    @weave.op()
    def classify_user_type(self, query_history: List[str]) -> UserType:
        """Classify user type based on query history."""
        if not query_history:
            return UserType.CASUAL
        
        # Analyze patterns in query history
        avg_complexity = np.mean([self.calculate_complexity(q) for q in query_history])
        comparison_queries = sum(1 for q in query_history if self.classify_intent(q) == QueryIntent.COMPARISON)
        detailed_queries = sum(1 for q in query_history if len(q.split()) > 10)
        
        if avg_complexity > 0.7:
            return UserType.RESEARCHER
        elif comparison_queries > len(query_history) * 0.3:
            return UserType.BUYER
        elif detailed_queries > len(query_history) * 0.4:
            return UserType.TROUBLESHOOTER
        else:
            return UserType.CASUAL


class VectorPerformanceMonitor:
    """Monitors vector database performance metrics."""
    
    def __init__(self):
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    @weave.op()
    def track_embedding_performance(self, query: str, embedding_time: float, embedding_dims: int) -> Dict[str, Any]:
        """Track embedding generation performance."""
        return {
            "query_length": len(query),
            "embedding_time_ms": round(embedding_time * 1000, 2),
            "embedding_dimensions": embedding_dims,
            "tokens_per_second": len(query.split()) / embedding_time if embedding_time > 0 else 0,
            "characters_per_ms": len(query) / (embedding_time * 1000) if embedding_time > 0 else 0
        }
    
    @weave.op()
    def track_search_performance(self, search_time: float, results: Dict, query: str) -> Dict[str, Any]:
        """Track vector search performance."""
        distances = results.get('distances', [[]])[0] if results.get('distances') else []
        metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
        
        # Calculate diversity metrics
        categories = [m.get('category', 'unknown') for m in metadatas]
        unique_categories = len(set(categories))
        
        return {
            "search_time_ms": round(search_time * 1000, 2),
            "results_count": len(distances),
            "avg_similarity_distance": np.mean(distances) if distances else 0,
            "min_similarity_distance": min(distances) if distances else 0,
            "max_similarity_distance": max(distances) if distances else 0,
            "result_diversity_score": unique_categories / len(categories) if categories else 0,
            "cache_hit": self.check_cache_hit(query),
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        }
    
    def check_cache_hit(self, query: str) -> bool:
        """Check if query is in cache."""
        query_hash = hash(query)
        if query_hash in self.cache:
            self.cache_hits += 1
            return True
        else:
            self.cache_misses += 1
            self.cache[query_hash] = True
            return False
    
    @weave.op()
    def analyze_search_quality(self, query: str, results: Dict, context: Dict) -> Dict[str, Any]:
        """Analyze the quality of search results."""
        metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
        
        # Check relevance indicators
        query_terms = set(query.lower().split())
        relevant_results = 0
        
        for metadata in metadatas:
            title = metadata.get('title', '').lower()
            description = metadata.get('description', '').lower()
            
            if any(term in title or term in description for term in query_terms):
                relevant_results += 1
        
        return {
            "relevance_score": relevant_results / len(metadatas) if metadatas else 0,
            "query_coverage": len([t for t in query_terms if any(t in str(m).lower() for m in metadatas)]) / len(query_terms) if query_terms else 0,
            "result_completeness": min(len(metadatas) / 5, 1.0),  # Assuming 5 is optimal
            "context_enrichment": {
                "products_found": context.get('num_products', 0),
                "reviews_found": context.get('num_reviews', 0),
                "total_context_items": context.get('num_products', 0) + context.get('num_reviews', 0)
            }
        }


# Global instances
trace_manager = TraceContextManager()
business_analyzer = BusinessMetricsAnalyzer()
performance_monitor = VectorPerformanceMonitor()


@weave.op()
def create_enhanced_trace_context(session_id: str = None, conversation_turn: int = 0) -> TraceContext:
    """Create enhanced trace context with global manager."""
    return trace_manager.create_context(session_id, conversation_turn)


@weave.op()
def get_current_trace_context() -> Optional[TraceContext]:
    """Get current trace context."""
    return trace_manager.get_current_context()


@weave.op()
def update_trace_context(**kwargs) -> None:
    """Update current trace context."""
    trace_manager.update_context(**kwargs)