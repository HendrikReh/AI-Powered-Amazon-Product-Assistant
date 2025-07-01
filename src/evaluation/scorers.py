"""
Scoring functions for RAG system evaluation using Weave.
"""

import re
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import weave

@dataclass
class ScoreResult:
    """Result of a scoring function."""
    score: float
    max_score: float
    details: Dict[str, Any]
    explanation: str

class RelevanceScorer:
    """Scores how relevant the response is to the query."""
    
    @weave.op()
    def __call__(self, query: str, response: str, expected_topics: List[str]) -> Dict[str, Any]:
        """Score relevance based on topic coverage and query alignment."""
        
        # Convert to lowercase for matching
        response_lower = response.lower()
        query_lower = query.lower()
        
        # Check topic coverage
        topics_covered = 0
        topic_details = {}
        
        for topic in expected_topics:
            topic_lower = topic.lower()
            # Check for exact matches and related terms
            topic_patterns = [
                topic_lower,
                topic_lower.replace('_', ' '),
                topic_lower.replace('_', '-')
            ]
            
            found = any(pattern in response_lower for pattern in topic_patterns)
            topic_details[topic] = found
            if found:
                topics_covered += 1
        
        topic_coverage_score = topics_covered / len(expected_topics) if expected_topics else 1.0
        
        # Check query term coverage
        query_terms = re.findall(r'\b\w+\b', query_lower)
        query_terms = [term for term in query_terms if len(term) > 3]  # Filter short words
        
        query_terms_covered = 0
        for term in query_terms:
            if term in response_lower:
                query_terms_covered += 1
        
        query_coverage_score = query_terms_covered / len(query_terms) if query_terms else 1.0
        
        # Combined relevance score
        relevance_score = (topic_coverage_score * 0.7) + (query_coverage_score * 0.3)
        
        return {
            'relevance_score': relevance_score,
            'topic_coverage': topic_coverage_score,
            'query_coverage': query_coverage_score,
            'topics_covered': topics_covered,
            'total_topics': len(expected_topics),
            'topic_details': topic_details
        }

class AccuracyScorer:
    """Scores factual accuracy by checking key facts and claims."""
    
    @weave.op()
    def __call__(self, response: str, expected_answer: str, expected_products: List[str]) -> Dict[str, Any]:
        """Score accuracy based on factual claims and product mentions."""
        
        response_lower = response.lower()
        expected_lower = expected_answer.lower()
        
        # Check for factual claims alignment
        # Extract key facts from expected answer (simple heuristic)
        expected_facts = self._extract_facts(expected_lower)
        response_facts = self._extract_facts(response_lower)
        
        # Check fact overlap
        fact_matches = 0
        for expected_fact in expected_facts:
            if any(self._fact_similarity(expected_fact, response_fact) > 0.7 
                   for response_fact in response_facts):
                fact_matches += 1
        
        fact_accuracy = fact_matches / len(expected_facts) if expected_facts else 1.0
        
        # Check product mention accuracy
        products_mentioned = 0
        product_details = {}
        
        for product in expected_products:
            product_lower = product.lower().replace('_', ' ')
            mentioned = product_lower in response_lower
            product_details[product] = mentioned
            if mentioned:
                products_mentioned += 1
        
        product_accuracy = products_mentioned / len(expected_products) if expected_products else 1.0
        
        # Combined accuracy score
        accuracy_score = (fact_accuracy * 0.8) + (product_accuracy * 0.2)
        
        return {
            'accuracy_score': accuracy_score,
            'fact_accuracy': fact_accuracy,
            'product_accuracy': product_accuracy,
            'facts_matched': fact_matches,
            'total_facts': len(expected_facts),
            'products_mentioned': products_mentioned,
            'total_products': len(expected_products),
            'product_details': product_details
        }
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract factual statements from text."""
        # Simple fact extraction - sentences with numbers, specific terms
        sentences = re.split(r'[.!?]+', text)
        facts = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
                
            # Look for sentences with numbers, specifications, or specific claims
            if (re.search(r'\d+', sentence) or 
                any(term in sentence for term in ['hours', 'mbps', 'ghz', 'gb', 'inches', 'warranty', 'compatible', 'supports'])):
                facts.append(sentence)
        
        return facts
    
    def _fact_similarity(self, fact1: str, fact2: str) -> float:
        """Calculate similarity between two facts."""
        # Simple word overlap similarity
        words1 = set(re.findall(r'\b\w+\b', fact1.lower()))
        words2 = set(re.findall(r'\b\w+\b', fact2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

class CompletenessScorer:
    """Scores how complete the response is."""
    
    @weave.op()
    def __call__(self, response: str, expected_answer: str, query_type: str) -> Dict[str, Any]:
        """Score completeness based on response length and content depth."""
        
        # Length-based completeness
        response_length = len(response.split())
        expected_length = len(expected_answer.split())
        
        length_ratio = min(response_length / expected_length, 1.0) if expected_length > 0 else 1.0
        
        # Content depth based on query type
        depth_score = self._assess_content_depth(response, query_type)
        
        # Structure completeness (paragraphs, organization)
        structure_score = self._assess_structure(response)
        
        # Combined completeness score
        completeness_score = (length_ratio * 0.4) + (depth_score * 0.4) + (structure_score * 0.2)
        
        return {
            'completeness_score': completeness_score,
            'length_ratio': length_ratio,
            'depth_score': depth_score,
            'structure_score': structure_score,
            'response_length': response_length,
            'expected_length': expected_length
        }
    
    def _assess_content_depth(self, response: str, query_type: str) -> float:
        """Assess content depth based on query type requirements."""
        response_lower = response.lower()
        
        # Define expected elements for each query type
        depth_indicators = {
            'product_info': ['features', 'specifications', 'price', 'performance'],
            'product_reviews': ['customers', 'reviews', 'ratings', 'feedback', 'experience'],
            'product_comparison': ['compare', 'versus', 'difference', 'better', 'advantages'],
            'product_complaints': ['problems', 'issues', 'complaints', 'negative', 'concerns'],
            'product_recommendation': ['recommend', 'suggest', 'alternative', 'best', 'consider'],
            'use_case': ['suitable', 'appropriate', 'effective', 'works', 'designed']
        }
        
        indicators = depth_indicators.get(query_type, depth_indicators['product_info'])
        indicators_found = sum(1 for indicator in indicators if indicator in response_lower)
        
        return indicators_found / len(indicators)
    
    def _assess_structure(self, response: str) -> float:
        """Assess response structure and organization."""
        sentences = len(re.split(r'[.!?]+', response))
        paragraphs = len(response.split('\n\n'))
        
        # Good structure has multiple sentences and some organization
        sentence_score = min(sentences / 5, 1.0)  # Ideal ~5 sentences
        paragraph_score = min(paragraphs / 2, 1.0)  # Ideal ~2 paragraphs
        
        return (sentence_score + paragraph_score) / 2

class FactualityScorer:
    """Scores factual correctness using pattern matching."""
    
    @weave.op()
    def __call__(self, response: str, expected_answer: str) -> Dict[str, Any]:
        """Score factual correctness by checking for contradictions and false claims."""
        
        # Check for obvious contradictions
        contradiction_score = self._check_contradictions(response)
        
        # Check for specific factual claims
        factual_claims_score = self._verify_factual_claims(response, expected_answer)
        
        # Check for appropriate uncertainty expressions
        uncertainty_score = self._check_uncertainty_handling(response)
        
        factuality_score = (contradiction_score * 0.4) + (factual_claims_score * 0.4) + (uncertainty_score * 0.2)
        
        return {
            'factuality_score': factuality_score,
            'contradiction_score': contradiction_score,
            'factual_claims_score': factual_claims_score,
            'uncertainty_score': uncertainty_score
        }
    
    def _check_contradictions(self, response: str) -> float:
        """Check for internal contradictions in the response."""
        # Simple contradiction detection
        contradictory_pairs = [
            ('cheap', 'expensive'),
            ('good', 'bad'),
            ('fast', 'slow'),
            ('reliable', 'unreliable'),
            ('works', 'broken')
        ]
        
        response_lower = response.lower()
        contradictions = 0
        
        for word1, word2 in contradictory_pairs:
            if word1 in response_lower and word2 in response_lower:
                # Check if they're in different contexts (basic check)
                sentences = re.split(r'[.!?]+', response_lower)
                for sentence in sentences:
                    if word1 in sentence and word2 in sentence:
                        contradictions += 1
                        break
        
        # Lower score for more contradictions
        return max(0, 1.0 - (contradictions * 0.3))
    
    def _verify_factual_claims(self, response: str, expected_answer: str) -> float:
        """Verify factual claims against expected answer."""
        # Extract numerical claims
        response_numbers = re.findall(r'\d+(?:\.\d+)?', response)
        expected_numbers = re.findall(r'\d+(?:\.\d+)?', expected_answer)
        
        if not expected_numbers:
            return 1.0  # No specific numbers to verify
        
        # Check if response numbers are reasonable (within order of magnitude)
        reasonable_numbers = 0
        for resp_num in response_numbers:
            for exp_num in expected_numbers:
                try:
                    resp_val = float(resp_num)
                    exp_val = float(exp_num)
                    # Within 2x order of magnitude is reasonable
                    if 0.1 <= resp_val / exp_val <= 10:
                        reasonable_numbers += 1
                        break
                except ValueError:
                    continue
        
        return reasonable_numbers / len(response_numbers) if response_numbers else 1.0
    
    def _check_uncertainty_handling(self, response: str) -> float:
        """Check if response appropriately handles uncertainty."""
        uncertainty_indicators = [
            'typically', 'usually', 'generally', 'often', 'may', 'might', 
            'can', 'could', 'approximately', 'around', 'about', 'varies'
        ]
        
        response_lower = response.lower()
        uncertainty_count = sum(1 for indicator in uncertainty_indicators 
                              if indicator in response_lower)
        
        # Appropriate level of uncertainty (not too absolute, not too uncertain)
        return min(uncertainty_count / 3, 1.0)

class ResponseQualityScorer:
    """Overall response quality scorer."""
    
    @weave.op()
    def __call__(self, response: str, query: str) -> Dict[str, Any]:
        """Score overall response quality including clarity and helpfulness."""
        
        # Clarity score
        clarity_score = self._assess_clarity(response) 
        
        # Helpfulness score
        helpfulness_score = self._assess_helpfulness(response, query)
        
        # Coherence score
        coherence_score = self._assess_coherence(response)
        
        # Combined quality score
        quality_score = (clarity_score * 0.4) + (helpfulness_score * 0.4) + (coherence_score * 0.2)
        
        return {
            'quality_score': quality_score,
            'clarity_score': clarity_score,
            'helpfulness_score': helpfulness_score,
            'coherence_score': coherence_score
        }
    
    def _assess_clarity(self, response: str) -> float:
        """Assess response clarity."""
        # Check sentence length (not too long)
        sentences = re.split(r'[.!?]+', response)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Ideal sentence length is 15-25 words
        length_score = 1.0 - abs(avg_sentence_length - 20) / 20
        length_score = max(0, min(1, length_score))
        
        # Check for clear structure
        has_structure = bool(re.search(r'\b(first|second|third|however|therefore|additionally)\b', response.lower()))
        structure_score = 1.0 if has_structure else 0.7
        
        return (length_score + structure_score) / 2
    
    def _assess_helpfulness(self, response: str, query: str) -> float:
        """Assess how helpful the response is."""
        # Check for actionable information
        actionable_terms = ['recommend', 'suggest', 'consider', 'choose', 'select', 'look for', 'avoid']
        has_actionable = any(term in response.lower() for term in actionable_terms)
        
        # Check for specific details
        has_specifics = bool(re.search(r'\d+|specific|model|brand|type', response.lower()))
        
        # Check for addressing the query directly
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        response_words = set(re.findall(r'\b\w+\b', response.lower()))
        word_overlap = len(query_words.intersection(response_words)) / len(query_words) if query_words else 0
        
        actionable_score = 1.0 if has_actionable else 0.6
        specifics_score = 1.0 if has_specifics else 0.7
        
        return (actionable_score + specifics_score + word_overlap) / 3
    
    def _assess_coherence(self, response: str) -> float:
        """Assess response coherence and flow."""
        sentences = re.split(r'[.!?]+', response)
        if len(sentences) < 2:
            return 1.0
        
        # Check for transition words
        transition_words = ['however', 'therefore', 'additionally', 'furthermore', 'while', 'although']
        has_transitions = any(word in response.lower() for word in transition_words)
        
        # Check for consistent topic (simple heuristic)
        # Count repeated key terms across sentences
        all_words = re.findall(r'\b\w+\b', response.lower())
        word_freq = {}
        for word in all_words:
            if len(word) > 4:  # Only consider longer words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        repeated_terms = sum(1 for count in word_freq.values() if count > 1)
        consistency_score = min(repeated_terms / 5, 1.0)
        
        transition_score = 1.0 if has_transitions else 0.8
        
        return (consistency_score + transition_score) / 2