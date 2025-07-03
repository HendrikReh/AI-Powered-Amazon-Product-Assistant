"""
Business Intelligence and User Experience Tracing Module.
Provides comprehensive business-level analytics and user journey tracking.
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import weave

from .trace_utils import (
    business_analyzer, get_current_trace_context, 
    UserType, QueryIntent
)


@dataclass
class UserJourney:
    """Tracks user journey and engagement patterns."""
    session_id: str
    user_type: Optional[UserType] = None
    journey_stage: str = "exploration"  # exploration, research, decision, action
    queries_count: int = 0
    session_duration: float = 0.0
    pain_points: List[str] = None
    conversion_indicators: List[str] = None
    satisfaction_scores: List[float] = None
    
    def __post_init__(self):
        if self.pain_points is None:
            self.pain_points = []
        if self.conversion_indicators is None:
            self.conversion_indicators = []
        if self.satisfaction_scores is None:
            self.satisfaction_scores = []


@dataclass
class BusinessMetrics:
    """Business-level metrics for user interactions."""
    user_satisfaction_prediction: float
    conversion_potential: float
    recommendation_effectiveness: float
    feature_usage_score: float
    query_success_rate: float
    response_quality_score: float


class UserJourneyStage(Enum):
    """Stages in the user journey."""
    EXPLORATION = "exploration"  # Initial browsing, general queries
    RESEARCH = "research"  # Detailed product investigation
    COMPARISON = "comparison"  # Comparing options
    DECISION = "decision"  # Ready to make choice
    ACTION = "action"  # Taking action (purchase-intent)
    SUPPORT = "support"  # Post-purchase or troubleshooting


class BusinessIntelligenceTracker:
    """Tracks business intelligence metrics and user experience."""
    
    def __init__(self):
        self.user_journeys: Dict[str, UserJourney] = {}
        self.session_metrics: Dict[str, Dict] = {}
        self.feature_usage = {
            "rag_usage": 0,
            "filter_usage": 0,
            "comparison_queries": 0,
            "recommendation_requests": 0,
            "review_inquiries": 0
        }
    
    @weave.op()
    def track_user_interaction(
        self, 
        query: str, 
        response: str, 
        context: Dict[str, Any],
        session_id: str,
        conversation_turn: int
    ) -> Dict[str, Any]:
        """Track comprehensive user interaction with business intelligence."""
        
        # Initialize or update user journey
        if session_id not in self.user_journeys:
            self.user_journeys[session_id] = UserJourney(
                session_id=session_id,
                queries_count=0,
                session_duration=0.0
            )
        
        journey = self.user_journeys[session_id]
        journey.queries_count += 1
        
        # Analyze query intent and user behavior
        query_analysis = self._analyze_query_business_context(query)
        response_analysis = self._analyze_response_quality(response, query, context)
        engagement_metrics = self._calculate_engagement_metrics(
            query, response, context, journey, conversation_turn
        )
        
        # Update user journey
        self._update_user_journey(journey, query_analysis, response_analysis)
        
        # Calculate business metrics
        business_metrics = self._calculate_business_metrics(
            query_analysis, response_analysis, engagement_metrics, journey
        )
        
        # Track feature usage
        self._track_feature_usage(query, context)
        
        # Store session metrics
        self.session_metrics[session_id] = {
            "last_interaction": time.time(),
            "total_queries": journey.queries_count,
            "user_type": journey.user_type.value if journey.user_type else "unknown",
            "journey_stage": journey.journey_stage,
            "avg_satisfaction": sum(journey.satisfaction_scores) / len(journey.satisfaction_scores) if journey.satisfaction_scores else 0.0
        }
        
        return {
            "query_analysis": query_analysis,
            "response_analysis": response_analysis,
            "engagement_metrics": engagement_metrics,
            "business_metrics": asdict(business_metrics),
            "user_journey": asdict(journey),
            "feature_usage": self.feature_usage.copy()
        }
    
    @weave.op()
    def _analyze_query_business_context(self, query: str) -> Dict[str, Any]:
        """Analyze query from business perspective."""
        if not business_analyzer:
            return {"error": "Business analyzer not available"}
        
        intent = business_analyzer.classify_intent(query)
        complexity = business_analyzer.calculate_complexity(query)
        specificity = business_analyzer.measure_specificity(query)
        product_focus = business_analyzer.extract_product_focus(query)
        
        # Business-specific analysis
        purchase_intent_indicators = [
            "buy", "purchase", "order", "price", "cost", "cheap", "budget",
            "best", "recommend", "should i", "worth it"
        ]
        
        research_indicators = [
            "review", "opinion", "experience", "feedback", "pros", "cons",
            "comparison", "vs", "versus", "compare", "difference"
        ]
        
        support_indicators = [
            "problem", "issue", "help", "trouble", "broken", "fix",
            "setup", "install", "configure"
        ]
        
        query_lower = query.lower()
        purchase_intent_score = sum(1 for indicator in purchase_intent_indicators if indicator in query_lower) / len(purchase_intent_indicators)
        research_intent_score = sum(1 for indicator in research_indicators if indicator in query_lower) / len(research_indicators)
        support_intent_score = sum(1 for indicator in support_indicators if indicator in query_lower) / len(support_indicators)
        
        return {
            "intent_category": intent.value if intent else "unknown",
            "complexity_score": complexity,
            "specificity_level": specificity,
            "product_focus": product_focus,
            "business_intent_scores": {
                "purchase_intent": purchase_intent_score,
                "research_intent": research_intent_score,
                "support_intent": support_intent_score
            },
            "query_characteristics": {
                "word_count": len(query.split()),
                "question_type": "question" if "?" in query else "statement",
                "urgency_indicators": len([word for word in ["urgent", "asap", "quickly", "now", "immediate"] if word in query_lower])
            }
        }
    
    @weave.op()
    def _analyze_response_quality(self, response: str, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze response quality from business perspective."""
        if not business_analyzer:
            return {"error": "Business analyzer not available"}
        
        length_category = business_analyzer.categorize_response_length(response)
        specificity_match = business_analyzer.measure_response_specificity(response, query)
        product_mentions = business_analyzer.count_product_mentions(response)
        actionable_content = business_analyzer.detect_actionable_content(response)
        
        # Additional business quality metrics
        response_lower = response.lower()
        
        # Value proposition indicators
        value_indicators = ["save", "deal", "discount", "value", "worth", "benefit"]
        value_score = sum(1 for indicator in value_indicators if indicator in response_lower) / len(value_indicators)
        
        # Trust building indicators
        trust_indicators = ["review", "rating", "customer", "feedback", "verified", "tested"]
        trust_score = sum(1 for indicator in trust_indicators if indicator in response_lower) / len(trust_indicators)
        
        # Completeness indicators
        completeness_factors = {
            "provides_specs": any(word in response_lower for word in ["specification", "feature", "dimension", "weight"]),
            "mentions_price": any(word in response_lower for word in ["price", "cost", "$", "dollar", "expensive", "cheap"]),
            "includes_comparison": any(word in response_lower for word in ["compare", "versus", "alternative", "similar"]),
            "addresses_concerns": any(word in response_lower for word in ["however", "but", "although", "consider", "note"])
        }
        
        completeness_score = sum(completeness_factors.values()) / len(completeness_factors)
        
        return {
            "length_category": length_category,
            "specificity_match": specificity_match,
            "product_mentions": product_mentions,
            "actionable_advice": actionable_content,
            "business_quality_metrics": {
                "value_proposition_score": value_score,
                "trust_building_score": trust_score,
                "completeness_score": completeness_score,
                "professional_tone": 1.0 if not any(word in response_lower for word in ["um", "uh", "like", "basically"]) else 0.5
            },
            "response_characteristics": {
                "character_count": len(response),
                "sentence_count": len([s for s in response.split('.') if s.strip()]),
                "paragraph_count": len([p for p in response.split('\n\n') if p.strip()]),
                "bullet_points": response.count('•') + response.count('*') + response.count('-')
            }
        }
    
    @weave.op()
    def _calculate_engagement_metrics(
        self, 
        query: str, 
        response: str, 
        context: Dict[str, Any],
        journey: UserJourney,
        conversation_turn: int
    ) -> Dict[str, Any]:
        """Calculate user engagement metrics."""
        
        # Session engagement
        session_duration = time.time() - (journey.session_duration if journey.session_duration > 0 else time.time())
        avg_time_per_query = session_duration / journey.queries_count if journey.queries_count > 0 else 0
        
        # Query progression analysis
        query_progression_score = min(conversation_turn * 0.1, 1.0)  # Progressive engagement
        
        # Predicted follow-up likelihood
        follow_up_likelihood = 0.0
        if business_analyzer:
            follow_up_likelihood = business_analyzer.predict_follow_up(query, response)
        
        # Conversion potential
        conversion_potential = 0.0
        if business_analyzer:
            conversion_potential = business_analyzer.estimate_conversion_potential(query, context)
        
        return {
            "session_duration": session_duration,
            "queries_per_session": journey.queries_count,
            "avg_time_per_query": avg_time_per_query,
            "conversation_depth": conversation_turn,
            "query_progression_score": query_progression_score,
            "follow_up_likelihood": follow_up_likelihood,
            "conversion_potential": conversion_potential,
            "engagement_indicators": {
                "repeat_user": journey.queries_count > 1,
                "deep_session": conversation_turn > 3,
                "exploration_breadth": len(set([query.lower()[:20] for query in [query]])),  # Simplified
                "context_utilization": bool(context.get('num_products', 0) or context.get('num_reviews', 0))
            }
        }
    
    @weave.op()
    def _update_user_journey(
        self, 
        journey: UserJourney, 
        query_analysis: Dict[str, Any],
        response_analysis: Dict[str, Any]
    ) -> None:
        """Update user journey based on interaction patterns."""
        
        # Classify user type based on behavior patterns
        if not journey.user_type:
            business_scores = query_analysis.get("business_intent_scores", {})
            
            if business_scores.get("research_intent", 0) > 0.3:
                journey.user_type = UserType.RESEARCHER
            elif business_scores.get("purchase_intent", 0) > 0.3:
                journey.user_type = UserType.BUYER
            elif business_scores.get("support_intent", 0) > 0.3:
                journey.user_type = UserType.TROUBLESHOOTER
            else:
                journey.user_type = UserType.CASUAL
        
        # Update journey stage
        business_scores = query_analysis.get("business_intent_scores", {})
        
        if business_scores.get("support_intent", 0) > 0.3:
            journey.journey_stage = UserJourneyStage.SUPPORT.value
        elif business_scores.get("purchase_intent", 0) > 0.4:
            journey.journey_stage = UserJourneyStage.ACTION.value
        elif "compare" in query_analysis.get("intent_category", "").lower():
            journey.journey_stage = UserJourneyStage.COMPARISON.value
        elif journey.queries_count > 3:
            journey.journey_stage = UserJourneyStage.DECISION.value
        elif journey.queries_count > 1:
            journey.journey_stage = UserJourneyStage.RESEARCH.value
        else:
            journey.journey_stage = UserJourneyStage.EXPLORATION.value
        
        # Track satisfaction prediction
        if business_analyzer:
            query = query_analysis.get("intent_category", "")  # Simplified for this example
            response = response_analysis.get("length_category", "")  # Simplified
            satisfaction = business_analyzer.predict_satisfaction(query, response)
            journey.satisfaction_scores.append(satisfaction)
        
        # Detect pain points
        if response_analysis.get("specificity_match", 0) < 0.3:
            journey.pain_points.append("low_response_relevance")
        
        if query_analysis.get("complexity_score", 0) > 0.7 and response_analysis.get("length_category") == "short":
            journey.pain_points.append("insufficient_detail_for_complex_query")
    
    @weave.op()
    def _calculate_business_metrics(
        self,
        query_analysis: Dict[str, Any],
        response_analysis: Dict[str, Any], 
        engagement_metrics: Dict[str, Any],
        journey: UserJourney
    ) -> BusinessMetrics:
        """Calculate comprehensive business metrics."""
        
        # User satisfaction prediction
        satisfaction = sum(journey.satisfaction_scores) / len(journey.satisfaction_scores) if journey.satisfaction_scores else 0.5
        
        # Conversion potential
        conversion = engagement_metrics.get("conversion_potential", 0.0)
        
        # Recommendation effectiveness (based on response quality and context usage)
        recommendation_effectiveness = (
            response_analysis.get("business_quality_metrics", {}).get("completeness_score", 0.0) * 0.4 +
            response_analysis.get("specificity_match", 0.0) * 0.3 +
            (1.0 if response_analysis.get("actionable_advice", False) else 0.0) * 0.3
        )
        
        # Feature usage score
        feature_usage_score = min(
            (self.feature_usage["rag_usage"] * 0.3 +
             self.feature_usage["filter_usage"] * 0.2 +
             self.feature_usage["comparison_queries"] * 0.25 +
             self.feature_usage["recommendation_requests"] * 0.25) / 100, 1.0
        )
        
        # Query success rate (simplified based on satisfaction scores)
        query_success_rate = satisfaction
        
        # Response quality score
        response_quality = (
            response_analysis.get("business_quality_metrics", {}).get("completeness_score", 0.0) * 0.3 +
            response_analysis.get("business_quality_metrics", {}).get("trust_building_score", 0.0) * 0.3 +
            response_analysis.get("business_quality_metrics", {}).get("value_proposition_score", 0.0) * 0.2 +
            response_analysis.get("business_quality_metrics", {}).get("professional_tone", 0.0) * 0.2
        )
        
        return BusinessMetrics(
            user_satisfaction_prediction=satisfaction,
            conversion_potential=conversion,
            recommendation_effectiveness=recommendation_effectiveness,
            feature_usage_score=feature_usage_score,
            query_success_rate=query_success_rate,
            response_quality_score=response_quality
        )
    
    @weave.op()
    def _track_feature_usage(self, query: str, context: Dict[str, Any]) -> None:
        """Track usage of specific features."""
        query_lower = query.lower()
        
        # Track RAG usage
        if context.get('num_products', 0) > 0 or context.get('num_reviews', 0) > 0:
            self.feature_usage["rag_usage"] += 1
        
        # Track filter usage (simplified detection)
        if any(word in query_lower for word in ["under", "below", "cheap", "budget", "expensive"]):
            self.feature_usage["filter_usage"] += 1
        
        # Track comparison queries
        if any(word in query_lower for word in ["compare", "vs", "versus", "difference"]):
            self.feature_usage["comparison_queries"] += 1
        
        # Track recommendation requests
        if any(word in query_lower for word in ["recommend", "suggest", "best", "should i"]):
            self.feature_usage["recommendation_requests"] += 1
        
        # Track review inquiries
        if any(word in query_lower for word in ["review", "opinion", "feedback", "experience"]):
            self.feature_usage["review_inquiries"] += 1
    
    @weave.op()
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session summary for business analysis."""
        if session_id not in self.user_journeys:
            return {"error": "Session not found"}
        
        journey = self.user_journeys[session_id]
        session_metrics = self.session_metrics.get(session_id, {})
        
        return {
            "session_overview": {
                "session_id": session_id,
                "user_type": journey.user_type.value if journey.user_type else "unknown",
                "journey_stage": journey.journey_stage,
                "total_queries": journey.queries_count,
                "session_duration": journey.session_duration,
                "avg_satisfaction": sum(journey.satisfaction_scores) / len(journey.satisfaction_scores) if journey.satisfaction_scores else 0.0
            },
            "pain_points": journey.pain_points,
            "conversion_indicators": journey.conversion_indicators,
            "business_insights": {
                "user_engagement_level": "high" if journey.queries_count > 5 else "medium" if journey.queries_count > 2 else "low",
                "conversion_likelihood": "high" if any("action" in stage for stage in [journey.journey_stage]) else "medium",
                "support_needs": len([p for p in journey.pain_points if "insufficient" in p or "low" in p])
            },
            "recommendations": self._generate_business_recommendations(journey)
        }
    
    @weave.op()
    def _generate_business_recommendations(self, journey: UserJourney) -> List[str]:
        """Generate business recommendations based on user journey."""
        recommendations = []
        
        if journey.user_type == UserType.RESEARCHER:
            recommendations.append("Provide more detailed technical specifications and comparisons")
        elif journey.user_type == UserType.BUYER:
            recommendations.append("Emphasize value propositions and price comparisons")
        elif journey.user_type == UserType.TROUBLESHOOTER:
            recommendations.append("Offer proactive support resources and troubleshooting guides")
        
        if "low_response_relevance" in journey.pain_points:
            recommendations.append("Improve RAG context retrieval and response relevance")
        
        if journey.queries_count > 5 and journey.journey_stage == UserJourneyStage.EXPLORATION.value:
            recommendations.append("Guide user toward more specific product recommendations")
        
        if len(journey.satisfaction_scores) > 0 and sum(journey.satisfaction_scores) / len(journey.satisfaction_scores) < 0.6:
            recommendations.append("Review response quality and provide more actionable insights")
        
        return recommendations


# Global business intelligence tracker
business_tracker = BusinessIntelligenceTracker()


@weave.op()
def track_business_interaction(
    query: str, 
    response: str, 
    context: Dict[str, Any],
    session_id: str = None,
    conversation_turn: int = 0
) -> Dict[str, Any]:
    """Track business-level user interaction with comprehensive analytics."""
    return business_tracker.track_user_interaction(
        query, response, context, session_id or "default", conversation_turn
    )


@weave.op()
def get_business_session_summary(session_id: str) -> Dict[str, Any]:
    """Get business intelligence summary for a session."""
    return business_tracker.get_session_summary(session_id)