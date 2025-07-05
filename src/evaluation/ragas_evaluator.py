"""
Ragas-based evaluator for RAG system evaluation.
Replaces custom evaluation metrics with industry-standard ragas metrics.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import weave
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy, 
        context_precision,
        context_recall
    )
    # Try to import optional metrics
    try:
        from ragas.metrics import context_utilization, answer_correctness
        ADVANCED_METRICS_AVAILABLE = True
    except ImportError:
        try:
            from ragas.metrics import ContextUtilization, AnswerCorrectness
            context_utilization = ContextUtilization()
            answer_correctness = AnswerCorrectness()
            ADVANCED_METRICS_AVAILABLE = True
        except ImportError:
            ADVANCED_METRICS_AVAILABLE = False
    
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from datasets import Dataset
except ImportError as e:
    logging.error(f"Failed to import ragas dependencies: {e}")
    logging.error("Run: uv sync to install required dependencies")
    raise

from .rag_adapter import RAGAdapter, MockLLMClient

logger = logging.getLogger(__name__)

@dataclass
class RagasEvaluationResult:
    """Result of ragas evaluation."""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    context_utilization: Optional[float] = None
    answer_correctness: Optional[float] = None
    overall_score: float = 0.0
    details: Dict[str, Any] = None

class RagasEvaluator:
    """Main evaluator using ragas metrics."""
    
    def __init__(self, openai_api_key: Optional[str] = None, 
                 llm_client=None, use_weave: bool = True):
        """Initialize ragas evaluator."""
        self.openai_api_key = openai_api_key
        self.use_weave = use_weave
        
        # Initialize RAG adapter
        self.rag_adapter = RAGAdapter(llm_client or MockLLMClient())
        
        # Initialize ragas LLM and embeddings if API key provided
        if openai_api_key:
            self.ragas_llm = LangchainLLMWrapper(ChatOpenAI(
                model="gpt-3.5-turbo",
                api_key=openai_api_key,
                temperature=0.1
            ))
            
            self.ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
                api_key=openai_api_key
            ))
        else:
            logger.warning("No OpenAI API key provided. Some ragas metrics may not work.")
            self.ragas_llm = None
            self.ragas_embeddings = None
        
        # Configure metrics based on available resources
        self.metrics = self._configure_metrics()
    
    def _configure_metrics(self) -> List:
        """Configure ragas metrics based on available resources."""
        available_metrics = []
        
        if self.ragas_llm:
            # Metrics that require LLM
            available_metrics.extend([
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ])
            
            # Optional metrics
            if ADVANCED_METRICS_AVAILABLE:
                try:
                    available_metrics.append(context_utilization)
                    available_metrics.append(answer_correctness)
                except:
                    logger.warning("Some advanced ragas metrics not available")
        else:
            logger.warning("No LLM available - using basic metrics only")
            # Use basic metrics that don't require LLM
            available_metrics = []
        
        logger.info(f"Configured {len(available_metrics)} ragas metrics")
        return available_metrics
    
    @weave.op()
    def evaluate_single(self, question: str, ground_truth: Optional[str] = None) -> RagasEvaluationResult:
        """Evaluate a single question using ragas metrics."""
        try:
            # Process through RAG system
            rag_input = self.rag_adapter.process_question(question)
            
            # Create dataset for ragas
            data = {
                'question': [rag_input.question],
                'contexts': [rag_input.contexts],
                'answer': [rag_input.answer]
            }
            
            if ground_truth:
                data['ground_truth'] = [ground_truth]
                data['reference'] = [ground_truth]  # Some metrics require 'reference' instead of 'ground_truth'
            
            dataset = Dataset.from_dict(data)
            
            # Run ragas evaluation
            if self.metrics:
                result = evaluate(dataset, metrics=self.metrics)
                
                # Extract scores
                scores = result.to_pandas().iloc[0]
                
                return RagasEvaluationResult(
                    faithfulness=scores.get('faithfulness', 0.0),
                    answer_relevancy=scores.get('answer_relevancy', 0.0),
                    context_precision=scores.get('context_precision', 0.0),
                    context_recall=scores.get('context_recall', 0.0),
                    context_utilization=scores.get('context_utilization'),
                    answer_correctness=scores.get('answer_correctness'),
                    overall_score=self._calculate_overall_score(scores),
                    details={
                        'question': question,
                        'answer': rag_input.answer,
                        'contexts': rag_input.contexts,
                        'ground_truth': ground_truth,
                        'raw_scores': scores.to_dict()
                    }
                )
            else:
                # Fallback when no metrics available
                return RagasEvaluationResult(
                    faithfulness=0.5,
                    answer_relevancy=0.5,
                    context_precision=0.5,
                    context_recall=0.5,
                    overall_score=0.5,
                    details={
                        'question': question,
                        'answer': rag_input.answer,
                        'contexts': rag_input.contexts,
                        'ground_truth': ground_truth,
                        'note': 'Fallback scores - no LLM available for evaluation'
                    }
                )
                
        except Exception as e:
            logger.error(f"Evaluation failed for question '{question}': {e}")
            return RagasEvaluationResult(
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_precision=0.0,
                context_recall=0.0,
                overall_score=0.0,
                details={'error': str(e), 'question': question}
            )
    
    def _calculate_overall_score(self, scores: pd.Series) -> float:
        """Calculate overall score from individual metrics."""
        # Weight the metrics
        weights = {
            'faithfulness': 0.3,
            'answer_relevancy': 0.3,
            'context_precision': 0.2,
            'context_recall': 0.2
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in scores and pd.notna(scores[metric]):
                total_score += scores[metric] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    @weave.op()
    def evaluate_batch(self, test_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a batch of test examples."""
        logger.info(f"Starting ragas evaluation on {len(test_examples)} examples")
        
        results = []
        failed_count = 0
        
        for i, example in enumerate(test_examples):
            logger.info(f"Evaluating example {i+1}/{len(test_examples)}")
            
            question = example.get('question', example.get('query', ''))
            ground_truth = example.get('ground_truth', example.get('expected_answer'))
            
            try:
                result = self.evaluate_single(question, ground_truth)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate example {i+1}: {e}")
                failed_count += 1
                results.append(RagasEvaluationResult(
                    faithfulness=0.0,
                    answer_relevancy=0.0,
                    context_precision=0.0,
                    context_recall=0.0,
                    overall_score=0.0,
                    details={'error': str(e), 'question': question}
                ))
        
        # Calculate summary statistics
        summary = self._calculate_summary(results)
        summary['failed_count'] = failed_count
        summary['success_rate'] = (len(results) - failed_count) / len(results) if results else 0.0
        
        # Get metric names safely
        metric_names = []
        if self.metrics:
            for metric in self.metrics:
                if hasattr(metric, '__class__'):
                    metric_names.append(metric.__class__.__name__)
                elif hasattr(metric, '__name__'):
                    metric_names.append(metric.__name__)
                else:
                    metric_names.append(str(metric))
        
        return {
            'results': results,
            'summary': summary,
            'metadata': {
                'total_examples': len(test_examples),
                'metrics_used': metric_names,
                'evaluator': 'ragas'
            }
        }
    
    def _calculate_summary(self, results: List[RagasEvaluationResult]) -> Dict[str, Any]:
        """Calculate summary statistics from results."""
        if not results:
            return {}
        
        # Extract scores
        faithfulness_scores = [r.faithfulness for r in results if r.faithfulness > 0]
        answer_relevancy_scores = [r.answer_relevancy for r in results if r.answer_relevancy > 0]
        context_precision_scores = [r.context_precision for r in results if r.context_precision > 0]
        context_recall_scores = [r.context_recall for r in results if r.context_recall > 0]
        overall_scores = [r.overall_score for r in results if r.overall_score > 0]
        
        summary = {
            'total_examples': len(results),
            'valid_examples': len(overall_scores)
        }
        
        if overall_scores:
            summary.update({
                'average_overall_score': sum(overall_scores) / len(overall_scores),
                'min_overall_score': min(overall_scores),
                'max_overall_score': max(overall_scores),
                'scores_above_0.8': sum(1 for s in overall_scores if s > 0.8),
                'scores_below_0.5': sum(1 for s in overall_scores if s < 0.5)
            })
        
        # Individual metric summaries
        if faithfulness_scores:
            summary['average_faithfulness'] = sum(faithfulness_scores) / len(faithfulness_scores)
        if answer_relevancy_scores:
            summary['average_answer_relevancy'] = sum(answer_relevancy_scores) / len(answer_relevancy_scores)
        if context_precision_scores:
            summary['average_context_precision'] = sum(context_precision_scores) / len(context_precision_scores)
        if context_recall_scores:
            summary['average_context_recall'] = sum(context_recall_scores) / len(context_recall_scores)
        
        return summary
    
    def convert_legacy_dataset(self, legacy_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert legacy evaluation dataset to ragas format."""
        ragas_examples = []
        
        for example in legacy_examples:
            ragas_example = {
                'question': example.get('query', ''),
                'ground_truth': example.get('expected_answer', ''),
                'metadata': {
                    'query_type': example.get('query_type'),
                    'difficulty': example.get('difficulty'),
                    'expected_products': example.get('expected_products', []),
                    'expected_topics': example.get('expected_topics', [])
                }
            }
            ragas_examples.append(ragas_example)
        
        return ragas_examples
    
    def export_results(self, evaluation_result: Dict[str, Any], 
                      output_path: str, format: str = 'json'):
        """Export evaluation results to file."""
        import json
        from pathlib import Path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            # Convert results to serializable format
            serializable_results = {
                'summary': evaluation_result['summary'],
                'metadata': evaluation_result['metadata'],
                'results': [asdict(result) for result in evaluation_result['results']]
            }
            
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
        
        elif format == 'csv':
            # Convert to DataFrame and save as CSV
            df_data = []
            for result in evaluation_result['results']:
                row = asdict(result)
                # Flatten details
                if 'details' in row and isinstance(row['details'], dict):
                    for key, value in row['details'].items():
                        if not isinstance(value, (list, dict)):
                            row[f'details_{key}'] = value
                    del row['details']
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.to_csv(output_path, index=False)
        
        logger.info(f"Results exported to {output_path}")

if __name__ == "__main__":
    # Test the evaluator
    logging.basicConfig(level=logging.INFO)
    
    import os
    
    # Test with mock examples
    test_examples = [
        {
            'question': 'What are iPhone charger cable features?',
            'ground_truth': 'iPhone charging cables feature Lightning connectors, MFi certification, and various lengths.'
        },
        {
            'question': 'How is Fire TV performance?',
            'ground_truth': 'Fire TV provides good streaming performance with some occasional buffering.'
        }
    ]
    
    try:
        # Initialize evaluator (with or without OpenAI API key)
        openai_key = os.getenv('OPENAI_API_KEY')
        evaluator = RagasEvaluator(openai_api_key=openai_key)
        
        # Test single evaluation
        result = evaluator.evaluate_single(test_examples[0]['question'], 
                                         test_examples[0]['ground_truth'])
        print(f"Single evaluation result: {result}")
        
        # Test batch evaluation
        batch_result = evaluator.evaluate_batch(test_examples)
        print(f"Batch evaluation summary: {batch_result['summary']}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()