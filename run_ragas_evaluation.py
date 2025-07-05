"""
Ragas-based RAG system evaluation runner.
Replaces the custom evaluation system with industry-standard ragas metrics.
"""

import asyncio
import argparse
import logging
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

try:
    import weave
    from src.evaluation.ragas_evaluator import RagasEvaluator
    from src.evaluation.ragas_reporter import RagasReporter
    try:
        from src.evaluation.dataset import load_evaluation_dataset
    except ImportError:
        load_evaluation_dataset = None
    WEAVE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some imports failed: {e}")
    WEAVE_AVAILABLE = False
    RagasEvaluator = None
    RagasReporter = None
    load_evaluation_dataset = None

logger = logging.getLogger(__name__)

class RagasEvaluationRunner:
    """Main runner for ragas-based RAG evaluation."""
    
    def __init__(self, openai_api_key: Optional[str] = None, 
                 use_weave: bool = True, wandb_api_key: Optional[str] = None):
        """Initialize evaluation runner."""
        self.openai_api_key = openai_api_key
        self.use_weave = use_weave and WEAVE_AVAILABLE
        self.wandb_api_key = wandb_api_key
        
        # Initialize Weave if available and requested
        if self.use_weave and self.wandb_api_key:
            try:
                import wandb
                wandb.login(key=self.wandb_api_key, anonymous="never", force=True)
                weave.init(project_name="ragas-evaluation")
                logger.info("🔍 Weave tracing enabled for ragas evaluation")
            except Exception as e:
                logger.warning(f"Weave initialization failed: {e}")
                self.use_weave = False
        elif self.use_weave:
            logger.warning("Weave requested but no WANDB_API_KEY provided")
            self.use_weave = False
        
        # Initialize evaluator
        self.evaluator = RagasEvaluator(
            openai_api_key=self.openai_api_key,
            use_weave=self.use_weave
        )
        
        # Initialize reporter
        self.reporter = RagasReporter()
    
    def load_test_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load test dataset from file."""
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        logger.info(f"Loading test dataset from {dataset_path}")
        
        if dataset_path.suffix == '.json':
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Handle different dataset formats
                if isinstance(data, list):
                    examples = data
                elif isinstance(data, dict):
                    if 'testset' in data:
                        # Ragas generated dataset
                        testset = data['testset']
                        if hasattr(testset, 'to_pandas'):
                            df = testset.to_pandas()
                            examples = df.to_dict('records')
                        else:
                            examples = testset
                    elif 'examples' in data:
                        examples = data['examples']
                    else:
                        # Assume the dict itself contains examples
                        examples = [data]
                else:
                    raise ValueError(f"Unsupported dataset format: {type(data)}")
        else:
            # Try legacy format
            try:
                examples = load_evaluation_dataset(str(dataset_path))
                # Convert legacy format to ragas format
                examples = self.evaluator.convert_legacy_dataset(examples)
            except Exception as e:
                raise ValueError(f"Failed to load dataset: {e}")
        
        logger.info(f"Loaded {len(examples)} test examples")
        return examples
    
    @weave.op()
    def run_evaluation(self, dataset_path: str, output_dir: str = "data/evaluation/ragas_results") -> Dict[str, Any]:
        """Run complete ragas evaluation."""
        logger.info("Starting ragas-based RAG evaluation")
        
        # Load test dataset
        test_examples = self.load_test_dataset(dataset_path)
        
        if not test_examples:
            raise ValueError("No test examples loaded")
        
        # Run evaluation
        logger.info(f"Evaluating {len(test_examples)} examples with ragas metrics")
        evaluation_result = self.evaluator.evaluate_batch(test_examples)
        
        # Generate timestamp for results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results_file = output_path / f"ragas_evaluation_results_{timestamp}.json"
        self.evaluator.export_results(evaluation_result, results_file, format='json')
        
        # Save CSV summary
        csv_file = output_path / f"ragas_evaluation_summary_{timestamp}.csv"
        self.evaluator.export_results(evaluation_result, csv_file, format='csv')
        
        # Generate reports
        html_report = output_path / f"ragas_evaluation_report_{timestamp}.html"
        self.reporter.generate_html_report(evaluation_result, html_report)
        
        # Log summary
        self._log_evaluation_summary(evaluation_result)
        
        return {
            'evaluation_result': evaluation_result,
            'output_files': {
                'results_json': str(results_file),
                'summary_csv': str(csv_file),
                'html_report': str(html_report)
            },
            'timestamp': timestamp
        }
    
    @weave.op()
    def run_single_query_evaluation(self, query: str, ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """Run evaluation on a single query."""
        logger.info(f"Evaluating single query: {query}")
        
        result = self.evaluator.evaluate_single(query, ground_truth)
        
        # Log results
        logger.info("Single query evaluation results:")
        logger.info(f"  Faithfulness: {result.faithfulness:.3f}")
        logger.info(f"  Answer Relevancy: {result.answer_relevancy:.3f}")
        logger.info(f"  Context Precision: {result.context_precision:.3f}")
        logger.info(f"  Context Recall: {result.context_recall:.3f}")
        logger.info(f"  Overall Score: {result.overall_score:.3f}")
        
        if result.details and 'answer' in result.details:
            logger.info(f"  Generated Answer: {result.details['answer'][:200]}...")
        
        return {
            'result': result,
            'query': query,
            'ground_truth': ground_truth
        }
    
    def generate_test_dataset(self, rag_documents_path: str, output_path: str, 
                            test_size: int = 50) -> str:
        """Generate test dataset from RAG documents using ragas."""
        logger.info(f"Generating test dataset from {rag_documents_path}")
        
        # Import and run dataset generation
        from scripts.generate_ragas_dataset import RagasDatasetGenerator
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key required for test dataset generation")
        
        generator = RagasDatasetGenerator(self.openai_api_key)
        
        # Load documents
        documents = generator.load_rag_documents(rag_documents_path)
        
        # Generate testset
        testset = asyncio.run(generator.generate_testset(documents, test_size))
        
        # Save testset
        generator.save_testset(testset, output_path)
        
        logger.info(f"Test dataset generated and saved to {output_path}")
        return output_path
    
    def _log_evaluation_summary(self, evaluation_result: Dict[str, Any]):
        """Log evaluation summary."""
        summary = evaluation_result.get('summary', {})
        metadata = evaluation_result.get('metadata', {})
        
        logger.info("=" * 60)
        logger.info("RAGAS EVALUATION SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"Total Examples: {summary.get('total_examples', 0)}")
        logger.info(f"Valid Examples: {summary.get('valid_examples', 0)}")
        logger.info(f"Success Rate: {summary.get('success_rate', 0.0):.1%}")
        
        if 'average_overall_score' in summary:
            logger.info(f"Average Overall Score: {summary['average_overall_score']:.3f}")
            logger.info(f"Score Range: {summary.get('min_overall_score', 0):.3f} - {summary.get('max_overall_score', 0):.3f}")
            logger.info(f"High Performance (>0.8): {summary.get('scores_above_0.8', 0)}")
            logger.info(f"Low Performance (<0.5): {summary.get('scores_below_0.5', 0)}")
        
        logger.info("\nMetric Averages:")
        for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
            key = f'average_{metric}'
            if key in summary:
                logger.info(f"  {metric.replace('_', ' ').title()}: {summary[key]:.3f}")
        
        logger.info(f"\nMetrics Used: {metadata.get('metrics_used', [])}")
        logger.info(f"Evaluator: {metadata.get('evaluator', 'ragas')}")
        
        failed_count = summary.get('failed_count', 0)
        if failed_count > 0:
            logger.warning(f"Failed Evaluations: {failed_count}")

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ragas_evaluation.log')
        ]
    )

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run ragas-based RAG evaluation')
    
    # Dataset options
    parser.add_argument('--dataset-path', 
                       default='data/evaluation/rag_evaluation_dataset.json',
                       help='Path to test dataset')
    parser.add_argument('--generate-dataset', action='store_true',
                       help='Generate new test dataset from RAG documents')
    parser.add_argument('--rag-documents', 
                       default='data/processed/electronics_rag_documents.jsonl',
                       help='Path to RAG documents for dataset generation')
    parser.add_argument('--test-size', type=int, default=50,
                       help='Number of test examples to generate')
    
    # Evaluation options
    parser.add_argument('--single-query', type=str,
                       help='Run evaluation on a single query')
    parser.add_argument('--ground-truth', type=str,
                       help='Ground truth for single query evaluation')
    parser.add_argument('--output-dir', 
                       default='data/evaluation/ragas_results',
                       help='Output directory for results')
    
    # Configuration
    parser.add_argument('--openai-api-key',
                       help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--wandb-api-key',
                       help='Weights & Biases API key (or set WANDB_API_KEY env var)')
    parser.add_argument('--no-weave', action='store_true',
                       help='Disable Weave tracing')
    
    # Report options
    parser.add_argument('--report-format', choices=['html', 'json', 'csv'], 
                       default='html',
                       help='Report format')
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Get API keys
    openai_api_key = args.openai_api_key or os.getenv('OPENAI_API_KEY')
    wandb_api_key = args.wandb_api_key or os.getenv('WANDB_API_KEY')
    
    try:
        # Initialize runner
        runner = RagasEvaluationRunner(
            openai_api_key=openai_api_key,
            use_weave=not args.no_weave,
            wandb_api_key=wandb_api_key
        )
        
        # Generate dataset if requested
        if args.generate_dataset:
            if not openai_api_key:
                logger.error("OpenAI API key required for dataset generation")
                sys.exit(1)
            
            logger.info("Generating new test dataset...")
            dataset_path = runner.generate_test_dataset(
                args.rag_documents, 
                args.dataset_path, 
                args.test_size
            )
            logger.info(f"Dataset generated: {dataset_path}")
            return
        
        # Single query evaluation
        if args.single_query:
            result = runner.run_single_query_evaluation(
                args.single_query, 
                args.ground_truth
            )
            logger.info("Single query evaluation completed")
            return
        
        # Full evaluation
        result = runner.run_evaluation(args.dataset_path, args.output_dir)
        
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("Output files:")
        for file_type, file_path in result['output_files'].items():
            logger.info(f"  {file_type}: {file_path}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Make sure the dataset file exists or use --generate-dataset to create one")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()