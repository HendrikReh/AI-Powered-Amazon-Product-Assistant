# Testing the Ragas-Based Evaluation System

This guide provides step-by-step instructions for testing your new ragas-based RAG evaluation system.

## Prerequisites

1. **Dependencies Installed**: Run `uv sync` to install all dependencies
2. **RAG Documents Available**: Ensure `data/processed/electronics_rag_documents.jsonl` exists
3. **Optional**: OpenAI API key for full ragas evaluation (set `OPENAI_API_KEY` environment variable)

## Test Levels

### Level 1: Basic Functionality Test (No API Key Required)

Test that the core components work without external dependencies:

```bash
# Test RAG adapter (connects your RAG system to ragas format)
uv run python -c "
from src.evaluation.rag_adapter import RAGAdapter, MockLLMClient
adapter = RAGAdapter(llm_client=MockLLMClient())
result = adapter.process_question('What are iPhone charger features?')
print(f'✅ RAG adapter works - got {len(result.contexts)} contexts')
print(f'Answer: {result.answer[:100]}...')
"
```

Expected output:
```
✅ RAG adapter works - got 5 contexts
Answer: iPhone charging cables feature Lightning connectors, MFi certification, various lengths from 3ft to ...
```

### Level 2: Test Data Generation (Requires OpenAI API Key)

Generate a ragas test dataset from your RAG documents:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your_openai_api_key_here"

# Generate test dataset (small size for testing)
uv run python scripts/generate_ragas_dataset.py --test-size 5 --output-file "data/evaluation/test_ragas_dataset.json"
```

Expected output:
```
Loading documents from data/processed/electronics_rag_documents.jsonl
Loaded 2000 documents
Generating test dataset with 5 examples
Testset saved successfully
=== Testset Summary ===
Total examples: 5
Question type distribution:
  simple: 2
  reasoning: 2
  multi_context: 1
```

### Level 3: Single Query Evaluation

Test ragas evaluation on a single query:

```bash
# Test with mock LLM (no API key required)
uv run python run_ragas_evaluation.py --single-query "What are iPhone charger features?" --ground-truth "iPhone chargers feature Lightning connectors and fast charging"
```

Expected output:
```
🔍 Evaluating single query: What are iPhone charger features?
Single query evaluation results:
  Faithfulness: 0.500
  Answer Relevancy: 0.500
  Context Precision: 0.500
  Context Recall: 0.500
  Overall Score: 0.500
```

Note: Scores will be 0.5 (fallback values) without OpenAI API key for LLM-based evaluation.

### Level 4: Full Ragas Evaluation (Requires OpenAI API Key)

Run complete evaluation with ragas metrics:

```bash
# Use existing dataset or generate new one
uv run python run_ragas_evaluation.py --dataset-path "data/evaluation/rag_evaluation_dataset.json"
```

Expected output:
```
RAGAS EVALUATION SUMMARY
========================================
Total Examples: 14
Valid Examples: 14
Success Rate: 100.0%
Average Overall Score: 0.726
Score Range: 0.580 - 0.890
High Performance (>0.8): 6
Low Performance (<0.5): 0

Metric Averages:
  Faithfulness: 0.750
  Answer Relevancy: 0.720
  Context Precision: 0.690
  Context Recall: 0.745

Output files:
  results_json: data/evaluation/ragas_results/ragas_evaluation_results_20240705_143022.json
  summary_csv: data/evaluation/ragas_results/ragas_evaluation_summary_20240705_143022.csv
  html_report: data/evaluation/ragas_results/ragas_evaluation_report_20240705_143022.html
```

### Level 5: Test Report Generation

Test the HTML report generation:

```bash
# Generate test report with mock data
uv run python -c "
from src.evaluation.ragas_reporter import RagasReporter
from src.evaluation.ragas_evaluator import RagasEvaluationResult

# Create mock results
mock_results = [
    RagasEvaluationResult(
        faithfulness=0.8, answer_relevancy=0.7, context_precision=0.75,
        context_recall=0.7, overall_score=0.75,
        details={'question': 'Test question 1'}
    ),
    RagasEvaluationResult(
        faithfulness=0.6, answer_relevancy=0.8, context_precision=0.65,
        context_recall=0.8, overall_score=0.71,
        details={'question': 'Test question 2'}
    )
]

mock_evaluation = {
    'results': mock_results,
    'summary': {
        'total_examples': 2,
        'average_overall_score': 0.73,
        'success_rate': 1.0,
        'scores_above_0.8': 1,
        'scores_below_0.5': 0,
        'average_faithfulness': 0.7,
        'average_answer_relevancy': 0.75,
        'average_context_precision': 0.7,
        'average_context_recall': 0.75
    },
    'metadata': {
        'evaluator': 'ragas',
        'metrics_used': ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
    }
}

reporter = RagasReporter()
html_path = reporter.generate_html_report(mock_evaluation, 'test_ragas_report.html')
print(f'✅ Test HTML report generated: {html_path}')
"
```

## Testing Legacy System (For Comparison)

Test that the legacy system still works:

```bash
# Test legacy evaluation
uv run python legacy_run_evaluation.py --single-query "What are iPhone charger features?" --mock-llm

# Test legacy RAG system
uv run python legacy_test_rag_system.py
```

## Troubleshooting

### Common Issues:

1. **Import Errors**: Run `uv sync` to ensure all dependencies are installed
2. **Missing RAG Documents**: Ensure `data/processed/electronics_rag_documents.jsonl` exists
3. **OpenAI API Errors**: Check that your API key is valid and has credits
4. **ChromaDB Errors**: These are warnings and don't affect functionality

### Dependencies Check:

```bash
# Check if ragas is installed
uv run python -c "import ragas; print(f'Ragas version: {ragas.__version__}')"

# Check if langchain is installed
uv run python -c "import langchain; print('Langchain available')"

# Check if your RAG documents exist
ls -la data/processed/electronics_rag_documents.jsonl
```

## Performance Expectations

- **Basic functionality**: Should work immediately
- **Dataset generation**: ~30 seconds for 5 examples, ~5 minutes for 50 examples
- **Single query evaluation**: ~2-5 seconds
- **Full evaluation**: ~1-3 minutes for 14 examples

## Validation Checklist

- [ ] RAG adapter connects to your existing RAG system
- [ ] Mock LLM client generates reasonable answers
- [ ] Context extraction works (should get 3-5 contexts per query)
- [ ] Ragas metrics calculate (even if using fallback values)
- [ ] HTML reports generate with charts and insights
- [ ] Legacy system still accessible for comparison

## Next Steps

Once testing is successful:

1. **Generate Production Dataset**: Create larger test datasets (100+ examples)
2. **Set Up Monitoring**: Integrate with Weave tracing for production monitoring
3. **Benchmark Performance**: Compare ragas scores against legacy metrics
4. **Optimize RAG System**: Use ragas insights to improve retrieval and generation

## Getting Help

- Check `src/evaluation/legacy/README.md` for legacy system documentation
- Review `CLAUDE.md` for complete command reference
- Examine example outputs in `data/evaluation/ragas_results/`