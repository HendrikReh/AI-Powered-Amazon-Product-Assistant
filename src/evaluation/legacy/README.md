# Legacy Evaluation System

This folder contains the archived custom evaluation system that was replaced by ragas-based evaluation.

## Archived Files

- `evaluator.py` - Original custom RAG evaluator using Weave
- `scorers.py` - Custom scoring functions (relevance, accuracy, completeness, factuality, quality)
- Project root:
  - `legacy_run_evaluation.py` - Original evaluation runner
  - `legacy_run_synthetic_evaluation.py` - Original synthetic data evaluation
  - `legacy_test_rag_system.py` - Original RAG system test script

## Migration to Ragas

The custom evaluation system has been replaced with industry-standard ragas metrics:

### Old Custom Metrics → New Ragas Metrics
- **Relevance Scorer** → `answer_relevancy`
- **Accuracy Scorer** → `faithfulness` + `answer_correctness`
- **Completeness Scorer** → `context_recall`
- **Factuality Scorer** → `faithfulness`
- **Quality Scorer** → `answer_relevancy`

### New Ragas Files
- `src/evaluation/ragas_evaluator.py` - Main ragas evaluator
- `src/evaluation/rag_adapter.py` - RAG system adapter for ragas
- `src/evaluation/ragas_reporter.py` - HTML/JSON report generation
- `run_ragas_evaluation.py` - New ragas-based evaluation runner
- `scripts/generate_ragas_dataset.py` - Test dataset generation

## Benefits of Ragas Migration

1. **Standardized Metrics**: Industry-recognized evaluation metrics
2. **Better Benchmarking**: Compare with other RAG systems
3. **Automated Test Generation**: Generate diverse test cases from documents
4. **Enhanced Reporting**: Professional HTML reports with visualizations
5. **Reduced Maintenance**: Use proven evaluation framework

## Usage

To use the new ragas-based evaluation:

```bash
# Generate test dataset
uv run python scripts/generate_ragas_dataset.py

# Run evaluation
uv run python run_ragas_evaluation.py

# Generate HTML report
uv run python run_ragas_evaluation.py --report-format html
```

## Backward Compatibility

The legacy files are preserved for reference and can be used if needed:

```bash
# Run legacy evaluation
uv run python legacy_run_evaluation.py --mock-llm

# Run legacy synthetic evaluation  
uv run python legacy_run_synthetic_evaluation.py

# Run legacy RAG system test
uv run python legacy_test_rag_system.py
```

---
*Archived on: $(date)*
*Replaced by: Ragas-based evaluation framework*