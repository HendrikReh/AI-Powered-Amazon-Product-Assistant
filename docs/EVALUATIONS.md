# RAG System Evaluation Framework

This document describes the comprehensive evaluation framework implemented for the AI-Powered Amazon Product Assistant RAG system.

## Overview

The evaluation framework provides systematic testing and measurement of RAG system performance using Weights & Biases Weave for experiment tracking and analysis. It includes automated scoring across multiple dimensions and detailed performance analytics.

## Architecture

### Core Components

```
src/evaluation/
├── __init__.py          # Module interface
├── evaluator.py         # Main RAG evaluator using Weave
├── dataset.py           # Evaluation dataset creation and management
└── scorers.py           # Scoring functions for metrics

run_evaluation.py        # Command-line evaluation runner
```

### Integration Points

- **RAG Pipeline**: Seamlessly integrates with existing `src/rag/query_processor.py`
- **Vector Database**: Works with both local and Docker implementations
- **Weave Tracing**: Leverages existing Weave integration for detailed tracking
- **LLM Providers**: Compatible with all supported providers (OpenAI, Groq, Google)

## Evaluation Metrics

### 1. Relevance Score (0-1)
**Purpose**: Measures how well the response addresses the user's query

**Components**:
- **Topic Coverage** (70% weight): Checks if expected topics are covered
- **Query Coverage** (30% weight): Measures overlap between query terms and response

**Implementation**:
```python
@weave.op()
def relevance_scorer(query: str, response: str, expected_topics: List[str]) -> Dict[str, Any]
```

### 2. Accuracy Score (0-1)
**Purpose**: Evaluates factual correctness and product mention accuracy

**Components**:
- **Fact Accuracy** (80% weight): Compares factual claims with expected answers
- **Product Accuracy** (20% weight): Verifies correct product mentions

**Features**:
- Extracts numerical facts and specifications
- Validates product name mentions
- Cross-references with ground truth data

### 3. Completeness Score (0-1)
**Purpose**: Assesses response depth and thoroughness

**Components**:
- **Length Ratio** (40% weight): Response length vs expected answer length
- **Content Depth** (40% weight): Query-type specific depth indicators
- **Structure** (20% weight): Organization and paragraph structure

**Query-Type Indicators**:
- `product_info`: Features, specifications, price, performance
- `product_reviews`: Customer feedback, ratings, experiences
- `product_comparison`: Differences, advantages, similarities
- `product_complaints`: Problems, issues, negative aspects
- `product_recommendation`: Suggestions, alternatives, considerations
- `use_case`: Suitability, effectiveness, application scenarios

### 4. Factuality Score (0-1)
**Purpose**: Detects contradictions and verifies claims

**Components**:
- **Contradiction Detection** (40% weight): Identifies internal contradictions
- **Factual Claims Verification** (40% weight): Validates numerical claims
- **Uncertainty Handling** (20% weight): Appropriate use of uncertainty language

**Features**:
- Pattern-based contradiction detection
- Numerical claim validation against expected ranges
- Assessment of appropriate hedging language

### 5. Quality Score (0-1)
**Purpose**: Evaluates overall response quality and user experience

**Components**:
- **Clarity** (40% weight): Sentence length and structure assessment
- **Helpfulness** (40% weight): Actionable information and specificity
- **Coherence** (20% weight): Logical flow and topic consistency

## Evaluation Dataset

### Dataset Composition

**Total Examples**: 14 comprehensive test cases
**Query Types Distribution**:
- Product Information: 3 examples
- Product Reviews: 2 examples  
- Product Complaints: 2 examples
- Product Comparison: 2 examples
- Product Recommendations: 3 examples
- Use Case Analysis: 2 examples

**Difficulty Distribution**:
- Easy: 3 examples (21%)
- Medium: 6 examples (43%)
- Hard: 5 examples (36%)

### Example Structure

Each evaluation example includes:

```python
@dataclass
class EvaluationExample:
    query: str                    # User query
    expected_answer: str          # Ground truth answer
    query_type: str              # Classification of query type
    expected_products: List[str]  # Products that should be mentioned
    expected_topics: List[str]    # Topics that should be covered
    difficulty: str              # easy, medium, hard
    metadata: Dict[str, Any]     # Additional context
```

### Sample Evaluation Examples

#### Product Information Query
```python
EvaluationExample(
    query="What are the key features of iPhone charging cables?",
    expected_answer="iPhone charging cables typically feature Lightning connectors, various lengths (3ft, 6ft, 10ft), MFi certification for compatibility, durable materials like braided nylon, fast charging support, and data transfer capabilities...",
    query_type="product_info",
    expected_products=["lightning_cable", "iphone_charger", "apple_cable"],
    expected_topics=["features", "compatibility", "durability", "charging_speed"],
    difficulty="easy"
)
```

#### Product Comparison Query
```python
EvaluationExample(
    query="Compare Ethernet cables vs USB cables for data transfer",
    expected_answer="Ethernet cables are designed for network connectivity with speeds up to 10Gbps (Cat6a/Cat7), longer distances (100m), and stable connections. USB cables prioritize device connectivity with varying speeds...",
    query_type="product_comparison",
    expected_products=["ethernet_cable", "usb_cable", "network_cable"],
    expected_topics=["speed", "distance", "reliability", "use_cases", "specifications"],
    difficulty="hard"
)
```

## Usage Guide

### Command Line Interface

#### 1. Create Evaluation Dataset
```bash
uv run python run_evaluation.py --create-dataset
```

**Output**: Creates `data/evaluation/rag_evaluation_dataset.json` with 14 test examples

#### 2. Single Query Evaluation
```bash
uv run python run_evaluation.py --single-query "What are iPhone charger features?" --mock-llm
```

**Output**: Detailed metrics for a single query:
```
Overall Score: 0.576
Relevance: 0.180
Accuracy: 0.800
Completeness: 0.510
Factuality: 0.867
Quality: 0.661
```

#### 3. Full Evaluation Suite
```bash
uv run python run_evaluation.py --mock-llm --project-name "rag-evaluation"
```

**Features**:
- Runs all 14 evaluation examples
- Logs detailed results to Weights & Biases
- Provides comprehensive performance analysis
- Generates score distributions and insights

#### 4. Custom Dataset Evaluation
```bash
uv run python run_evaluation.py --dataset-path "data/evaluation/custom_dataset.json" --mock-llm
```

### Integration with Real LLM Providers

To use with actual LLM providers instead of mock client:

```python
# Example integration with existing LLM client
from evaluation.evaluator import RAGEvaluator
from rag.query_processor import create_rag_processor
# Import your actual LLM client

rag_processor = create_rag_processor()
llm_client = YourLLMClient()  # Replace with actual client
evaluator = RAGEvaluator(rag_processor, llm_client)

# Run evaluation
result = asyncio.run(evaluator.run_evaluation())
```

## Weave Integration

### Automatic Experiment Tracking

The evaluation framework automatically logs to Weights & Biases Weave:

- **Model Predictions**: Full RAG system responses
- **Individual Scores**: All 5 metric scores per example
- **Detailed Breakdowns**: Sub-metric components and explanations
- **Performance Analytics**: Aggregated statistics and distributions
- **Execution Metrics**: Response times and system performance

### Dashboard Features

**Available in Weave Dashboard**:
- Score trend analysis across examples
- Query type performance comparison
- Difficulty level breakdown
- Individual example drill-down
- Model latency tracking
- Error analysis and debugging

### Example Weave Output

```json
{
  "overall_scorer": {
    "overall_score": {
      "mean": 0.41014434516200504
    },
    "relevance": {
      "mean": 0.3583333333333332
    },
    "accuracy": {
      "mean": 0.06666666666666667
    },
    "completeness": {
      "mean": 0.40880042951372914
    },
    "factuality": {
      "mean": 0.8666666666666667
    },
    "quality": {
      "mean": 0.6142283950617284
    }
  },
  "model_latency": {
    "mean": 0.917356014251709
  }
}
```

## Performance Benchmarks

### Baseline Performance (Mock LLM)

**Average Scores Across 14 Examples**:
- Overall Score: 0.41 (41%)
- Relevance: 0.36 (36%)
- Accuracy: 0.07 (7%)
- Completeness: 0.41 (41%)
- Factuality: 0.87 (87%)
- Quality: 0.61 (61%)

**Performance Distribution**:
- Excellent (>0.8): 0 examples
- Good (0.6-0.8): 0 examples  
- Acceptable (0.4-0.6): 1 example
- Poor (<0.4): 2 examples

**Key Insights**:
- High factuality scores indicate good contradiction detection
- Low accuracy reflects mock LLM limitations in product-specific responses
- Quality scores show reasonable response structure and clarity
- Relevance improvements needed for better topic coverage

### Recommended Performance Targets

**Production System Targets**:
- Overall Score: >0.75
- Relevance: >0.80
- Accuracy: >0.85
- Completeness: >0.70
- Factuality: >0.90
- Quality: >0.80

## Customization and Extension

### Adding New Metrics

Create new scoring functions in `src/evaluation/scorers.py`:

```python
class CustomScorer:
    @weave.op()
    def __call__(self, query: str, response: str, context: Dict) -> Dict[str, Any]:
        # Implement custom scoring logic
        return {
            'custom_score': score,
            'details': detailed_breakdown
        }
```

### Expanding the Dataset

Add new examples in `src/evaluation/dataset.py`:

```python
def create_evaluation_dataset() -> List[EvaluationExample]:
    examples = [
        # Existing examples...
        EvaluationExample(
            query="Your new test query",
            expected_answer="Expected response",
            query_type="your_query_type",
            expected_products=["product1", "product2"],
            expected_topics=["topic1", "topic2"],
            difficulty="medium",
            metadata={"category": "electronics"}
        )
    ]
    return examples
```

### Custom Query Types

Extend query type support in `completeness_scorer`:

```python
depth_indicators = {
    'your_new_type': ['indicator1', 'indicator2', 'indicator3'],
    # ... existing types
}
```

## Continuous Integration

### Automated Evaluation Pipeline

Recommended CI/CD integration:

```yaml
# .github/workflows/evaluation.yml
name: RAG Evaluation
on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: uv sync
      - name: Run evaluation
        run: uv run python run_evaluation.py --mock-llm
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
```

### Performance Regression Detection

Monitor key metrics for regressions:

```python
# Example threshold checks
def check_performance_regression(current_scores, baseline_scores):
    regressions = []
    for metric in ['overall_score', 'relevance', 'accuracy']:
        if current_scores[metric] < baseline_scores[metric] - 0.05:
            regressions.append(f"{metric}: {current_scores[metric]:.3f} < {baseline_scores[metric]:.3f}")
    return regressions
```

## Future Enhancements

### Planned Features

1. **Semantic Similarity Scoring**: Using embedding-based similarity for response quality
2. **Retrieval Quality Metrics**: Measuring RAG context relevance and coverage
3. **Multi-turn Conversation Evaluation**: Testing conversational coherence
4. **A/B Testing Framework**: Comparing different RAG configurations
5. **User Feedback Integration**: Incorporating real user ratings
6. **Domain-Specific Metrics**: Electronics product knowledge validation

### Advanced Analytics

1. **Query Type Performance Analysis**: Detailed breakdown by query category
2. **Difficulty Progression Tracking**: Performance across complexity levels
3. **Temporal Performance Monitoring**: Tracking improvements over time
4. **Error Pattern Analysis**: Identifying common failure modes
5. **Correlation Analysis**: Understanding metric relationships

## Troubleshooting

### Common Issues

**Issue**: "No valid scores found in evaluation result"
**Solution**: Check that the `analyze_results` function matches the actual Weave result structure

**Issue**: "RAGSystemModel object has no field"
**Solution**: Ensure private attributes (`_rag_processor`, `_llm_client`) are used in the Model class

**Issue**: "Vector database not initialized"
**Solution**: Verify that `data/processed/electronics_rag_documents.jsonl` exists and is accessible

### Debug Commands

```bash
# Test RAG processor separately
uv run python src/rag/query_processor.py

# Test individual scorer
uv run python -c "from src.evaluation.scorers import RelevanceScorer; print('Scorers loaded successfully')"

# Check dataset creation
uv run python -c "from src.evaluation.dataset import create_evaluation_dataset; print(f'Created {len(create_evaluation_dataset())} examples')"
```

## API Reference

### RAGEvaluator Class

```python
class RAGEvaluator:
    def __init__(self, rag_processor, llm_client)
    async def run_evaluation(self, examples=None, project_name="rag-evaluation") -> Dict[str, Any]
    def run_single_evaluation(self, query, expected_answer, expected_products, expected_topics, query_type) -> EvaluationMetrics
    def analyze_results(self, evaluation_result) -> Dict[str, Any]
```

### Scoring Functions

```python
# All scorers follow this pattern
@weave.op()
def scorer_function(input_params) -> Dict[str, Any]:
    return {
        'primary_score': float,  # Main metric (0-1)
        'sub_metrics': dict,     # Detailed breakdown
        'details': dict          # Additional information
    }
```

---

**Last Updated**: 2025-07-01  
**Version**: 1.0.0  
**Maintainer**: AI Engineering Team