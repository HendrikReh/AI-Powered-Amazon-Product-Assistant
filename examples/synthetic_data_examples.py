"""
Examples demonstrating synthetic test data generation for RAG evaluation.
This file shows various approaches and best practices for creating synthetic data.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from evaluation.synthetic_data_generator import (
    SyntheticDataGenerator,
    SyntheticDataConfig,
    create_synthetic_dataset,
    create_mixed_dataset
)
from evaluation.dataset import create_evaluation_dataset
import weave


def example_1_basic_synthetic_generation():
    """Example 1: Basic synthetic data generation."""
    print("=== Example 1: Basic Synthetic Data Generation ===")
    
    # Create default configuration
    config = SyntheticDataConfig()
    
    # Generate synthetic examples
    synthetic_examples = create_synthetic_dataset(config, num_examples=10)
    
    print(f"Generated {len(synthetic_examples)} synthetic examples")
    
    # Show sample
    for i, example in enumerate(synthetic_examples[:3]):
        print(f"\nExample {i+1}:")
        print(f"Query: {example.query}")
        print(f"Type: {example.query_type}")
        print(f"Difficulty: {example.difficulty}")
        print(f"Synthetic: {example.metadata.get('synthetic', False)}")


def example_2_custom_configuration():
    """Example 2: Custom configuration for specific testing needs."""
    print("\n=== Example 2: Custom Configuration ===")
    
    # Custom configuration for gaming-focused evaluation
    config = SyntheticDataConfig(
        num_examples_per_category=2,
        difficulty_distribution={"easy": 0.2, "medium": 0.3, "hard": 0.5},  # More challenging
        query_type_weights={
            "product_info": 0.3,
            "product_reviews": 0.3,
            "product_comparison": 0.2,
            "product_complaints": 0.1,
            "product_recommendation": 0.1,
            "use_case": 0.0  # Skip use_case queries
        },
        variation_techniques=["rephrase", "specificity", "context"]
    )
    
    generator = SyntheticDataGenerator(config)
    examples = generator.generate_synthetic_examples(8)
    
    print(f"Generated {len(examples)} examples with custom config")
    
    # Show distribution
    difficulties = {}
    query_types = {}
    for ex in examples:
        difficulties[ex.difficulty] = difficulties.get(ex.difficulty, 0) + 1
        query_types[ex.query_type] = query_types.get(ex.query_type, 0) + 1
    
    print(f"Difficulty distribution: {difficulties}")
    print(f"Query type distribution: {query_types}")


def example_3_mixed_dataset():
    """Example 3: Creating mixed dataset (original + synthetic)."""
    print("\n=== Example 3: Mixed Dataset Creation ===")
    
    # Load original examples
    original_examples = create_evaluation_dataset()
    print(f"Original dataset: {len(original_examples)} examples")
    
    # Create mixed dataset with 50% synthetic data
    mixed_examples = create_mixed_dataset(
        original_examples, 
        synthetic_ratio=0.5,
        config=SyntheticDataConfig()
    )
    
    print(f"Mixed dataset: {len(mixed_examples)} examples")
    
    # Count synthetic vs original
    synthetic_count = sum(1 for ex in mixed_examples if ex.metadata.get("synthetic", False))
    original_count = len(mixed_examples) - synthetic_count
    
    print(f"Original: {original_count}, Synthetic: {synthetic_count}")


def example_4_dataset_augmentation():
    """Example 4: Augmenting existing dataset with variations."""
    print("\n=== Example 4: Dataset Augmentation ===")
    
    # Start with original dataset
    original_examples = create_evaluation_dataset()[:5]  # Use first 5 for demo
    
    generator = SyntheticDataGenerator()
    augmented_examples = generator.augment_existing_dataset(
        original_examples,
        augmentation_factor=1.0  # 100% augmentation (double the size)
    )
    
    print(f"Original: {len(original_examples)} examples")
    print(f"Augmented: {len(augmented_examples)} examples")
    
    # Show original vs variation
    original = original_examples[0]
    variations = [ex for ex in augmented_examples if 
                 ex.metadata.get("generation_method") == "variation" and
                 original.query[:20] in ex.metadata.get("base_example", "")]
    
    if variations:
        print(f"\nOriginal: {original.query}")
        print(f"Variation: {variations[0].query}")


def example_5_weave_integration():
    """Example 5: Using synthetic data with Weave tracking."""
    print("\n=== Example 5: Weave Integration ===")
    
    # Initialize Weave (optional - remove if not available)
    try:
        weave.init("synthetic-data-demo")
        print("Weave initialized for tracking")
        
        # Generate data with Weave ops tracking
        config = SyntheticDataConfig(num_examples_per_category=1)
        examples = create_synthetic_dataset(config, num_examples=6)
        
        print(f"Generated {len(examples)} examples with Weave tracking")
        print("Check Weave dashboard for operation traces")
        
    except Exception as e:
        print(f"Weave not available or error: {e}")
        print("Continuing without Weave tracking...")
        
        # Generate without Weave
        config = SyntheticDataConfig(num_examples_per_category=1)
        generator = SyntheticDataGenerator(config)
        examples = generator.generate_synthetic_examples(6)
        print(f"Generated {len(examples)} examples without Weave")


def example_6_quality_analysis():
    """Example 6: Analyzing synthetic data quality."""
    print("\n=== Example 6: Quality Analysis ===")
    
    config = SyntheticDataConfig()
    examples = create_synthetic_dataset(config, num_examples=15)
    
    # Analyze query diversity
    unique_queries = set(ex.query for ex in examples)
    print(f"Query uniqueness: {len(unique_queries)}/{len(examples)} unique")
    
    # Analyze query length distribution
    lengths = [len(ex.query.split()) for ex in examples]
    avg_length = sum(lengths) / len(lengths)
    print(f"Average query length: {avg_length:.1f} words")
    print(f"Length range: {min(lengths)}-{max(lengths)} words")
    
    # Analyze topic coverage
    all_topics = set()
    for ex in examples:
        all_topics.update(ex.expected_topics)
    print(f"Topic coverage: {len(all_topics)} unique topics")
    
    # Show category distribution
    categories = {}
    for ex in examples:
        cat = ex.metadata.get("subcategory", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    print(f"Category distribution: {categories}")


def example_7_best_practices_demo():
    """Example 7: Demonstrating synthetic data best practices."""
    print("\n=== Example 7: Best Practices Demo ===")
    
    print("Best Practices for Synthetic Test Data:")
    print("1. Maintain realistic query patterns")
    print("2. Ensure diverse difficulty levels")
    print("3. Cover all query types systematically")
    print("4. Include edge cases and variations")
    print("5. Balance original and synthetic data")
    print("6. Track generation methods for analysis")
    print("7. Validate synthetic data quality")
    
    # Demonstrate balanced generation
    config = SyntheticDataConfig(
        difficulty_distribution={"easy": 0.33, "medium": 0.34, "hard": 0.33},
        variation_techniques=["rephrase", "specificity", "context", "perspective"]
    )
    
    examples = create_synthetic_dataset(config, num_examples=12)
    
    # Validate balance
    difficulties = {}
    for ex in examples:
        difficulties[ex.difficulty] = difficulties.get(ex.difficulty, 0) + 1
    
    print(f"\nGenerated balanced dataset: {difficulties}")
    
    # Show technique usage
    techniques_used = set()
    for ex in examples:
        if ex.metadata.get("synthetic"):
            # This would require modification to track techniques used
            techniques_used.add(ex.metadata.get("generation_method", "unknown"))
    
    print(f"Generation techniques: {techniques_used}")


def run_all_examples():
    """Run all synthetic data examples."""
    print("Synthetic Test Data Generation Examples")
    print("=" * 50)
    
    example_1_basic_synthetic_generation()
    example_2_custom_configuration()
    example_3_mixed_dataset()
    example_4_dataset_augmentation()
    example_5_weave_integration()
    example_6_quality_analysis()
    example_7_best_practices_demo()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("\nNext steps:")
    print("1. Run: uv run python run_synthetic_evaluation.py --synthetic-only")
    print("2. Run: uv run python run_synthetic_evaluation.py --mixed-dataset")
    print("3. Check generated datasets in data/evaluation/")
    print("4. View Weave dashboard for detailed traces")


if __name__ == "__main__":
    run_all_examples()