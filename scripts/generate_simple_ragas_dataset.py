"""
Simple ragas test dataset generation from existing evaluation dataset.
This converts your existing evaluation dataset to ragas format for testing.
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_existing_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load existing evaluation dataset."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def convert_to_ragas_format(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert existing examples to ragas format."""
    ragas_examples = []
    
    for example in examples:
        ragas_example = {
            'question': example.get('query', ''),
            'ground_truth': example.get('expected_answer', ''),
            'reference': example.get('expected_answer', ''),  # Some metrics require 'reference'
            'metadata': {
                'query_type': example.get('query_type'),
                'difficulty': example.get('difficulty'),
                'expected_products': example.get('expected_products', []),
                'expected_topics': example.get('expected_topics', [])
            }
        }
        ragas_examples.append(ragas_example)
    
    return ragas_examples

def create_simple_synthetic_examples() -> List[Dict[str, Any]]:
    """Create simple synthetic examples for testing."""
    examples = [
        {
            'question': 'What are the key features of iPhone charging cables?',
            'ground_truth': 'iPhone charging cables feature Lightning connectors, MFi certification, various lengths (3ft to 10ft), fast charging support, and durable materials. Quality varies by brand with Apple and certified third-party options being most reliable.',
            'reference': 'iPhone charging cables feature Lightning connectors, MFi certification, various lengths (3ft to 10ft), fast charging support, and durable materials. Quality varies by brand with Apple and certified third-party options being most reliable.',
            'metadata': {'query_type': 'product_info', 'difficulty': 'easy'}
        },
        {
            'question': 'How is Fire TV performance for streaming?',
            'ground_truth': 'Fire TV provides 1080p HD streaming, supports popular apps like Netflix and Prime Video, has Alexa Voice Remote, and compact design. Performance is generally good but may have occasional buffering with high-demand content.',
            'reference': 'Fire TV provides 1080p HD streaming, supports popular apps like Netflix and Prime Video, has Alexa Voice Remote, and compact design. Performance is generally good but may have occasional buffering with high-demand content.',
            'metadata': {'query_type': 'product_info', 'difficulty': 'medium'}
        },
        {
            'question': 'What do customers say about laptop backpack durability?',
            'ground_truth': 'Customer reviews on laptop backpack durability are mixed. Positive feedback highlights reinforced stitching and water-resistant materials. Common complaints include strap wear after 6-12 months and zipper failures.',
            'reference': 'Customer reviews on laptop backpack durability are mixed. Positive feedback highlights reinforced stitching and water-resistant materials. Common complaints include strap wear after 6-12 months and zipper failures.',
            'metadata': {'query_type': 'product_reviews', 'difficulty': 'medium'}
        },
        {
            'question': 'Compare ethernet cables vs USB cables for data transfer',
            'ground_truth': 'Ethernet cables are designed for network connectivity with speeds up to 10Gbps, longer distances (100m), and stable connections. USB cables prioritize device connectivity with varying speeds up to 40Gbps for USB-C, shorter distances (5m), and power delivery capabilities.',
            'reference': 'Ethernet cables are designed for network connectivity with speeds up to 10Gbps, longer distances (100m), and stable connections. USB cables prioritize device connectivity with varying speeds up to 40Gbps for USB-C, shorter distances (5m), and power delivery capabilities.',
            'metadata': {'query_type': 'product_comparison', 'difficulty': 'hard'}
        },
        {
            'question': 'Recommend budget-friendly noise-canceling headphones alternatives',
            'ground_truth': 'Budget-friendly alternatives include TaoTronics SoundSurge 60 ($60), Anker Soundcore Life Q20 ($40), and Cowin E7 ($50). These offer decent active noise cancellation, 20-30 hour battery life, and comfortable fit while providing 70-80% of premium performance at 20% of the cost.',
            'reference': 'Budget-friendly alternatives include TaoTronics SoundSurge 60 ($60), Anker Soundcore Life Q20 ($40), and Cowin E7 ($50). These offer decent active noise cancellation, 20-30 hour battery life, and comfortable fit while providing 70-80% of premium performance at 20% of the cost.',
            'metadata': {'query_type': 'product_recommendation', 'difficulty': 'medium'}
        }
    ]
    
    return examples

def save_ragas_dataset(examples: List[Dict[str, Any]], output_path: str):
    """Save ragas dataset to file."""
    import pandas as pd
    
    # Convert to ragas-compatible format
    ragas_data = {
        'examples': examples,
        'metadata': {
            'generator': 'simple_ragas_converter',
            'format': 'ragas_compatible',
            'created_at': str(pd.Timestamp.now()),
            'total_examples': len(examples)
        }
    }
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ragas_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Ragas dataset saved to {output_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate simple ragas test dataset')
    parser.add_argument('--input-file', 
                       default='data/evaluation/rag_evaluation_dataset.json',
                       help='Input evaluation dataset file')
    parser.add_argument('--output-file',
                       default='data/evaluation/ragas_test_dataset.json',
                       help='Output ragas dataset file')
    parser.add_argument('--synthetic-only', action='store_true',
                       help='Generate only synthetic examples')
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).parent.parent
    input_file = project_root / args.input_file
    output_file = project_root / args.output_file
    
    try:
        if args.synthetic_only:
            logger.info("Generating synthetic examples only")
            examples = create_simple_synthetic_examples()
        else:
            if input_file.exists():
                logger.info(f"Loading existing dataset from {input_file}")
                existing_examples = load_existing_dataset(str(input_file))
                examples = convert_to_ragas_format(existing_examples)
            else:
                logger.warning(f"Input file not found: {input_file}. Creating synthetic examples.")
                examples = create_simple_synthetic_examples()
        
        # Save ragas dataset
        save_ragas_dataset(examples, str(output_file))
        
        # Print summary
        logger.info("=== Dataset Summary ===")
        logger.info(f"Total examples: {len(examples)}")
        
        # Show query types
        query_types = {}
        for example in examples:
            qtype = example.get('metadata', {}).get('query_type', 'unknown')
            query_types[qtype] = query_types.get(qtype, 0) + 1
        
        logger.info(f"Query types: {query_types}")
        
        # Sample questions
        logger.info("Sample questions:")
        for i, example in enumerate(examples[:3], 1):
            logger.info(f"  Q{i}: {example['question']}")
        
        logger.info("Dataset generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()