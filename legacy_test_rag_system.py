"""
Test script for the RAG system implementation.
Tests core functionality and performance.
"""

import time
import logging
import weave
import sys
from pathlib import Path

# Add the chatbot-ui directory to sys.path to import config
chatbot_ui_dir = Path(__file__).parent / "src" / "chatbot-ui"
sys.path.append(str(chatbot_ui_dir))

from src.rag.query_processor import create_rag_processor
from core.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Weave for tracing (following streamlit_app.py pattern)
if config.WANDB_API_KEY:
    try:
        import wandb
        # Login to W&B with API key before initializing Weave
        wandb.login(key=config.WANDB_API_KEY, anonymous="never", force=True)
        weave.init(project_name="Bootcamp")
        logger.info("🔍 Weave tracing enabled")
    except Exception as e:
        logger.error(f"❌ Weave initialization failed: {str(e)}")
        logger.info("Continuing without tracing...")
else:
    logger.info("ℹ️ Weave tracing disabled (no WANDB_API_KEY)")

@weave.op()
def test_rag_system():
    """Test the RAG system with various query types."""
    print("🔍 Testing RAG System for Amazon Electronics Assistant")
    print("=" * 60)
    
    # Initialize processor
    print("\n1. Initializing RAG processor...")
    start_time = time.time()
    processor = create_rag_processor()
    init_time = time.time() - start_time
    print(f"   ✅ Initialized in {init_time:.2f} seconds")
    
    if not processor.vector_db:
        print("   ❌ Vector database not available")
        return False
    
    # Test database stats
    print("\n2. Checking database stats...")
    stats = processor.vector_db.get_collection_stats()
    print(f"   📊 Total documents: {stats['total_documents']}")
    print(f"   📈 Document types: {stats['document_types']}")
    
    # Test queries
    test_queries = [
        ("Product Info", "What do people say about iPhone charger cables?"),
        ("Product Reviews", "Reviews for laptop backpacks"),
        ("Product Complaints", "What are the main complaints about Fire TV?"),
        ("Product Comparison", "Compare ethernet cables and USB cables"),
        ("Budget Recommendation", "Recommend a budget-friendly tablet under $100"),
        ("Use Case", "Is the Ethernet cable good for gaming?"),
        ("General Search", "best electronics for home office")
    ]
    
    print("\n3. Testing different query types...")
    
    results_summary = []
    total_test_time = 0
    
    for query_type, query in test_queries:
        print(f"\n   🔍 {query_type}: '{query}'")
        
        start_time = time.time()
        result = processor.process_query(query)
        query_time = time.time() - start_time
        total_test_time += query_time
        
        if result["success"]:
            metadata = result["metadata"]
            print(f"      ✅ Success ({query_time:.2f}s)")
            print(f"      📦 Products: {metadata['num_products']}")
            print(f"      💬 Reviews: {metadata['num_reviews']}")
            print(f"      🎯 Query type: {metadata['query_type']}")
            
            results_summary.append({
                "query_type": query_type,
                "success": True,
                "time": query_time,
                "products": metadata['num_products'],
                "reviews": metadata['num_reviews']
            })
        else:
            print(f"      ❌ Failed: {result.get('error', 'Unknown error')}")
            results_summary.append({
                "query_type": query_type,
                "success": False,
                "time": query_time,
                "error": result.get('error', 'Unknown error')
            })
    
    # Performance summary
    print("\n4. Performance Summary")
    print("-" * 40)
    successful_queries = [r for r in results_summary if r["success"]]
    
    if successful_queries:
        avg_time = sum(r["time"] for r in successful_queries) / len(successful_queries)
        avg_products = sum(r["products"] for r in successful_queries) / len(successful_queries)
        avg_reviews = sum(r["reviews"] for r in successful_queries) / len(successful_queries)
        
        print(f"   ✅ Success rate: {len(successful_queries)}/{len(test_queries)} ({len(successful_queries)/len(test_queries)*100:.1f}%)")
        print(f"   ⏱️  Average query time: {avg_time:.2f} seconds")
        print(f"   📦 Average products retrieved: {avg_products:.1f}")
        print(f"   💬 Average reviews retrieved: {avg_reviews:.1f}")
        print(f"   🚀 Total test time: {total_test_time:.2f} seconds")
    
    # RAG prompt example
    print("\n5. Sample RAG Prompt Generation")
    print("-" * 40)
    sample_query = "What do people say about iPhone charger cables?"
    result = processor.process_query(sample_query)
    
    if result["success"]:
        print(f"   Query: {sample_query}")
        print(f"   Enhanced Prompt Length: {len(result['enhanced_prompt'])} characters")
        print(f"   Preview:\n{result['enhanced_prompt'][:300]}...")
    
    print("\n" + "=" * 60)
    print("🎉 RAG System Test Complete!")
    
    return len(successful_queries) == len(test_queries)

if __name__ == "__main__":
    success = test_rag_system()
    exit(0 if success else 1)