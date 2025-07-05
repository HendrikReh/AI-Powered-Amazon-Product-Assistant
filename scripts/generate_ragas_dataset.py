"""
Generate ragas test dataset from electronics RAG documents.
"""

import json
import asyncio
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
import sys
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.prompt_values import PromptValue

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RagasDatasetGenerator:
    """Generate test dataset using ragas TestsetGenerator."""
    
    def __init__(self, openai_api_key: str):
        """Initialize with OpenAI API key."""
        self.openai_api_key = openai_api_key
        
        # Create a custom wrapper to fix compatibility issues
        class FixedLangchainLLMWrapper(LangchainLLMWrapper):
            """Fixed wrapper that provides missing methods for compatibility.
            
            This wrapper addresses the issue where ragas expects agenerate_prompt
            and generate_prompt methods, but the base LangchainLLMWrapper only
            provides agenerate_text and generate_text methods.
            """
            
            def __init__(self, langchain_llm, **kwargs):
                """Initialize the wrapper with proper parent class initialization."""
                super().__init__(langchain_llm, **kwargs)
            
            async def agenerate_prompt(self, prompt=None, prompts=None, **kwargs):
                """Provide missing agenerate_prompt method with flexible signature.
                
                This method is called by ragas framework for async text generation.
                It handles both 'prompt' and 'prompts' parameters for compatibility.
                """
                try:
                    # Handle both prompt and prompts parameters
                    if prompts is not None:
                        # Handle multiple prompts
                        if isinstance(prompts, list):
                            # Process first prompt for now
                            processed_prompt = self._process_prompt(prompts[0] if prompts else "")
                        else:
                            processed_prompt = self._process_prompt(prompts)
                    elif prompt is not None:
                        processed_prompt = self._process_prompt(prompt)
                    else:
                        processed_prompt = ""
                    
                    return await self.agenerate_text(prompt=processed_prompt, **kwargs)
                except Exception as e:
                    logger.error(f"Error in agenerate_prompt: {e}")
                    raise
            
            def generate_prompt(self, prompt=None, prompts=None, **kwargs):
                """Provide missing generate_prompt method with flexible signature.
                
                This method is called by ragas framework for sync text generation.
                It handles both 'prompt' and 'prompts' parameters for compatibility.
                """
                try:
                    # Handle both prompt and prompts parameters
                    if prompts is not None:
                        # Handle multiple prompts
                        if isinstance(prompts, list):
                            # Process first prompt for now
                            processed_prompt = self._process_prompt(prompts[0] if prompts else "")
                        else:
                            processed_prompt = self._process_prompt(prompts)
                    elif prompt is not None:
                        processed_prompt = self._process_prompt(prompt)
                    else:
                        processed_prompt = ""
                    
                    return self.generate_text(prompt=processed_prompt, **kwargs)
                except Exception as e:
                    logger.error(f"Error in generate_prompt: {e}")
                    raise
            
            def _process_prompt(self, prompt):
                """Process prompt to ensure it's in the correct format.
                
                The ragas framework may pass prompts in different formats:
                - PromptValue objects (already processed)
                - String prompts (need conversion)
                - List of prompts (need joining and conversion)
                """
                try:
                    # If it's already a PromptValue, return as is
                    if isinstance(prompt, PromptValue):
                        return prompt
                    
                    # If it's a string, convert to PromptValue
                    if isinstance(prompt, str):
                        prompt_template = PromptTemplate.from_template(prompt)
                        return prompt_template.format_prompt()
                    
                    # If it's a list or other iterable, convert to string first
                    if hasattr(prompt, '__iter__') and not isinstance(prompt, str):
                        prompt_str = ' '.join(str(p) for p in prompt)
                        prompt_template = PromptTemplate.from_template(prompt_str)
                        return prompt_template.format_prompt()
                    
                    # Fallback: convert to string and then to PromptValue
                    prompt_str = str(prompt)
                    prompt_template = PromptTemplate.from_template(prompt_str)
                    return prompt_template.format_prompt()
                    
                except Exception as e:
                    logger.error(f"Error processing prompt: {e}")
                    # Final fallback: try to create a simple string prompt
                    prompt_template = PromptTemplate.from_template(str(prompt))
                    return prompt_template.format_prompt()
        
        # Initialize LLM and embeddings for ragas
        self.llm = FixedLangchainLLMWrapper(ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=openai_api_key,
            temperature=0.3
        ))
        
        self.embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
            api_key=openai_api_key
        ))
        
        # Initialize TestsetGenerator
        self.generator = TestsetGenerator.from_langchain(
            llm=self.llm,
            embedding_model=self.embeddings
        )
    
    def load_rag_documents(self, file_path: str, max_docs: int = 100) -> List[Document]:
        """Load RAG documents and convert to Langchain Documents."""
        documents = []
        
        logger.info(f"Loading documents from {file_path} (max {max_docs} documents)")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if len(documents) >= max_docs:
                    logger.info(f"Reached maximum documents limit: {max_docs}")
                    break
                    
                try:
                    doc_data = json.loads(line.strip())
                    
                    # Create document content
                    content = doc_data.get('content', '')
                    
                    # Create metadata
                    metadata = {
                        'id': doc_data.get('id', f'doc_{line_num}'),
                        'type': doc_data.get('type', 'unknown'),
                        'source': 'electronics_rag_documents'
                    }
                    
                    # Add additional metadata based on document type
                    if doc_data.get('type') == 'product':
                        metadata.update({
                            'title': doc_data.get('title', ''),
                            'price': doc_data.get('price'),
                            'rating': doc_data.get('average_rating'),
                            'store': doc_data.get('store', ''),
                            'categories': doc_data.get('categories', [])
                        })
                    elif doc_data.get('type') == 'review_summary':
                        metadata.update({
                            'product_title': doc_data.get('product_title', ''),
                            'total_reviews': doc_data.get('total_reviews', 0),
                            'positive_reviews': doc_data.get('positive_reviews', 0),
                            'negative_reviews': doc_data.get('negative_reviews', 0)
                        })
                    
                    # Create Langchain Document
                    doc = Document(
                        page_content=content,
                        metadata=metadata
                    )
                    documents.append(doc)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def generate_testset(self, documents: List[Document], 
                        test_size: int = 50) -> Dict[str, Any]:
        """Generate test dataset using ragas."""
        logger.info(f"Generating test dataset with {test_size} examples from {len(documents)} documents")
        logger.info("⚠️  This may take several minutes depending on document count and OpenAI API response times")
        
        try:
            # Generate testset with progress monitoring
            testset = self.generator.generate_with_langchain_docs(
                documents=documents,
                testset_size=test_size,
                with_debugging_logs=True
            )
            
            logger.info("✅ Testset generation completed successfully")
            return testset
            
        except Exception as e:
            logger.error(f"❌ Testset generation failed: {e}")
            logger.info("💡 Try reducing --max-docs or --test-size for better performance")
            raise
    
    def save_testset(self, testset, output_path: str):
        """Save testset to file."""
        logger.info(f"Saving testset to {output_path}")
        
        # Convert testset to DataFrame then to dict
        if hasattr(testset, 'to_pandas'):
            df = testset.to_pandas()
            testset_dict = df.to_dict('records')
        else:
            testset_dict = testset
        
        # Convert to serializable format
        testset_data = {
            'testset': testset_dict,
            'metadata': {
                'generator': 'ragas',
                'source': 'electronics_rag_documents',
                'created_at': str(pd.Timestamp.now())
            }
        }
        
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(testset_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Testset saved successfully")
    
    def print_testset_summary(self, testset):
        """Print summary of generated testset."""
        logger.info("=== Testset Summary ===")
        
        if hasattr(testset, 'to_pandas'):
            df = testset.to_pandas()
            logger.info(f"Total examples: {len(df)}")
            
            # Sample questions
            logger.info("Sample questions:")
            question_col = 'question' if 'question' in df.columns else df.columns[0]
            for i, row in df.head(3).iterrows():
                logger.info(f"  Q{i+1}: {row.get(question_col, 'N/A')}")
            
            # Show available columns
            logger.info(f"Dataset columns: {list(df.columns)}")
        else:
            logger.info(f"Generated testset: {type(testset)}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate ragas test dataset')
    parser.add_argument('--input-file', 
                       default='data/processed/electronics_rag_documents.jsonl',
                       help='Input RAG documents file')
    parser.add_argument('--output-file',
                       default='data/evaluation/ragas_test_dataset.json',
                       help='Output test dataset file')
    parser.add_argument('--test-size', type=int, default=50,
                       help='Number of test examples to generate')
    parser.add_argument('--max-docs', type=int, default=100,
                       help='Maximum number of documents to process (for performance)')
    parser.add_argument('--openai-api-key',
                       help='OpenAI API key (or set OPENAI_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Get OpenAI API key
    import os
    openai_api_key = args.openai_api_key or os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("OpenAI API key required. Set OPENAI_API_KEY env var or use --openai-api-key")
        sys.exit(1)
    
    # Resolve paths
    project_root = Path(__file__).parent.parent
    input_file = project_root / args.input_file
    output_file = project_root / args.output_file
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)
    
    try:
        # Initialize generator
        generator = RagasDatasetGenerator(openai_api_key)
        
        # Load documents (limit for performance)
        documents = generator.load_rag_documents(str(input_file), max_docs=args.max_docs)
        
        if not documents:
            logger.error("No documents loaded")
            sys.exit(1)
        
        # Generate testset
        testset = generator.generate_testset(documents, args.test_size)
        
        # Save testset
        generator.save_testset(testset, str(output_file))
        
        # Print summary
        generator.print_testset_summary(testset)
        
        logger.info("Dataset generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()