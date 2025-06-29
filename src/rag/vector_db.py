"""
Vector database implementation for Amazon Electronics RAG system.
Handles document ingestion, embedding, and retrieval using ChromaDB.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import chromadb
from chromadb.config import Settings
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ElectronicsVectorDB:
    """Vector database for Amazon Electronics products and reviews."""
    
    def __init__(self, persist_directory: str = "data/chroma_db"):
        """Initialize the vector database."""
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Collection for storing both products and reviews
        self.collection_name = "electronics_products_reviews"
        self.collection = None
        
    def create_collection(self) -> None:
        """Create or get the collection for storing documents."""
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Retrieved existing collection: {self.collection_name}")
        except Exception:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def load_documents(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """Load documents from JSONL file."""
        documents = []
        with open(jsonl_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    doc = json.loads(line.strip())
                    documents.append(doc)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line: {e}")
                    continue
        
        logger.info(f"Loaded {len(documents)} documents from {jsonl_path}")
        return documents
    
    def prepare_document_for_ingestion(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a document for vector database ingestion."""
        doc_id = doc["id"]
        content = doc["content"]
        
        # Base metadata
        metadata = {
            "type": doc["type"],
            "parent_asin": doc["parent_asin"],
        }
        
        # Add type-specific metadata
        if doc["type"] == "product":
            metadata.update({
                "title": doc.get("title", ""),
                "price": doc.get("price"),
                "average_rating": doc.get("average_rating"),
                "rating_number": doc.get("rating_number", 0),
                "review_count": doc.get("review_count", 0),
                "store": doc.get("store", ""),
                "categories": json.dumps(doc.get("categories", [])),
                "features": doc.get("features", "")[:500],  # Truncate for metadata
            })
        elif doc["type"] == "review_summary":
            metadata.update({
                "product_title": doc.get("product_title", ""),
                "total_reviews": doc.get("total_reviews", 0),
                "positive_reviews": doc.get("positive_reviews", 0),
                "negative_reviews": doc.get("negative_reviews", 0),
            })
        
        # Clean metadata - remove None values and ensure string values
        cleaned_metadata = {}
        for key, value in metadata.items():
            if value is not None:
                if isinstance(value, (int, float)):
                    cleaned_metadata[key] = value
                else:
                    cleaned_metadata[key] = str(value)
        
        return {
            "id": doc_id,
            "content": content,
            "metadata": cleaned_metadata
        }
    
    def ingest_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """Ingest documents into the vector database."""
        if not self.collection:
            raise ValueError("Collection not initialized. Call create_collection() first.")
        
        logger.info(f"Starting ingestion of {len(documents)} documents...")
        
        # Process documents in batches
        for i in tqdm(range(0, len(documents), batch_size), desc="Ingesting documents"):
            batch = documents[i:i + batch_size]
            
            ids = []
            contents = []
            metadatas = []
            
            for doc in batch:
                prepared_doc = self.prepare_document_for_ingestion(doc)
                ids.append(prepared_doc["id"])
                contents.append(prepared_doc["content"])
                metadatas.append(prepared_doc["metadata"])
            
            try:
                self.collection.add(
                    ids=ids,
                    documents=contents,
                    metadatas=metadatas
                )
            except Exception as e:
                logger.error(f"Failed to ingest batch starting at index {i}: {e}")
                continue
        
        logger.info("Document ingestion completed.")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        if not self.collection:
            return {"error": "Collection not initialized"}
        
        count = self.collection.count()
        
        # Get sample of documents to analyze types
        sample_results = self.collection.get(limit=100)
        
        type_counts = {}
        for metadata in sample_results['metadatas']:
            doc_type = metadata.get('type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        return {
            "total_documents": count,
            "document_types": type_counts,
            "collection_name": self.collection_name
        }
    
    def search_products(
        self, 
        query: str, 
        n_results: int = 5,
        price_range: Optional[tuple] = None,
        min_rating: Optional[float] = None,
        categories: Optional[List[str]] = None,
        store: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search for products with optional filtering."""
        if not self.collection:
            raise ValueError("Collection not initialized")
        
        # Build where clause for filtering
        where_conditions = {"type": {"$eq": "product"}}
        
        if price_range:
            where_conditions["$and"] = [
                {"price": {"$gte": price_range[0]}},
                {"price": {"$lte": price_range[1]}}
            ]
        
        if min_rating:
            if "$and" in where_conditions:
                where_conditions["$and"].append({"average_rating": {"$gte": min_rating}})
            else:
                where_conditions["average_rating"] = {"$gte": min_rating}
        
        if store:
            if "$and" in where_conditions:
                where_conditions["$and"].append({"store": {"$eq": store}})
            else:
                where_conditions["store"] = {"$eq": store}
        
        where_clause = where_conditions
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause
            )
            
            return {
                "query": query,
                "results": results,
                "filters_applied": {
                    "price_range": price_range,
                    "min_rating": min_rating,
                    "categories": categories,
                    "store": store
                }
            }
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"error": str(e)}
    
    def search_reviews(
        self, 
        query: str, 
        n_results: int = 5,
        product_asin: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search for product reviews."""
        if not self.collection:
            raise ValueError("Collection not initialized")
        
        where_conditions = {"type": "review_summary"}
        
        if product_asin:
            where_conditions["parent_asin"] = product_asin
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_conditions
            )
            
            return {
                "query": query,
                "results": results,
                "product_asin": product_asin
            }
        except Exception as e:
            logger.error(f"Review search failed: {e}")
            return {"error": str(e)}
    
    def get_product_with_reviews(self, parent_asin: str) -> Dict[str, Any]:
        """Get a specific product and its reviews by ASIN."""
        if not self.collection:
            raise ValueError("Collection not initialized")
        
        try:
            # Get both product and review documents for this ASIN
            results = self.collection.get(
                where={"parent_asin": parent_asin}
            )
            
            product_doc = None
            review_doc = None
            
            for i, metadata in enumerate(results['metadatas']):
                if metadata['type'] == 'product':
                    product_doc = {
                        'id': results['ids'][i],
                        'content': results['documents'][i],
                        'metadata': metadata
                    }
                elif metadata['type'] == 'review_summary':
                    review_doc = {
                        'id': results['ids'][i],
                        'content': results['documents'][i],
                        'metadata': metadata
                    }
            
            return {
                "parent_asin": parent_asin,
                "product": product_doc,
                "reviews": review_doc
            }
        except Exception as e:
            logger.error(f"Failed to get product with reviews: {e}")
            return {"error": str(e)}
    
    def hybrid_search(
        self, 
        query: str, 
        n_results: int = 10,
        include_products: bool = True,
        include_reviews: bool = True,
        **filters
    ) -> Dict[str, Any]:
        """Perform hybrid search across products and reviews."""
        if not self.collection:
            raise ValueError("Collection not initialized")
        
        # Build type filter
        type_filter = []
        if include_products:
            type_filter.append("product")
        if include_reviews:
            type_filter.append("review_summary")
        
        if not type_filter:
            return {"error": "Must include at least products or reviews"}
        
        where_conditions = {"type": {"$in": type_filter}}
        
        # Add additional filters
        for key, value in filters.items():
            if value is not None:
                where_conditions[key] = value
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_conditions
            )
            
            return {
                "query": query,
                "results": results,
                "search_type": "hybrid",
                "included_types": type_filter
            }
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return {"error": str(e)}
    
    def reset_collection(self) -> None:
        """Reset the collection (delete all data)."""
        if self.collection:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        self.collection = None


def setup_vector_database(jsonl_path: str, persist_directory: str = "data/chroma_db") -> ElectronicsVectorDB:
    """Setup and populate the vector database."""
    logger.info("Setting up vector database...")
    
    # Initialize database
    vector_db = ElectronicsVectorDB(persist_directory=persist_directory)
    
    # Create collection
    vector_db.create_collection()
    
    # Check if collection is already populated
    stats = vector_db.get_collection_stats()
    if stats.get("total_documents", 0) > 0:
        logger.info(f"Collection already populated with {stats['total_documents']} documents")
        return vector_db
    
    # Load and ingest documents
    documents = vector_db.load_documents(jsonl_path)
    vector_db.ingest_documents(documents)
    
    # Show final stats
    final_stats = vector_db.get_collection_stats()
    logger.info(f"Vector database setup complete: {final_stats}")
    
    return vector_db


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Setup database
    jsonl_path = "data/processed/electronics_rag_documents.jsonl"
    vector_db = setup_vector_database(jsonl_path)
    
    # Test queries
    print("\n=== Testing Product Search ===")
    results = vector_db.search_products("ethernet cable high speed", n_results=3)
    if "error" not in results:
        for i, (doc_id, distance) in enumerate(zip(results["results"]["ids"][0], results["results"]["distances"][0])):
            print(f"{i+1}. {doc_id} (distance: {distance:.3f})")
    
    print("\n=== Testing Review Search ===")
    results = vector_db.search_reviews("quality problems issues", n_results=3)
    if "error" not in results:
        for i, (doc_id, distance) in enumerate(zip(results["results"]["ids"][0], results["results"]["distances"][0])):
            print(f"{i+1}. {doc_id} (distance: {distance:.3f})")
    
    print("\n=== Collection Stats ===")
    stats = vector_db.get_collection_stats()
    print(json.dumps(stats, indent=2))