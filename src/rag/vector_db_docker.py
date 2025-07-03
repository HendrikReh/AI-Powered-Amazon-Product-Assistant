"""
Docker-compatible vector database implementation for Amazon Electronics RAG system.
Connects to external ChromaDB service in containerized environment.
"""

import os
import json
import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import weave

# Import enhanced tracing utilities
try:
    from ..tracing.trace_utils import (
        get_current_trace_context, performance_monitor,
        VectorPerformanceMonitor
    )
except ImportError:
    # Fallback for cases where tracing utils are not available
    get_current_trace_context = lambda: None
    performance_monitor = None
    VectorPerformanceMonitor = None

logger = logging.getLogger(__name__)


class ElectronicsVectorDBDocker:
    """Docker-compatible vector database for Amazon Electronics products and reviews."""
    
    def __init__(self, chroma_host: str = None, chroma_port: int = None):
        """Initialize the vector database with Docker service connection."""
        
        # Get connection details from environment or defaults
        self.chroma_host = chroma_host or os.getenv("CHROMA_HOST", "localhost")
        self.chroma_port = chroma_port or int(os.getenv("CHROMA_PORT", "8000"))
        
        logger.info(f"Connecting to ChromaDB at {self.chroma_host}:{self.chroma_port}")
        
        # Use default embedding for Docker to avoid GTE model download
        # This saves 670MB of disk space in containers
        self.embedding_function = None  # Use ChromaDB default
        
        # Initialize Chroma client for Docker environment
        try:
            self.client = chromadb.HttpClient(
                host=self.chroma_host,
                port=self.chroma_port,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Test connection
            self.client.heartbeat()
            logger.info("Successfully connected to ChromaDB service")
            
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            # Fallback to local persistent client if service unavailable
            logger.info("Falling back to local persistent ChromaDB")
            persist_directory = Path("/app/data/chroma_db")
            persist_directory.mkdir(parents=True, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=str(persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        
        # Collection for storing both products and reviews (keeping GTE name for compatibility)
        self.collection_name = "electronics_products_reviews_gte"
        self.collection = None
        
    @weave.op()
    def create_collection(self) -> None:
        """Create or get the collection for storing documents."""
        try:
            # Try to get existing collection first (with or without embedding function)
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Retrieved existing collection: {self.collection_name}")
        except Exception:
            # Collection doesn't exist, create it with default embedding
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    @weave.op()
    def load_documents(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """Load documents from JSONL file."""
        documents = []
        
        # Handle both absolute and relative paths in Docker
        if not Path(jsonl_path).exists():
            # Try alternative paths for Docker environment
            alt_paths = [
                f"/app/{jsonl_path}",
                f"/app/data/processed/electronics_rag_documents.jsonl"
            ]
            
            for alt_path in alt_paths:
                if Path(alt_path).exists():
                    jsonl_path = alt_path
                    break
            else:
                logger.error(f"Could not find JSONL file at any of these paths: {[jsonl_path] + alt_paths}")
                return documents
        
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
    
    @weave.op()
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
    
    @weave.op()
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
    
    @weave.op()
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        if not self.collection:
            return {"error": "Collection not initialized"}
        
        try:
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
                "collection_name": self.collection_name,
                "chroma_host": self.chroma_host,
                "chroma_port": self.chroma_port
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    @weave.op()
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
    
    @weave.op()
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
    
    @weave.op()
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


@weave.op()
def setup_vector_database_docker(jsonl_path: str = "data/processed/electronics_rag_documents.jsonl") -> ElectronicsVectorDBDocker:
    """Setup and populate the Docker-compatible vector database."""
    logger.info("Setting up Docker-compatible vector database...")
    
    # Initialize database
    vector_db = ElectronicsVectorDBDocker()
    
    # Create collection
    vector_db.create_collection()
    
    # Check if collection is already populated
    stats = vector_db.get_collection_stats()
    if stats.get("total_documents", 0) > 0:
        logger.info(f"Collection already populated with {stats['total_documents']} documents")
        return vector_db
    
    # Load and ingest documents
    documents = vector_db.load_documents(jsonl_path)
    if documents:
        vector_db.ingest_documents(documents)
        
        # Show final stats
        final_stats = vector_db.get_collection_stats()
        logger.info(f"Vector database setup complete: {final_stats}")
    else:
        logger.warning("No documents loaded - database will be empty")
    
    return vector_db


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Setup database
    vector_db = setup_vector_database_docker()
    
    # Show stats
    stats = vector_db.get_collection_stats()
    print(json.dumps(stats, indent=2))