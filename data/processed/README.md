# Processed Data Documentation

This directory contains processed and optimized datasets derived from the Amazon Reviews 2023 - Electronics Category. All files are in JSONL (JSON Lines) format, with one JSON object per line.

## Dataset Overview

- **Source**: Amazon Reviews 2023 - Electronics Category
- **Processing Date**: 2025-06-26
- **Total Products**: 1,000 carefully selected electronics products
- **Total Reviews**: 20,000 customer reviews (20 reviews per product)
- **Date Range**: 2003-2023 (20 years of review data)

## File Structure

### 1. `electronics_top1000_products.jsonl`
**Purpose**: Core product metadata and specifications
**Records**: 1,000 product records
**Size**: ~5.4MB

#### Schema:
```json
{
  "main_category": "string",           // Primary product category
  "title": "string",                   // Product title
  "average_rating": float,             // Average customer rating (1-5)
  "rating_number": integer,            // Total number of ratings
  "features": ["string"],              // Product feature list
  "description": ["string"],           // Product descriptions
  "price": float,                      // Product price (USD)
  "images": [                          // Product image URLs
    {
      "thumb": "url",
      "large": "url", 
      "variant": "string",
      "hi_res": "url"
    }
  ],
  "videos": [                          // Product video content
    {
      "title": "string",
      "url": "url",
      "user_id": "string"
    }
  ],
  "store": "string",                   // Seller/store name
  "categories": ["string"],            // Hierarchical category path
  "details": {                         // Technical specifications
    "Brand": "string",
    "Connector Type": "string",
    // ... additional specs
  },
  "parent_asin": "string",             // Amazon Standard Identification Number
  "bought_together": null,             // Frequently bought together items
  "review_count": integer              // Number of reviews for this product
}
```

#### Key Features:
- Complete product metadata from Amazon
- Hierarchical category structure (up to 7 levels deep)
- Rich media content (images, videos)
- Technical specifications and details
- Pricing and rating information

---

### 2. `electronics_top1000_products_reviews.jsonl`
**Purpose**: Individual customer reviews for all products
**Records**: 20,000 review records
**Size**: ~8.9MB

#### Schema:
```json
{
  "asin": "string",                    // Product ASIN
  "parent_asin": "string",             // Parent product ASIN
  "rating": integer,                   // Review rating (1-5 stars)
  "title": "string",                   // Review title/headline
  "text": "string",                     // Full review text content
  "timestamp": integer,                // Unix timestamp (milliseconds)
  "verified_purchase": boolean,        // Amazon verified purchase flag
  "helpful_vote": integer              // Number of helpful votes
}
```

#### Key Features:
- 20 reviews sampled per product for balanced representation
- Verified purchase status for authenticity
- Temporal data spanning 2003-2023
- Community engagement metrics (helpful votes)
- Complete review text for sentiment analysis

---

### 3. `electronics_rag_documents.jsonl`
**Purpose**: RAG-optimized documents for retrieval systems
**Records**: 2,000 records (1,000 products + 1,000 review summaries)
**Size**: ~5.9MB

#### Product Records Schema:
```json
{
  "id": "product_{asin}",              // Unique document ID
  "type": "product",                   // Document type
  "parent_asin": "string",             // Product ASIN
  "title": "string",                   // Product title
  "description": "string",             // Product description
  "features": "string",                // Concatenated features
  "price": float,                      // Product price
  "average_rating": float,             // Average rating
  "rating_number": integer,            // Total ratings
  "review_count": integer,             // Review count
  "store": "string",                   // Store name
  "categories": ["string"],            // Category hierarchy
  "details": {},                       // Product specifications
  "content": "string"                  // Searchable content summary
}
```

#### Review Summary Records Schema:
```json
{
  "id": "reviews_{asin}",              // Unique document ID
  "type": "review_summary",            // Document type
  "parent_asin": "string",             // Product ASIN
  "product_title": "string",          // Associated product title
  "total_reviews": integer,            // Number of reviews summarized
  "positive_reviews": integer,         // Count of positive reviews
  "negative_reviews": integer,         // Count of negative reviews
  "content": "string"                  // Aggregated review insights
}
```

#### Key Features:
- **Paired Structure**: Each product has exactly one review summary
- **Optimized for RAG**: Condensed content for efficient retrieval
- **Linked Records**: Connected via `parent_asin` field
- **Searchable Content**: Pre-processed text for vector embeddings
- **Sentiment Analysis**: Positive/negative review breakdown

---

### 4. `dataset_summary.json`
**Purpose**: Dataset statistics and metadata
**Records**: 1 summary record
**Size**: ~668B

#### Schema:
```json
{
  "dataset_info": {
    "source": "string",                // Data source information
    "citation": "string",             // Academic citation
    "processing_date": "string",      // ISO timestamp
    "selection_criteria": {           // Processing parameters
      "target_products": integer,
      "min_reviews_per_product": integer,
      "max_reviews_per_product": integer
    }
  },
  "statistics": {
    "total_products": integer,        // Product count
    "total_reviews": integer,         // Review count
    "products_with_metadata": integer,
    "products_with_reviews": integer,
    "review_count_stats": {           // Original review statistics
      "min": integer,
      "max": integer,
      "mean": float,
      "median": float
    }
  }
}
```

#### Key Features:
- Complete processing metadata
- Data quality metrics
- Selection criteria documentation
- Statistical summaries

## Data Relationships

```
electronics_top1000_products.jsonl (1,000 products)
         │
         ├── Linked by parent_asin
         │
electronics_top1000_products_reviews.jsonl (20,000 reviews)
         │
         └── Summarized into
         
electronics_rag_documents.jsonl (2,000 RAG docs)
         ├── 1,000 product records
         └── 1,000 review summary records
```

## Usage Examples

### Loading Product Data
```python
import json
import pandas as pd

# Load products
products = []
with open('electronics_top1000_products.jsonl', 'r') as f:
    for line in f:
        products.append(json.loads(line.strip()))

df_products = pd.DataFrame(products)
print(f"Loaded {len(df_products)} products")
```

### Loading Reviews
```python
# Load reviews
reviews = []
with open('electronics_top1000_products_reviews.jsonl', 'r') as f:
    for line in f:
        reviews.append(json.loads(line.strip()))

df_reviews = pd.DataFrame(reviews)
print(f"Loaded {len(df_reviews)} reviews")
```

### Loading RAG Documents
```python
# Load RAG documents
rag_docs = []
with open('electronics_rag_documents.jsonl', 'r') as f:
    for line in f:
        rag_docs.append(json.loads(line.strip()))

# Separate by type
products = [doc for doc in rag_docs if doc['type'] == 'product']
summaries = [doc for doc in rag_docs if doc['type'] == 'review_summary']

print(f"Loaded {len(products)} product docs and {len(summaries)} review summaries")
```

## Data Quality Notes

1. **Completeness**: 100% completeness for ratings, stores, categories; 77.2% for pricing
2. **Consistency**: All products have exactly 20 reviews in the reviews file
3. **Validation**: All ASINs are validated and cross-referenced
4. **Deduplication**: No duplicate products or reviews
5. **Temporal Coverage**: Reviews span 20 years (2003-2023)

## Processing Pipeline

The data was processed through the following steps:

1. **Raw Data Ingestion**: Amazon Reviews 2023 dataset (25GB+)
2. **Product Selection**: Top 1000 products by review volume and quality
3. **Review Sampling**: 20 representative reviews per product
4. **Data Cleaning**: Missing value handling, validation
5. **RAG Optimization**: Content summarization and formatting
6. **Quality Assurance**: Statistical validation and integrity checks

## File Sizes and Performance

| File | Records | Size | Load Time* |
|------|---------|------|------------|
| products.jsonl | 1,000 | 5.4MB | ~0.5s |
| reviews.jsonl | 20,000 | 8.9MB | ~1.2s |
| rag_documents.jsonl | 2,000 | 5.9MB | ~0.8s |
| dataset_summary.json | 1 | 668B | <0.1s |

*Approximate load times on standard hardware

## Citation

If you use this dataset, please cite:

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
``` 