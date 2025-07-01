#!/bin/bash

# Docker entrypoint script for AI Product Assistant
# Initializes ChromaDB and starts Streamlit app

set -e

echo "🚀 Starting AI Product Assistant..."

# Wait for ChromaDB service to be ready
echo "⏳ Waiting for ChromaDB service..."
timeout=60
while ! curl -s http://chromadb:8000/api/v1/heartbeat > /dev/null 2>&1; do
    sleep 2
    timeout=$((timeout - 2))
    if [ $timeout -le 0 ]; then
        echo "❌ ChromaDB service not ready after 60 seconds"
        echo "🔄 Continuing with local fallback..."
        break
    fi
done

if curl -s http://chromadb:8000/api/v1/heartbeat > /dev/null 2>&1; then
    echo "✅ ChromaDB service is ready"
    
    # Initialize ChromaDB with data if needed
    echo "🔍 Checking ChromaDB initialization..."
    
    # Check if collection already exists to avoid re-downloading models
    python3 -c "
import os
import sys
sys.path.append('src')
try:
    from rag.vector_db_docker import ElectronicsVectorDBDocker
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Try to connect and check for existing collection
    db = ElectronicsVectorDBDocker()
    db.create_collection()
    stats = db.get_collection_stats()
    
    if stats.get('total_documents', 0) == 0:
        print('📦 Initializing vector database with documents...')
        from rag.vector_db_docker import setup_vector_database_docker
        setup_vector_database_docker()
        print('✅ Vector database initialized successfully')
    else:
        print(f'✅ Found existing collection with {stats[\"total_documents\"]} documents')
        
except Exception as e:
    print(f'⚠️  Database initialization failed: {e}')
    print('🔄 Continuing without pre-populated database...')
    import traceback
    traceback.print_exc()
"
    echo "✅ ChromaDB initialization complete"
else
    echo "⚠️  ChromaDB service unavailable, using local storage"
fi

# Start Streamlit app
echo "🎯 Starting Streamlit application..."
exec streamlit run src/chatbot-ui/streamlit_app.py --server.address=0.0.0.0 "$@"