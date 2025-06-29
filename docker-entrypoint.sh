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
    python src/rag/vector_db_docker.py
    echo "✅ ChromaDB initialization complete"
else
    echo "⚠️  ChromaDB service unavailable, using local storage"
fi

# Start Streamlit app
echo "🎯 Starting Streamlit application..."
exec streamlit run src/chatbot-ui/streamlit_app.py --server.address=0.0.0.0 "$@"