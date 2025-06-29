run-streamlit:
	streamlit run src/chatbot-ui/streamlit_app.py

build-docker-streamlit:
	docker-compose build streamlit-app

run-docker-streamlit:
	docker-compose up -d

stop-docker-streamlit:
	docker-compose down

logs-docker-streamlit:
	docker-compose logs -f

restart-docker-streamlit:
	docker-compose restart

# Legacy single container commands (deprecated)
build-docker-streamlit-legacy:
	docker build -t streamlit-app:latest .

run-docker-streamlit-legacy:
	docker run -v "${PWD}/.env:/app/.env" -p 8501:8501 streamlit-app:latest

clean-notebook-outputs:
	jupyter nbconvert --clear-output --inplace notebooks/*.ipynb

# Docker cleanup commands
clean-docker:
	docker-compose down -v
	docker system prune -f

# Initialize ChromaDB with data
init-docker-chroma:
	docker-compose up -d chromadb
	docker-compose exec streamlit-app python src/rag/vector_db_docker.py