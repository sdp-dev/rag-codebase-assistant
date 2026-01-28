#!/bin/bash
# Pull the LLM model into Ollama container

echo "Waiting for Ollama to start..."
sleep 5

echo "Pulling llama3 model (this may take a while)..."
docker exec ollama ollama pull llama3

echo "Done! Model is ready."
echo "Access the app at http://localhost:8501"
