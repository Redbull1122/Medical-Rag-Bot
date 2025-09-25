#!/bin/bash

# Stop any running containers
echo "Stopping any running containers..."
docker stop $(docker ps -q) 2>/dev/null || true

# Build and run the medical RAG bot with environment variables
echo "Building and starting Medical RAG Bot..."
docker build -t medical-rag-bot .
docker run --rm -p 8501:8501 --env-file .env medical-rag-bot