#!/bin/sh

# Exit immediately if any command fails
set -e

echo "===================================================="
echo "🎯 Starting Local MLOps Development Stack..."
echo "===================================================="

cleanup() {
    echo ""
    echo "===================================================="
    echo "🛑 Shutting down local MLOps stack..."
    echo "===================================================="
    
    echo "🔻 Tearing down Docker containers..."
    docker compose -f ./docker-compose-local.yaml down
    
    echo "✨ Clean up complete. Have a great day!"
}
# Trap Ctrl+C (SIGINT) and exit signals to run the cleanup function
trap cleanup EXIT SIGINT SIGTERM

# Give MLflow 3 seconds to spin up completely
sleep 3

# 2. Spin up the Local Docker Compose stack
echo "🐳 Building and spinning up Docker containers..."
echo "👉 Web app will be live at: http://localhost:8501"
echo "👉 API docs will be live at: http://localhost:8000/docs"
echo "----------------------------------------------------"

docker compose -f ./docker-compose-local.yaml up --build

