#!/bin/bash

# Exit on any error
set -e

echo "Starting Stock Market Pattern Clustering API..."

# Check if data file exists
if [ ! -f "SPY Chart 2025-08-22-09-36.csv" ]; then
    echo "Error: Data file 'SPY Chart 2025-08-22-09-36.csv' not found!"
    echo "Please ensure the CSV file is in the container."
    exit 1
fi

echo "Data file found. Starting API server..."

# Start the FastAPI application
exec uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    --access-log
