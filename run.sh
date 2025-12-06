#!/bin/bash
# Quick run script for Options Calculator Pro

echo "Starting Options Calculator Pro..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Running setup first..."
    ./setup.sh
fi

# Activate virtual environment
source .venv/bin/activate

# Run the application
python main.py