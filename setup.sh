#!/bin/bash
# Options Calculator Pro - Setup Script for Mac/Linux

echo "=== Professional Options Calculator Pro Setup ==="

# Check Python version
python_version=$(python3 --version 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "✓ Python found: $python_version"
else
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Create/activate virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required dependencies
echo "Installing core dependencies..."
pip install PySide6 yfinance pandas numpy scipy matplotlib scikit-learn requests

# Install additional useful packages
echo "Installing additional packages..."
pip install python-dotenv plotly seaborn

# Create necessary directories
echo "Creating project directories..."
mkdir -p logs config data exports

# Create requirements.txt for future reference
echo "Creating requirements.txt..."
pip freeze > requirements.txt

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To start the Options Calculator Pro:"
echo "1. Activate virtual environment: source .venv/bin/activate"
echo "2. Run the application: python main.py"
echo ""
echo "Or use the quick start script: ./run.sh"