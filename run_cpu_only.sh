#!/bin/bash

echo "========================================"
echo "Audio Source Separation - CPU Only"
echo "========================================"

echo ""
echo "Step 1: Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    echo "Please make sure Python3 is installed"
    exit 1
fi

echo ""
echo "Step 2: Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Step 3: Upgrading pip..."
python -m pip install --upgrade pip

echo ""
echo "Step 4: Installing basic packages..."
pip install flask==3.1.2 werkzeug==3.1.3 soundfile librosa numpy scipy pyyaml

echo ""
echo "Step 5: Installing PyTorch CPU version..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "Step 6: Starting CPU-only web app..."
echo "Opening http://localhost:5000 in your browser..."
if command -v xdg-open > /dev/null; then
    xdg-open http://localhost:5000
elif command -v open > /dev/null; then
    open http://localhost:5000
fi
python app.py
