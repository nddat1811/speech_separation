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
pip install flask==3.1.2 werkzeug==3.1.3 soundfile librosa numpy scipy pyyaml matplotlib SpeechRecognition

echo ""
echo "Step 5: Installing PyTorch CPU version..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "Step 6: Starting CPU-only web app..."
echo "Please wait for the model to load (this may take a minute)..."
echo ""

# Function to check if port is open
check_port() {
    if command -v nc > /dev/null; then
        nc -z localhost 5000 > /dev/null 2>&1
    elif command -v timeout > /dev/null && command -v bash > /dev/null; then
        timeout 1 bash -c "echo > /dev/tcp/localhost/5000" > /dev/null 2>&1
    else
        # Fallback: just check if curl can connect
        curl -s http://localhost:5000/health > /dev/null 2>&1
    fi
}

# Start the Flask app in background
python app.py &
APP_PID=$!

# Wait for server to be ready (check every 2 seconds, max 60 seconds)
echo "Waiting for server to start..."
MAX_WAIT=360
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if check_port; then
        echo "Server is ready!"
        sleep 2  # Give it a moment more to be fully ready
        break
    fi
    if ! ps -p $APP_PID > /dev/null 2>&1; then
        echo ""
        echo "ERROR: The server failed to start. Check the error messages above."
        echo "Common issues:"
        echo "  - Missing checkpoint files in checkpoints/Libri2Mix_min_adam/MossFormer2_SS_8K_clean/"
        echo "  - Missing dependencies (run: pip install -r requirements.txt)"
        exit 1
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    echo -n "."
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo ""
    echo "WARNING: Server did not become ready within $MAX_WAIT seconds."
    echo "It may still be loading the model. You can try opening http://localhost:5000 manually."
else
    echo ""
    echo "Opening http://localhost:5000 in your browser..."
    if command -v xdg-open > /dev/null; then
        xdg-open http://localhost:5000 &
    elif command -v open > /dev/null; then
        open http://localhost:5000 &
    fi
fi

echo ""
echo "Server is running (PID: $APP_PID). Press Ctrl+C to stop."
echo ""

# Wait for the Flask app process
wait $APP_PID
