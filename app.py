#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import tempfile
import shutil
import subprocess
import yaml
import soundfile as sf
import librosa
import numpy as np
import time
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import inference modules
from utils.misc import reload_for_eval
from utils.decode import decode_one_audio
from networks import network_wrapper

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for model
model = None
device = None
args = None

def load_model():
    """Load the MossFormer2 model for inference"""
    global model, device, args
    
    print("Loading MossFormer2 model...")
    
    try:
        # Load config
        config_path = "config/inference/MossFormer2_SS_8K.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Create args object
        class Args:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    setattr(self, key, value)
        
        args = Args(config)
        args.use_cuda = 0  # Force CPU mode
        args.checkpoint_dir = "checkpoints/Libri2Mix_min_adam/MossFormer2_SS_8K_clean"
        
        # Set device
        device = torch.device('cpu')
        print(f"Using device: {device}")
        
        # Create model
        print("Creating model...")
        model = network_wrapper(args).ss_network
        model.to(device)
        
        # Load checkpoint
        print("Loading checkpoint...")
        reload_for_eval(model, args.checkpoint_dir, args.use_cuda)
        model.eval()
        
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

def ensure_directories():
    """Create necessary directories"""
    os.makedirs("outputs/try/input", exist_ok=True)
    os.makedirs("outputs/try/output", exist_ok=True)

def process_audio_file(input_path, output_dir):
    """Process audio file using MossFormer2"""
    global model, device, args
    
    try:
        print(f"Processing audio file: {input_path}")
        
        # Load audio
        audio, sr = librosa.load(input_path, sr=args.sampling_rate)
        
        # Convert to numpy array and add batch dimension
        audio_np = audio.astype(np.float32)
        audio_tensor = torch.FloatTensor(audio_np).unsqueeze(0)
        
        # Process with model
        with torch.no_grad():
            output_audios = decode_one_audio(model, device, audio_tensor, args)
        
        # Save separated audio files
        output_files = []
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        for spk in range(args.num_spks):
            output_audio = output_audios[spk]
            if isinstance(output_audio, torch.Tensor):
                output_audio = output_audio.cpu().numpy()
            
            output_filename = f"{base_name}_s{spk+1}.wav"
            output_path = os.path.join(output_dir, output_filename)
            
            sf.write(output_path, output_audio, args.sampling_rate)
            output_files.append(output_filename)
            print(f"Saved: {output_path}")
        
        return output_files
        
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        raise e

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if file:
            # Secure filename
            filename = secure_filename(file.filename)
            if not filename:
                filename = f"audio_{int(time.time())}.wav"
            
            # Ensure .wav extension
            if not filename.lower().endswith('.wav'):
                filename = os.path.splitext(filename)[0] + '.wav'
            
            # Save uploaded file
            input_path = os.path.join("outputs/try/input", filename)
            file.save(input_path)
            
            print(f"File saved to: {input_path}")
            
            # Process the audio
            output_files = process_audio_file(input_path, "outputs/try/output")
            
            # Prepare response
            response_data = {
                'success': True,
                'files': output_files,
                'file_paths': output_files,
                'temp_dir': 'outputs/try/output'
            }
            
            return jsonify(response_data)
            
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download/<path:filename>')
def download_file(filename):
    """Serve audio files for download/playback"""
    try:
        # Try different possible paths
        possible_paths = [
            os.path.join("outputs/try/output", filename),
            os.path.join("outputs/try/input", filename),
            os.path.join("outputs/MossFormer2_SS_8K", filename),
            filename  # Direct path
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return send_file(path, as_attachment=True)
        
        return jsonify({'error': 'File not found'}), 404
        
    except Exception as e:
        print(f"Download error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/demo_files')
def demo_files():
    """Return demo files information"""
    try:
        demo_files = []
        
        # Check for demo files in outputs directory
        demo_dir = "outputs/MossFormer2_SS_8K"
        if os.path.exists(demo_dir):
            for file in os.listdir(demo_dir):
                if file.endswith('.wav'):
                    demo_files.append({
                        'name': file,
                        'before': file,  # Assuming original mixed file
                        'after': file    # Assuming separated file
                    })
        
        return jsonify({'demo_files': demo_files})
        
    except Exception as e:
        print(f"Demo files error: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    print("=" * 50)
    print("Audio Source Separation Web App")
    print("=" * 50)
    
    # Ensure directories exist
    ensure_directories()
    
    # Load model
    try:
        load_model()
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        print("Please make sure the checkpoint files are available")
        sys.exit(1)
    
    print("\nStarting Flask server...")
    print("Open http://localhost:5000 in your browser")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
