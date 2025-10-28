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
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
import warnings
import io

# Matplotlib backend for server-side image rendering
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import speech_recognition as sr

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
# Enable template auto-reload during development
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

# Global variables for model
model = None
device = None
args = None
recognizer = sr.Recognizer()

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

def process_audio_file(input_path, output_dir, output_prefix=None):
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
        # If an explicit prefix is provided (e.g., timestamp), use it; otherwise fall back to input base name
        base_name = output_prefix if output_prefix else os.path.splitext(os.path.basename(input_path))[0]
        
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

def transcribe_audio_vi(audio_path: str) -> str:
    """Transcribe Vietnamese speech from a WAV file using Google Web Speech API."""
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language="vi-VN")
        return text
    except Exception as e:
        print(f"Transcription error ({audio_path}): {e}")
        return ""

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
            raw_name = secure_filename(file.filename)
            # Build a timestamp to ensure uniqueness across uploads
            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            base = os.path.splitext(raw_name)[0] if raw_name else 'audio'
            filename = f"{base}_{ts}.wav"
            
            # Save uploaded file
            input_path = os.path.join("outputs/try/input", filename)
            file.save(input_path)
            
            print(f"File saved to: {input_path}")
            
            # Process the audio with the same base+timestamp prefix
            output_prefix = f"{base}_{ts}"
            output_files = process_audio_file(input_path, "outputs/try/output", output_prefix=output_prefix)
            # Transcribe each output file (Vietnamese)
            transcripts = []
            for file_name in output_files:
                abs_out = os.path.join("outputs/try/output", file_name)
                transcripts.append(transcribe_audio_vi(abs_out))
            
            # Prepare response
            response_data = {
                'success': True,
                'files': output_files,
                'file_paths': output_files,
                'temp_dir': 'outputs/try/output',
                'input_file_path': f"outputs/try/input/{filename}",
                'input_file_name': filename,
                'transcripts': transcripts
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

@app.route('/waveform/<path:filepath>')
def waveform_image(filepath):
    """Render waveform image (PNG) for the given audio file path.
    Only allow files inside known output directories.
    """
    try:
        # Build absolute paths and validate
        base_dir = os.path.abspath(os.getcwd())
        allowed_roots = [
            os.path.abspath(os.path.join(base_dir, 'outputs/try/input')),
            os.path.abspath(os.path.join(base_dir, 'outputs/try/output')),
            os.path.abspath(os.path.join(base_dir, 'outputs/MossFormer2_SS_8K')),
        ]
        abs_path = os.path.abspath(os.path.join(base_dir, filepath))
        if not any(abs_path.startswith(root + os.sep) or abs_path == root for root in allowed_roots):
            return jsonify({'error': 'Access denied'}), 403

        # Load audio (mono)
        audio, sr = librosa.load(abs_path, sr=None, mono=True)

        # Create figure
        fig = plt.figure(figsize=(8, 1.6), dpi=150)
        ax = fig.add_subplot(111)
        ax.plot(np.linspace(0, len(audio)/sr, num=len(audio)), audio, color='#2b6cb0', linewidth=0.8)
        ax.set_xlim(0, len(audio)/sr)
        ax.set_ylim(-1.0, 1.0)
        ax.axis('off')
        fig.tight_layout(pad=0)

        # Save to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', transparent=True)
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        print(f"Waveform error: {e}")
        return jsonify({'error': str(e)}), 500

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
    
    # Run Flask app (enable auto-reload if FLASK_DEBUG=1)
    debug_flag = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=5000, debug=debug_flag, use_reloader=debug_flag)
