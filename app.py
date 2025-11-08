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
from utils.misc import reload_for_eval, load_checkpoint
from utils.decode import decode_one_audio
from networks import network_wrapper

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
# Enable template auto-reload during development
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

# Global variables for models - dictionary to store multiple models
models = {}  # {model_type: {'model': model_obj, 'device': device, 'args': args}}
recognizer = sr.Recognizer()

def get_model_info(checkpoint_dir):
    """Get epoch and SIDR information from checkpoint directory
    
    Returns:
        dict with 'epoch' and 'sidr' (SI-SDR in dB)
    """
    info = {'epoch': None, 'sidr': None}
    
    try:
        # Get epoch from checkpoint file
        best_checkpoint_file = os.path.join(checkpoint_dir, 'last_best_checkpoint')
        checkpoint_file = os.path.join(checkpoint_dir, 'last_checkpoint')
        
        checkpoint_path = None
        if os.path.isfile(best_checkpoint_file):
            with open(best_checkpoint_file, 'r') as f:
                checkpoint_name = f.readline().strip()
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        elif os.path.isfile(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint_name = f.readline().strip()
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = load_checkpoint(checkpoint_path, use_cuda=0)
            if 'epoch' in checkpoint:
                info['epoch'] = checkpoint['epoch']
        
        # Get SIDR from train.log (best validation loss = -SI-SDR)
        log_file = os.path.join(checkpoint_dir, 'train.log')
        if os.path.exists(log_file):
            best_val_loss = None
            best_epoch = None
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if 'Val Summary' in line:
                        # Parse: "Val Summary | End of Epoch X | Time Y | Val Loss Z"
                        parts = line.split('|')
                        if len(parts) >= 4:
                            try:
                                epoch_part = parts[1].strip()
                                epoch = int(epoch_part.split()[-1])
                                loss_part = parts[3].strip()
                                val_loss = float(loss_part.split()[-1])
                                
                                # Loss is negative SI-SDR, so SI-SDR = -loss
                                # We want the best (highest) SI-SDR = lowest (most negative) loss
                                if best_val_loss is None or val_loss < best_val_loss:
                                    best_val_loss = val_loss
                                    best_epoch = epoch
                            except (ValueError, IndexError):
                                continue
            
            if best_val_loss is not None:
                # Convert loss to SIDR: loss = -SI-SDR, so SI-SDR = -loss
                info['sidr'] = -best_val_loss
                # If we found a better epoch from log, use it
                if best_epoch is not None and info['epoch'] is None:
                    info['epoch'] = best_epoch
    
    except Exception as e:
        print(f"Warning: Could not get model info: {str(e)}")
    
    return info

def load_model(model_type='clean'):
    """Load the MossFormer2 model for inference
    
    Args:
        model_type: 'clean', 'finetune', or 'noise'
    """
    global models
    
    print(f"\n{'='*50}")
    print(f"Loading MossFormer2 model: {model_type}")
    print(f"{'='*50}")
    
    # Model configurations
    model_configs = {
        'clean': {
            'config_path': 'config/inference/MossFormer2_SS_8K.yaml',
            'checkpoint_dir': 'checkpoints/Libri2Mix_min_adam/MossFormer2_SS_8K_clean'
        },
        'finetune': {
            'config_path': 'config/inference/MossFormer2_SS_8K.yaml',
            'checkpoint_dir': 'checkpoints/VietVivoMix/MossFormer2_SS_8K_clean_finetune'
        },
        'noise': {
            'config_path': 'config/inference/MossFormer2_SS_8K.yaml',
            'checkpoint_dir': 'checkpoints/Libri2Mix_min_adam/MossFormer2_SS_8k_noise'
        }
    }
    
    if model_type not in model_configs:
        raise ValueError(f"Unknown model_type: {model_type}. Supported: {list(model_configs.keys())}")
    
    config_info = model_configs[model_type]
    config_path = config_info['config_path']
    checkpoint_dir = config_info['checkpoint_dir']
    
    # Check if checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        print(f"WARNING: Checkpoint directory not found: {checkpoint_dir}")
        print(f"Skipping model: {model_type}")
        return False
    
    try:
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Create args object
        class Args:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    setattr(self, key, value)
        
        args = Args(config)
        args.use_cuda = 0  # Force CPU mode
        args.checkpoint_dir = checkpoint_dir
        
        # Set device
        device = torch.device('cpu')
        print(f"Using device: {device}")
        
        # Create model
        print("Creating model...")
        model = network_wrapper(args).ss_network
        model.to(device)
        
        # Load checkpoint
        print(f"Loading checkpoint from: {checkpoint_dir}")
        reload_for_eval(model, args.checkpoint_dir, args.use_cuda)
        model.eval()
        
        # Get model info (epoch and SIDR)
        model_info = get_model_info(checkpoint_dir)
        
        # Store model in dictionary
        models[model_type] = {
            'model': model,
            'device': device,
            'args': args,
            'epoch': model_info['epoch'],
            'sidr': model_info['sidr']
        }
        
        info_str = f"Model '{model_type}' loaded successfully!"
        if model_info['epoch'] is not None:
            info_str += f" (Epoch: {model_info['epoch']}"
        if model_info['sidr'] is not None:
            info_str += f", SIDR: {model_info['sidr']:.2f} dB"
        if model_info['epoch'] is not None or model_info['sidr'] is not None:
            info_str += ")"
        print(info_str)
        return True
        
    except Exception as e:
        print(f"Error loading model '{model_type}': {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def load_all_models():
    """Load all available models"""
    print("\n" + "="*50)
    print("Loading all models...")
    print("="*50)
    
    model_types = ['clean', 'finetune', 'noise']
    loaded_count = 0
    
    for model_type in model_types:
        try:
            if load_model(model_type):
                loaded_count += 1
        except Exception as e:
            print(f"Failed to load model '{model_type}': {str(e)}")
    
    print(f"\n{'='*50}")
    print(f"Loaded {loaded_count}/{len(model_types)} models")
    print(f"Available models: {list(models.keys())}")
    print(f"{'='*50}\n")
    
    if loaded_count == 0:
        raise RuntimeError("No models were loaded successfully!")

def ensure_directories():
    """Create necessary directories for each model type"""
    model_types = ['clean', 'noise', 'finetune']
    for model_type in model_types:
        os.makedirs(f"outputs/{model_type}/input", exist_ok=True)
        os.makedirs(f"outputs/{model_type}/output", exist_ok=True)

def process_audio_file(input_path, output_dir, output_prefix=None, model_type='clean'):
    """Process audio file using MossFormer2
    
    Args:
        input_path: Path to input audio file
        output_dir: Directory to save output files
        output_prefix: Prefix for output filenames
        model_type: 'clean', 'finetune', or 'noise'
    """
    global models
    
    if model_type not in models:
        raise ValueError(f"Model '{model_type}' not loaded. Available models: {list(models.keys())}")
    
    model_info = models[model_type]
    model = model_info['model']
    device = model_info['device']
    args = model_info['args']
    
    try:
        print(f"Processing audio file: {input_path} with model: {model_type}")
        
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
            
            # Add "FT" suffix for finetune model
            if model_type == 'finetune':
                output_filename = f"{base_name}_{model_type}_s{spk+1}_FT.wav"
            else:
                output_filename = f"{base_name}_{model_type}_s{spk+1}.wav"
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

def detect_vietnamese(text: str) -> bool:
    """Detect if text contains Vietnamese characters."""
    if not text:
        return False
    # Check for Vietnamese-specific characters
    vietnamese_chars = set('àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ')
    return any(char.lower() in vietnamese_chars for char in text) or any(ord(char) >= 0x1E00 and ord(char) <= 0x1EFF for char in text)

def detect_gender_from_transcript(transcript: str) -> str:
    """Detect gender from Vietnamese transcript. Returns 'Nam', 'Nữ', or 'Unknown'."""
    if not transcript:
        return 'Unknown'
    
    transcript_lower = transcript.lower()
    
    # Male indicators
    male_indicators = ['ông', 'anh', 'chú', 'bác', 'nam', 'con trai', 'cậu', 'em trai', 'ông ấy', 'anh ấy']
    # Female indicators  
    female_indicators = ['bà', 'chị', 'cô', 'dì', 'nữ', 'con gái', 'cô ấy', 'chị ấy', 'bà ấy']
    
    male_count = sum(1 for word in male_indicators if word in transcript_lower)
    female_count = sum(1 for word in female_indicators if word in transcript_lower)
    
    if male_count > female_count:
        return 'Nam'
    elif female_count > male_count:
        return 'Nữ'
    else:
        return 'Unknown'

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
        
        # Get model_type from form data, default to 'clean'
        model_type = request.form.get('model_type', 'clean')
        
        # Validate model_type
        if model_type not in models:
            return jsonify({
                'success': False, 
                'error': f"Model '{model_type}' not available. Available models: {list(models.keys())}"
            })
        
        if file:
            # Secure filename
            raw_name = secure_filename(file.filename)
            # Build a timestamp to ensure uniqueness across uploads
            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            base = os.path.splitext(raw_name)[0] if raw_name else 'audio'
            filename = f"{base}_{ts}.wav"
            
            # Save uploaded file to model-specific directory
            input_dir = f"outputs/{model_type}/input"
            os.makedirs(input_dir, exist_ok=True)
            input_path = os.path.join(input_dir, filename)
            file.save(input_path)
            
            print(f"File saved to: {input_path}")
            print(f"Using model: {model_type}")
            
            # Process the audio with the same base+timestamp prefix and selected model
            output_dir = f"outputs/{model_type}/output"
            os.makedirs(output_dir, exist_ok=True)
            output_prefix = f"{base}_{ts}"
            output_files = process_audio_file(
                input_path, 
                output_dir, 
                output_prefix=output_prefix,
                model_type=model_type
            )
            # Transcribe each output file (Vietnamese)
            transcripts = []
            for file_name in output_files:
                abs_out = os.path.join(output_dir, file_name)
                transcripts.append(transcribe_audio_vi(abs_out))
            
            # Prepare response
            response_data = {
                'success': True,
                'files': output_files,
                'file_paths': output_files,
                'temp_dir': output_dir,
                'input_file_path': f"{input_dir}/{filename}",
                'input_file_name': filename,
                'transcripts': transcripts,
                'model_type': model_type
            }
            
            return jsonify(response_data)
            
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download/<path:filename>')
def download_file(filename):
    """Serve audio files for download/playback"""
    try:
        # Try different possible paths - check model-specific directories first
        model_types = ['clean', 'noise', 'finetune']
        possible_paths = []
        
        # Add model-specific paths
        for model_type in model_types:
            possible_paths.append(os.path.join(f"outputs/{model_type}/output", filename))
            possible_paths.append(os.path.join(f"outputs/{model_type}/input", filename))
        
        # Add legacy paths for backward compatibility
        possible_paths.extend([
            os.path.join("outputs/try/output", filename),
            os.path.join("outputs/try/input", filename),
            os.path.join("outputs/MossFormer2_SS_8K", filename),
            filename  # Direct path
        ])
        
        for path in possible_paths:
            if os.path.exists(path):
                return send_file(path, as_attachment=True)
        
        return jsonify({'error': 'File not found'}), 404
        
    except Exception as e:
        print(f"Download error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/demo_files')
def demo_files():
    """Return demo files - one most recent file for each gender combination type
    Query parameter: model_type (optional) - if provided, only return files from that model
    """
    try:
        # Get model_type from query parameter (if provided)
        requested_model_type = request.args.get('model_type', None)
        
        all_files = []
        # If model_type is specified, only process that model; otherwise process all
        if requested_model_type and requested_model_type in ['clean', 'noise', 'finetune']:
            model_types = [requested_model_type]
        else:
            model_types = ['clean', 'noise', 'finetune']
        
        # Collect all output files with their metadata
        for model_type in model_types:
            output_dir = f"outputs/{model_type}/output"
            input_dir = f"outputs/{model_type}/input"
            
            if not os.path.exists(output_dir):
                continue
            
            # Get all speaker files (s1, s2)
            for filename in os.listdir(output_dir):
                if filename.endswith('.wav') and ('_s1.wav' in filename or '_s2.wav' in filename or '_s1_FT.wav' in filename or '_s2_FT.wav' in filename):
                    # Extract base name (remove _model_type_s1.wav, _model_type_s2.wav, or _model_type_s1_FT.wav, _model_type_s2_FT.wav)
                    # Format: base_timestamp_model_type_s1.wav -> base_timestamp.wav
                    # Format: base_timestamp_model_type_s1_FT.wav -> base_timestamp.wav (for finetune)
                    base_match = filename
                    for mt in model_types:
                        # Handle finetune files with _FT suffix
                        if mt == 'finetune':
                            if f'_{mt}_s1_FT.wav' in filename:
                                base_match = filename.replace(f'_{mt}_s1_FT.wav', '.wav')
                                break
                            elif f'_{mt}_s2_FT.wav' in filename:
                                base_match = filename.replace(f'_{mt}_s2_FT.wav', '.wav')
                                break
                            # Also handle old finetune files without _FT (backward compatibility)
                            elif f'_{mt}_s1.wav' in filename:
                                base_match = filename.replace(f'_{mt}_s1.wav', '.wav')
                                break
                            elif f'_{mt}_s2.wav' in filename:
                                base_match = filename.replace(f'_{mt}_s2.wav', '.wav')
                                break
                        else:
                            # Handle other model types (clean, noise)
                            if f'_{mt}_s1.wav' in filename:
                                base_match = filename.replace(f'_{mt}_s1.wav', '.wav')
                                break
                            elif f'_{mt}_s2.wav' in filename:
                                base_match = filename.replace(f'_{mt}_s2.wav', '.wav')
                                break
                    
                    # Find corresponding input file
                    input_file = base_match if base_match != filename else None
                    
                    # Verify input file exists
                    if input_file and os.path.exists(input_dir):
                        input_path = os.path.join(input_dir, input_file)
                        if not os.path.exists(input_path):
                            input_file = None
                    else:
                        input_file = None
                    
                    file_path = os.path.join(output_dir, filename)
                    file_time = os.path.getmtime(file_path)
                    
                    all_files.append({
                        'filename': filename,
                        'file_path': file_path,
                        'model_type': model_type,
                        'input_file': input_file,
                        'input_dir': input_dir,
                        'output_dir': output_dir,
                        'timestamp': file_time,
                        'base_name': base_match
                    })
        
        # Group files by base_name and model_type first
        grouped_files = {}
        for file_info in all_files:
            key = f"{file_info['base_name']}_{file_info['model_type']}"
            if key not in grouped_files:
                grouped_files[key] = []
            grouped_files[key].append(file_info)
        
        # Process each group to get gender labels and create entries
        all_entries = []
        for key, files in grouped_files.items():
            if len(files) < 2:  # Need at least s1 and s2
                continue
            
            # Sort files to ensure s1 comes before s2 (handle both regular and _FT suffix)
            files_sorted = sorted(files, key=lambda x: (
                '_s2_FT.wav' in x['filename'] or '_s2.wav' in x['filename'], 
                x['timestamp']
            ))
            
            # Try to detect gender from filename first (faster)
            base_name = files[0]['base_name'].lower()
            # Also check original filename for patterns
            filename_lower = files[0].get('filename', '').lower()
            combined_name = f"{base_name} {filename_lower}".lower()
            
            genders_from_filename = []
            is_vietnamese_from_filename = False
            
            # Check filename for gender patterns (check both base_name and original filename)
            if 'nam_nam' in combined_name or 'nam-nam' in combined_name:
                genders_from_filename = ['Nam', 'Nam']
                is_vietnamese_from_filename = '_tv' in combined_name or '-tv' in combined_name
            elif 'nam_nu' in combined_name or 'nam-nu' in combined_name or 'nam_nữ' in combined_name or 'nam-nữ' in combined_name:
                genders_from_filename = ['Nam', 'Nữ']
                is_vietnamese_from_filename = '_tv' in combined_name or '-tv' in combined_name
            elif 'nu_nu' in combined_name or 'nu-nu' in combined_name or 'nữ_nữ' in combined_name or 'nữ-nữ' in combined_name:
                genders_from_filename = ['Nữ', 'Nữ']
                is_vietnamese_from_filename = '_tv' in combined_name or '-tv' in combined_name
            
            # If we found gender from filename, use it; otherwise transcribe
            if len(genders_from_filename) == 2:
                genders = genders_from_filename
                is_vietnamese = is_vietnamese_from_filename
                transcripts = ['', '']  # Empty transcripts if using filename detection
            else:
                # Get transcripts and detect gender/language from audio
                transcripts = []
                genders = []
                is_vietnamese = False
                
                for file_info in files_sorted:
                    if '_s1_FT.wav' in file_info['filename'] or '_s1.wav' in file_info['filename'] or '_s2_FT.wav' in file_info['filename'] or '_s2.wav' in file_info['filename']:
                        speaker_path = file_info['file_path']
                        try:
                            transcript = transcribe_audio_vi(speaker_path)
                            transcripts.append(transcript)
                            
                            # Detect gender
                            gender = detect_gender_from_transcript(transcript)
                            genders.append(gender)
                            
                            # Detect Vietnamese
                            if detect_vietnamese(transcript):
                                is_vietnamese = True
                        except Exception as e:
                            print(f"Error transcribing {speaker_path}: {e}")
                            transcripts.append('')
                            genders.append('Unknown')
            
            # Skip if we don't have both speaker genders
            if len(genders) < 2:
                continue
            
            # Create gender label (normalize to lowercase for category key, but keep display format)
            if genders[0] != 'Unknown' and genders[1] != 'Unknown':
                gender_label_lower = f"{genders[0].lower()} {genders[1].lower()}"
                gender_label_display = f"{genders[0]} - {genders[1]}"  # Display format with dash
            else:
                gender_label_lower = None
                gender_label_display = None
            
            if not gender_label_lower:
                continue  # Skip if we can't determine gender
            
            # Get input file from any file in the group
            input_file = files[0].get('input_file')
            
            # Collect speaker files (handle both regular and _FT suffix for finetune)
            speaker_files = []
            for file_info in files_sorted:
                if '_s1_FT.wav' in file_info['filename'] or '_s1.wav' in file_info['filename']:
                    speaker_files.insert(0, file_info['filename'])  # s1 first
                elif '_s2_FT.wav' in file_info['filename'] or '_s2.wav' in file_info['filename']:
                    speaker_files.append(file_info['filename'])  # s2 second
            
            if len(speaker_files) < 2:
                continue
            
            # Create gender label with TV suffix if Vietnamese (for display)
            if is_vietnamese:
                gender_label_display_with_tv = f"{gender_label_display} TV"
            else:
                gender_label_display_with_tv = gender_label_display
            
            # Header MUST be gender label ONLY, NEVER model type
            # Force header to be gender_label_display_with_tv
            header = gender_label_display_with_tv  # e.g., "Nam - Nam" or "Nam - Nam TV"
            
            # Create category key: gender_label + TV suffix if Vietnamese
            category_key = f"{gender_label_lower}_tv" if is_vietnamese else gender_label_lower
            
            entry = {
                'header': header,  # MUST be gender label: "Nam - Nam" or "Nam - Nam TV", NEVER "Finetune"
                'model_type': files[0]['model_type'],  # Keep model_type separate for internal use only
                'gender_label': gender_label_display,  # Display format: "Nam - Nữ" (without TV)
                'gender_label_with_tv': gender_label_display_with_tv,  # With TV suffix if Vietnamese: "Nam - Nam TV"
                'category_key': category_key,  # For grouping: "nam nữ" or "nam nữ_tv"
                'is_vietnamese': is_vietnamese,
                'input_file': input_file,
                'input_dir': files[0]['input_dir'],
                'output_dir': files[0]['output_dir'],
                'speaker_files': speaker_files,
                'transcripts': transcripts,
                'base_name': files[0]['base_name'],
                'timestamp': max(f['timestamp'] for f in files)
            }
            
            # Debug print to verify header is correct
            print(f"DEBUG entry: header='{entry['header']}', gender_label='{entry['gender_label']}', gender_label_with_tv='{entry['gender_label_with_tv']}', model_type='{entry['model_type']}'")
            
            all_entries.append(entry)
        
        # Group entries by category_key and get the most recent one for each category
        category_groups = {}
        for entry in all_entries:
            category_key = entry['category_key']
            if category_key not in category_groups:
                category_groups[category_key] = []
            category_groups[category_key].append(entry)
        
        # For each category, get the most recent entry (by timestamp)
        demo_files_list = []
        for category_key, entries in category_groups.items():
            # Sort by timestamp (most recent first) and take the first one
            entries.sort(key=lambda x: x['timestamp'], reverse=True)
            demo_files_list.append(entries[0])
        
        # Define the order of categories to display
        category_order = ['nam nam', 'nam nữ', 'nữ nữ', 'nam nam_tv', 'nam nữ_tv', 'nữ nữ_tv']
        
        # Sort demo_files_list according to category_order
        demo_files_list.sort(key=lambda x: (
            category_order.index(x['category_key']) if x['category_key'] in category_order else 999,
            -x['timestamp']  # Secondary sort by timestamp (most recent first)
        ))
        
        # Limit to 6 entries (one for each category)
        demo_files_list = demo_files_list[:6]
        
        return jsonify({'demo_files': demo_files_list})
        
    except Exception as e:
        print(f"Demo files error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'models_loaded': list(models.keys()),
        'models_count': len(models)
    })

@app.route('/models')
def get_models():
    """Get list of available models"""
    return jsonify({
        'available_models': list(models.keys()),
        'models_info': {
            model_type: {
                'loaded': True,
                'checkpoint_dir': info['args'].checkpoint_dir,
                'epoch': info.get('epoch'),
                'sidr': info.get('sidr')
            }
            for model_type, info in models.items()
        }
    })

@app.route('/waveform/<path:filepath>')
def waveform_image(filepath):
    """Render waveform image (PNG) for the given audio file path.
    Only allow files inside known output directories.
    """
    try:
        # Build absolute paths and validate
        base_dir = os.path.abspath(os.getcwd())
        model_types = ['clean', 'noise', 'finetune']
        allowed_roots = []
        
        # Add model-specific directories
        for model_type in model_types:
            allowed_roots.append(os.path.abspath(os.path.join(base_dir, f'outputs/{model_type}/input')))
            allowed_roots.append(os.path.abspath(os.path.join(base_dir, f'outputs/{model_type}/output')))
        
        # Add legacy paths for backward compatibility
        allowed_roots.extend([
            os.path.abspath(os.path.join(base_dir, 'outputs/try/input')),
            os.path.abspath(os.path.join(base_dir, 'outputs/try/output')),
            os.path.abspath(os.path.join(base_dir, 'outputs/MossFormer2_SS_8K')),
        ])
        
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
    
    # Load all models
    try:
        load_all_models()
    except Exception as e:
        print(f"\n{'='*50}")
        print("ERROR: Failed to load models")
        print(f"{'='*50}")
        print(f"Error details: {str(e)}")
        print(f"\nPlease check:")
        print(f"  1. Checkpoint files exist for at least one model:")
        print(f"     - clean: checkpoints/Libri2Mix_min_adam/MossFormer2_SS_8K_clean/")
        print(f"     - finetune: checkpoints/VietVivoMix/MossFormer2_SS_8K_clean_finetune/")
        print(f"     - noise: checkpoints/Libri2Mix_min_adam/MossFormer2_SS_8k_noise/ (optional)")
        print(f"  2. Required files: last_best_checkpoint or last_checkpoint")
        print(f"  3. All dependencies are installed (run: pip install -r requirements.txt)")
        print(f"  4. Config file exists: config/inference/MossFormer2_SS_8K.yaml")
        print(f"{'='*50}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\nStarting Flask server...")
    print("Open http://localhost:5000 in your browser")
    
    # Run Flask app (enable auto-reload if FLASK_DEBUG=1)
    debug_flag = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=5000, debug=debug_flag, use_reloader=debug_flag)
