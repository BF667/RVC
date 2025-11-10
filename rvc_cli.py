#!/usr/bin/env python3
"""
Enhanced RVC Command Line Interface
Simple and intuitive CLI for voice conversion
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_config():
    """Load configuration file"""
    config_file = "config.json"
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            return json.load(f)
    return {}

def list_models():
    """List all available RVC models"""
    config = load_config()
    models_dir = config.get("models", {}).get("rvc_models_dir", "models/RVC_models")
    
    if not os.path.exists(models_dir):
        print("📁 No models directory found")
        return
    
    models = []
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path):
            files = os.listdir(item_path)
            has_model = any(f.endswith('.pth') for f in files)
            if has_model:
                models.append(item)
    
    if not models:
        print("📦 No models installed yet")
        print("💡 Download a model: python3 rvc_cli.py download <url> <name>")
        return
    
    print(f"🎭 Available RVC Models ({len(models)} total):")
    print("=" * 40)
    for i, model in enumerate(sorted(models), 1):
        print(f"{i:2d}. {model}")
    
    print(f"\n💡 Use model name in conversion: python3 rvc_cli.py infer <audio> {models[0]}")

def download_model(url, name, use_poluvr=False):
    """Download a model from URL"""
    try:
        from rvc.modules.model_manager import enhanced_download_from_url
        import gradio as gr
        
        # Create progress callback
        def progress_callback(step, message):
            print(f"📥 {message}")
        
        result = enhanced_download_from_url(
            url, 
            name, 
            progress=gr.Progress(),
            use_poluvr=use_poluvr
        )
        print(f"✅ {result}")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False
    return True

def convert_audio(input_path, model_name, **kwargs):
    """Convert audio using RVC"""
    try:
        from rvc.infer.infer import rvc_infer
        
        # Set default parameters
        params = {
            "rvc_model": model_name,
            "input_path": input_path,
            "f0_method": kwargs.get("f0_method", "rmvpe"),
            "f0_min": kwargs.get("f0_min", 50),
            "f0_max": kwargs.get("f0_max", 1100),
            "hop_length": kwargs.get("hop_length", 128),
            "rvc_pitch": kwargs.get("rvc_pitch", 0),
            "protect": kwargs.get("protect", 0.33),
            "index_rate": kwargs.get("index_rate", 0.25),
            "volume_envelope": kwargs.get("volume_envelope", 1.0),
            "output_format": kwargs.get("output_format", "wav"),
        }
        
        print(f"🎵 Converting {input_path} with model {model_name}...")
        result = rvc_infer(**params)
        print(f"✅ Conversion completed!")
        return result
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return None

def test_system():
    """Test system compatibility"""
    print("🧪 Testing Enhanced RVC System...")
    print("=" * 40)
    
    # Test Python version
    python_version = sys.version_info
    print(f"🐍 Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Test PyTorch
    try:
        import torch
        print(f"🔥 PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"⚡ CUDA: {torch.version.cuda}")
            print(f"🎯 GPU: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  CUDA: Not available (using CPU)")
    except ImportError:
        print("❌ PyTorch: Not installed")
        return False
    
    # Test dependencies
    dependencies = [
        ("gradio", "Gradio"),
        ("librosa", "Librosa"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
    ]
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name}: Available")
        except ImportError:
            print(f"❌ {name}: Not installed")
    
    # Test RVC modules
    try:
        from rvc.infer.infer import rvc_infer
        print(f"✅ RVC Core: Available")
    except ImportError as e:
        print(f"❌ RVC Core: {e}")
        return False
    
    # Test custom kernels
    try:
        from rvc.lib.custom_kernels import CustomCUDAKernels
        print(f"⚡ Custom Kernels: Available")
    except ImportError:
        print(f"⚠️  Custom Kernels: Not available")
    
    # Check models
    config = load_config()
    models_dir = config.get("models", {}).get("rvc_models_dir", "models/RVC_models")
    if os.path.exists(models_dir):
        model_count = len([d for d in os.listdir(models_dir) 
                          if os.path.isdir(os.path.join(models_dir, d)) and 
                          any(f.endswith('.pth') for f in os.listdir(os.path.join(models_dir, d)))])
        print(f"📦 Models: {model_count} installed")
    else:
        print(f"📦 Models: No models directory")
    
    print("\n🎉 System test completed!")
    return True

def show_help():
    """Show detailed help"""
    print("""
🎵 Enhanced RVC - Command Line Interface
========================================

USAGE:
    python3 rvc_cli.py <command> [options]

COMMANDS:
    download <url> <name>     Download a voice model
    infer <audio> <model>     Convert audio to new voice
    list                      List available models
    test                      Test system compatibility
    help                      Show this help message

EXAMPLES:
    # Download a model
    python3 rvc_cli.py download "https://example.com/model.zip" "my_voice"

    # Convert audio with default settings
    python3 rvc_cli.py infer "input.wav" "my_voice"

    # Convert with custom settings
    python3 rvc_cli.py infer "input.wav" "my_voice" --pitch 2 --index-rate 0.5

    # List available models
    python3 rvc_cli.py list

    # Test system
    python3 rvc_cli.py test

INFERENCE OPTIONS:
    --f0-method METHOD        F0 extraction method (rmvpe, fcpe, crepe)
    --f0-min MIN             Minimum F0 frequency (Hz)
    --f0-max MAX             Maximum F0 frequency (Hz)
    --hop-length LENGTH      Hop length for processing
    --pitch SHIFT            Voice pitch shift (-24 to 24)
    --protect RATE           Voice protection rate (0 to 0.5)
    --index-rate RATE        Index rate for feature matching (0 to 1)
    --volume-envelope RATE   Volume envelope adjustment
    --output-format FORMAT   Output format (wav, mp3, flac)

DOWNLOAD OPTIONS:
    --use-poluvr            Apply PolUVR processing to audio files
    --poluvr-model MODEL    PolUVR model to use

QUICK START:
    1. Download a model: python3 rvc_cli.py download <url> <name>
    2. Convert audio: python3 rvc_cli.py infer <audio> <model>
    3. Find output in: output/RVC_output/

For more information, see README.md
""")

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced RVC Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download a voice model')
    download_parser.add_argument('url', help='Model URL')
    download_parser.add_argument('name', help='Model name')
    download_parser.add_argument('--use-poluvr', action='store_true', help='Apply PolUVR processing')
    download_parser.add_argument('--poluvr-model', default='UVR-MDX-NET 1 2 3', help='PolUVR model')
    
    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Convert audio')
    infer_parser.add_argument('input', help='Input audio file')
    infer_parser.add_argument('model', help='RVC model name')
    infer_parser.add_argument('--f0-method', default='rmvpe', help='F0 method')
    infer_parser.add_argument('--f0-min', type=float, default=50, help='F0 min frequency')
    infer_parser.add_argument('--f0-max', type=float, default=1100, help='F0 max frequency')
    infer_parser.add_argument('--hop-length', type=int, default=128, help='Hop length')
    infer_parser.add_argument('--pitch', type=float, default=0, help='Pitch shift')
    infer_parser.add_argument('--protect', type=float, default=0.33, help='Protection rate')
    infer_parser.add_argument('--index-rate', type=float, default=0.25, help='Index rate')
    infer_parser.add_argument('--volume-envelope', type=float, default=1.0, help='Volume envelope')
    infer_parser.add_argument('--output-format', default='wav', help='Output format')
    
    # Other commands
    subparsers.add_parser('list', help='List available models')
    subparsers.add_parser('test', help='Test system compatibility')
    
    args = parser.parse_args()
    
    if not args.command:
        show_help()
        return
    
    # Execute command
    if args.command == 'download':
        download_model(args.url, args.name, args.use_poluvr)
        
    elif args.command == 'infer':
        if not os.path.exists(args.input):
            print(f"❌ Input file not found: {args.input}")
            return
        
        kwargs = {
            'f0_method': args.f0_method,
            'f0_min': args.f0_min,
            'f0_max': args.f0_max,
            'hop_length': args.hop_length,
            'rvc_pitch': args.pitch,
            'protect': args.protect,
            'index_rate': args.index_rate,
            'volume_envelope': args.volume_envelope,
            'output_format': args.output_format,
        }
        
        result = convert_audio(args.input, args.model, **kwargs)
        if result:
            print(f"🎧 Output: {result}")
        
    elif args.command == 'list':
        list_models()
        
    elif args.command == 'test':
        test_system()
        
    else:
        show_help()

if __name__ == "__main__":
    main()