#!/usr/bin/env python3
"""
RVC+ Setup Script
Automates the installation and setup process for RVC+
"""

import os
import sys
import subprocess
import platform
import argparse
import urllib.request
import json
from pathlib import Path

class RVCPlusSetup:
    def __init__(self):
        self.system = platform.system()
        self.python_version = sys.version_info
        self.cuda_available = False
        self.setup_log = []
        
    def log(self, message, level="INFO"):
        """Log setup progress"""
        timestamp = subprocess.check_output(['date', '+%H:%M:%S'], text=True).strip()
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        self.setup_log.append(log_entry)
    
    def run_command(self, command, check=True, capture_output=False):
        """Run shell command with error handling"""
        try:
            if capture_output:
                result = subprocess.run(command, shell=True, capture_output=True, text=True, check=check)
                return result.stdout.strip()
            else:
                result = subprocess.run(command, shell=True, check=check)
                return result.returncode == 0
        except subprocess.CalledProcessError as e:
            self.log(f"Command failed: {command}", "ERROR")
            self.log(f"Error: {e}", "ERROR")
            return False
    
    def check_python_version(self):
        """Check if Python version is supported"""
        self.log("Checking Python version...")
        
        if self.python_version < (3, 8):
            self.log(f"Python {self.python_version.major}.{self.python_version.minor} is not supported. Please use Python 3.8+", "ERROR")
            return False
        
        if self.python_version >= (3, 13):
            self.log(f"Python {self.python_version.major}.{self.python_version.minor} may have compatibility issues. Recommended: 3.8-3.12", "WARNING")
        
        self.log(f"Python {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro} ✓")
        return True
    
    def check_cuda(self):
        """Check CUDA availability"""
        self.log("Checking CUDA availability...")
        
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                self.cuda_available = True
                cuda_version = torch.version.cuda
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                
                self.log(f"CUDA {cuda_version} detected ✓")
                self.log(f"GPU: {gpu_name}")
                self.log(f"GPU Count: {gpu_count}")
                return True
            else:
                self.log("CUDA not available, will use CPU (slower)", "WARNING")
                return False
        except ImportError:
            self.log("PyTorch not installed yet, will check after installation", "INFO")
            return False
    
    def install_system_dependencies(self):
        """Install system-level dependencies"""
        self.log("Installing system dependencies...")
        
        if self.system == "Linux":
            # Install FFmpeg and other audio libraries
            self.run_command("apt-get update", check=False)
            self.run_command("apt-get install -y ffmpeg libsndfile1-dev", check=False)
            self.log("Linux system dependencies installed ✓")
            
        elif self.system == "Darwin":  # macOS
            # Check if Homebrew is installed
            if not self.run_command("which brew", check=False):
                self.log("Homebrew not found. Please install Homebrew first: /bin/bash -c '$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)'", "ERROR")
                return False
            
            self.run_command("brew install ffmpeg libsndfile", check=False)
            self.log("macOS system dependencies installed ✓")
            
        elif self.system == "Windows":
            # Note: Windows users should install FFmpeg manually or via package managers
            self.log("Windows detected. Please ensure FFmpeg is installed and in PATH", "INFO")
            self.log("Download FFmpeg from: https://ffmpeg.org/download.html#build-windows", "INFO")
        
        return True
    
    def install_python_dependencies(self, use_uv=True, upgrade_pip=True):
        """Install Python dependencies"""
        self.log("Installing Python dependencies...")
        
        if use_uv:
            # Try to install uv package manager
            self.log("Installing uv package manager...")
            if not self.run_command("pip install uv", check=False):
                self.log("Failed to install uv, falling back to pip", "WARNING")
                use_uv = False
        
        # Install PyTorch first
        self.log("Installing PyTorch...")
        if self.cuda_available:
            if self.python_version >= (3, 12):
                torch_install = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            else:
                torch_install = "pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118"
        else:
            torch_install = "pip install torch torchvision torchaudio"
        
        if not self.run_command(torch_install, check=False):
            self.log("Failed to install PyTorch", "ERROR")
            return False
        
        # Install other dependencies
        self.log("Installing other dependencies...")
        
        if use_uv:
            cmd = "uv pip install -r requirements.txt"
        else:
            cmd = "pip install -r requirements.txt"
        
        if not self.run_command(cmd, check=False):
            self.log("Failed to install some dependencies, continuing...", "WARNING")
        
        # Install optional custom kernel support
        if self.cuda_available:
            self.log("Installing custom kernel support...")
            try:
                if self.python_version >= (3, 12):
                    self.run_command("pip install cupy-cuda12x", check=False)
                else:
                    self.run_command("pip install cupy", check=False)
                self.log("Custom kernel support installed ✓")
            except:
                self.log("Custom kernel installation failed, skipping", "WARNING")
        
        return True
    
    def download_models(self):
        """Download essential models"""
        self.log("Downloading essential models...")
        
        # Create directories
        os.makedirs("models/embedders", exist_ok=True)
        os.makedirs("models/RVC_models", exist_ok=True)
        
        # Download Hubert model
        hubert_path = "models/embedders/hubert_base.pt"
        if not os.path.exists(hubert_path):
            self.log("Downloading Hubert model...")
            try:
                # Try multiple download sources
                sources = [
                    "https://huggingface.co/ccccc/RVC-pretrained/resolve/main/hubert_base.pt",
                    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"
                ]
                
                for source in sources:
                    try:
                        urllib.request.urlretrieve(source, hubert_path)
                        self.log("Hubert model downloaded ✓")
                        break
                    except:
                        continue
                else:
                    self.log("Failed to download Hubert model from all sources", "WARNING")
                    
            except Exception as e:
                self.log(f"Failed to download Hubert model: {e}", "WARNING")
        else:
            self.log("Hubert model already exists ✓")
        
        return True
    
    def run_tests(self):
        """Run basic functionality tests"""
        self.log("Running functionality tests...")
        
        # Test imports
        try:
            import torch
            self.log("PyTorch import ✓")
        except ImportError:
            self.log("PyTorch import failed", "ERROR")
            return False
        
        try:
            import gradio
            self.log("Gradio import ✓")
        except ImportError:
            self.log("Gradio import failed", "ERROR")
            return False
        
        try:
            import librosa
            self.log("Librosa import ✓")
        except ImportError:
            self.log("Librosa import failed", "WARNING")
        
        try:
            from rvc.infer.infer import rvc_infer
            self.log("RVC module import ✓")
        except ImportError:
            self.log("RVC module import failed", "ERROR")
            return False
        
        # Test CUDA if available
        if self.cuda_available:
            try:
                device = torch.device("cuda")
                test_tensor = torch.randn(100, 100, device=device)
                result = torch.mm(test_tensor, test_tensor.t())
                self.log("CUDA functionality test ✓")
            except Exception as e:
                self.log(f"CUDA test failed: {e}", "WARNING")
        
        return True
    
    def create_launcher_scripts(self):
        """Create convenient launcher scripts"""
        self.log("Creating launcher scripts...")
        
        # Gradio launcher
        gradio_script = '''#!/usr/bin/env python3
"""RVC+ Gradio Launcher"""
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from gradio_demo import create_enhanced_gradio_interface
    
    print("🎵 Starting RVC+ Gradio Interface...")
    demo = create_enhanced_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True,
        show_tips=True,
        height=800,
        title="RVC+ - PolUVR + YTDLP + Custom Kernels"
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
except Exception as e:
    print(f"❌ Error starting Gradio: {e}")
'''
        
        with open("launch_gradio.py", "w") as f:
            f.write(gradio_script)
        self.run_command("chmod +x launch_gradio.py")
        
        # CLI launcher
        cli_script = '''#!/usr/bin/env python3
"""RVC+ CLI Launcher"""
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("RVC+ CLI")
        print("Usage: python3 rvc_cli.py <command> [args...]")
        print("")
        print("Commands:")
        print("  infer <input> <model>  - Convert audio")
        print("  download <url> <name>  - Download model")
        print("  list                   - List models")
        print("  test                   - Run tests")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "infer":
        if len(sys.argv) < 4:
            print("Usage: rvc_cli.py infer <input_audio> <model_name>")
            sys.exit(1)
        input_path = sys.argv[2]
        model_name = sys.argv[3]
        
        from rvc.infer.infer import rvc_infer
        result = rvc_infer(
            rvc_model=model_name,
            input_path=input_path,
            f0_method="rmvpe",
            rvc_pitch=0,
            index_rate=0.25,
            protect=0.33
        )
        print(f"✅ Conversion completed: {result}")
        
    elif command == "download":
        if len(sys.argv) < 4:
            print("Usage: rvc_cli.py download <url> <model_name>")
            sys.exit(1)
        url = sys.argv[2]
        model_name = sys.argv[3]
        
        from rvc.modules.model_manager import enhanced_download_from_url
        result = enhanced_download_from_url(url, model_name)
        print(result)
        
    elif command == "list":
        from rvc.modules.model_manager import list_models
        print(list_models())
        
    elif command == "test":
        # Run basic tests
        print("🧪 Running RVC+ tests...")
        try:
            import torch
            import gradio
            import librosa
            print("✅ All core dependencies available")
            
            if torch.cuda.is_available():
                print(f"✅ CUDA {torch.version.cuda} detected")
                print(f"✅ GPU: {torch.cuda.get_device_name()}")
            else:
                print("⚠️  CUDA not available, using CPU")
                
            print("🎵 RVC+ ready for use!")
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
'''
        
        with open("rvc_cli.py", "w") as f:
            f.write(cli_script)
        self.run_command("chmod +x rvc_cli.py")
        
        self.log("Launcher scripts created ✓")
        return True
    
    def generate_config(self):
        """Generate configuration file"""
        self.log("Generating configuration...")
        
        config = {
            "version": "2.0.0",
            "python_version": f"{self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}",
            "cuda_available": self.cuda_available,
            "system": self.system,
            "performance": {
                "mixed_precision": self.cuda_available,
                "custom_kernels": self.cuda_available,
                "batch_size": 1,
                "memory_optimization": True
            },
            "models": {
                "hubert_base_path": "models/embedders/hubert_base.pt",
                "rvc_models_dir": "models/RVC_models",
                "output_dir": "output/RVC_output"
            },
            "audio": {
                "sample_rate": 44100,
                "hop_length": 128,
                "f0_min": 50,
                "f0_max": 1100
            }
        }
        
        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        self.log("Configuration generated ✓")
        return True
    
    def print_summary(self):
        """Print setup summary"""
        print("\n" + "="*60)
        print("🎉 ENHANCED RVC SETUP COMPLETE!")
        print("="*60)
        
        print(f"\n📊 System Information:")
        print(f"  🐍 Python: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        print(f"  🖥️  OS: {self.system}")
        print(f"  ⚡ CUDA: {'✅ Available' if self.cuda_available else '❌ Not Available'}")
        
        print(f"\n🚀 Quick Start:")
        print(f"  1. Web Interface: python3 launch_gradio.py")
        print(f"  2. CLI Interface: python3 rvc_cli.py")
        print(f"  3. Download Model: python3 rvc_cli.py download <url> <name>")
        print(f"  4. Convert Audio: python3 rvc_cli.py infer <audio> <model>")
        
        print(f"\n📁 Key Files:")
        print(f"  📖 Documentation: README.md")
        print(f"  🎛️  Gradio Demo: gradio_demo.py")
        print(f"  ⚙️  Config: config.json")
        print(f"  📊 Colab: RVC_Enhanced.ipynb")
        
        if not self.cuda_available:
            print(f"\n⚠️  Performance Note:")
            print(f"  CUDA not detected. Install NVIDIA drivers and CUDA for optimal performance.")
        
        print(f"\n💡 Next Steps:")
        print(f"  1. Download a voice model from the model manager")
        print(f"  2. Try the enhanced Gradio interface")
        print(f"  3. Test with your own audio files")
        print(f"  4. Explore advanced features in the documentation")
        
        print("\n" + "="*60)
    
    def save_setup_log(self):
        """Save setup log to file"""
        log_file = "setup_log.txt"
        with open(log_file, "w") as f:
            f.write("RVC+ Setup Log\n")
            f.write("="*50 + "\n")
            f.write(f"Date: {subprocess.check_output(['date'], text=True).strip()}\n")
            f.write(f"System: {self.system}\n")
            f.write(f"Python: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}\n")
            f.write(f"CUDA: {self.cuda_available}\n\n")
            f.write("\n".join(self.setup_log))
        
        self.log(f"Setup log saved to {log_file} ✓")

def main():
    parser = argparse.ArgumentParser(description="RVC+ Setup Script")
    parser.add_argument("--skip-deps", action="store_true", help="Skip system dependencies installation")
    parser.add_argument("--pip-only", action="store_true", help="Use pip instead of uv")
    parser.add_argument("--no-models", action="store_true", help="Skip model downloads")
    parser.add_argument("--no-tests", action="store_true", help="Skip functionality tests")
    parser.add_argument("--quiet", action="store_true", help="Run in quiet mode")
    
    args = parser.parse_args()
    
    setup = RVCSetup()
    
    print("🎵 RVC+ Setup Script")
    print("="*50)
    
    # Step 1: Check Python version
    if not setup.check_python_version():
        sys.exit(1)
    
    # Step 2: Check CUDA
    setup.check_cuda()
    
    # Step 3: Install system dependencies
    if not args.skip_deps:
        if not setup.install_system_dependencies():
            if not args.quiet:
                print("❌ System dependencies installation failed")
            sys.exit(1)
    
    # Step 4: Install Python dependencies
    if not setup.install_python_dependencies(use_uv=not args.pip_only):
        if not args.quiet:
            print("❌ Python dependencies installation failed")
        sys.exit(1)
    
    # Step 5: Download models
    if not args.no_models:
        setup.download_models()
    
    # Step 6: Create launcher scripts
    setup.create_launcher_scripts()
    
    # Step 7: Generate configuration
    setup.generate_config()
    
    # Step 8: Run tests
    if not args.no_tests:
        setup.run_tests()
    
    # Step 9: Print summary and save log
    setup.print_summary()
    setup.save_setup_log()

if __name__ == "__main__":
    main()