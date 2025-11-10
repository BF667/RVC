# 🚀 Quick Installation Guide

## One-Command Setup
```bash
# Download and run the setup script
curl -fsSL https://raw.githubusercontent.com/BF667/RVC/main/setup.py | python3
```

## Manual Installation

### 1. Clone Repository
```bash
git clone https://github.com/BF667/RVC.git
cd RVC
```

### 2. Run Setup Script
```bash
python3 setup.py
```

### 3. Download a Model
```bash
# Download from URL
python3 rvc_cli.py download "MODEL_URL" "my_voice"

# Or use the model manager directly
python3 -m rvc.modules.model_manager --url "MODEL_URL" --model-name "my_voice"
```

### 4. Start Converting!
```bash
# Launch Gradio interface
python3 launch_gradio.py

# Or use CLI
python3 rvc_cli.py infer "input.wav" "my_voice"
```

## Google Colab (No Installation Required)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BF667/RVC/blob/main/RVC_Enhanced.ipynb)

## Troubleshooting

### CUDA Issues
```bash
# Check CUDA installation
nvidia-smi
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Install CUDA drivers if needed
# Visit: https://developer.nvidia.com/cuda-downloads
```

### Dependency Issues
```bash
# Reinstall with pip
pip install --upgrade pip
pip install -r requirements.txt

# Or with uv (recommended)
pip install uv
uv pip install -r requirements.txt
```

### FFmpeg Issues
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from: https://ffmpeg.org/download.html
```

## Next Steps
1. Read the full [README.md](README.md) for advanced features
2. Check out the [Colab notebook](RVC_Enhanced.ipynb) for tutorials
3. Explore the [Gradio interface](gradio_demo.py) for easy voice conversion
4. Join our [Discord community](https://discord.gg/rvc) for support

---
*Enhanced RVC v2.0 - PolUVR + YTDLP + Custom Kernels*