#!/bin/bash
# Setup script for Lambda Cloud GPU instances (GH200 / A100 / H100).
# Run: bash setup_lambda.sh

set -e

echo "=== Quantization & Alignment Research Pipeline Setup ==="

# Detect architecture
ARCH=$(uname -m)
echo "Architecture: $ARCH"

# Lambda instances come with CUDA, PyTorch, and Python pre-installed.
# Create a venv that inherits system packages (PyTorch, CUDA).
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv --system-site-packages venv
fi

source venv/bin/activate

echo "Installing project dependencies..."
pip install --upgrade pip

# On GH200 (aarch64), bitsandbytes needs to be built with ARM support.
# Recent versions (>=0.43) ship aarch64 wheels. If install fails, build from source.
if [ "$ARCH" = "aarch64" ]; then
    echo "Detected ARM (aarch64) — GH200 Grace Hopper"
    echo "Installing bitsandbytes with aarch64 support..."
    pip install bitsandbytes>=0.43.0 || {
        echo "Pre-built wheel failed, building bitsandbytes from source..."
        pip install bitsandbytes --no-binary bitsandbytes
    }
fi

pip install -r requirements.txt

# Install Jupyter for notebook usage
echo "Installing Jupyter..."
pip install jupyterlab ipykernel ipywidgets

# Register the venv as a Jupyter kernel
python -m ipykernel install --user --name=quantization --display-name="Quantization Research"

# Verify GPU availability
echo ""
echo "=== GPU Check ==="
python3 -c "
import torch, platform
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  GPU: {gpu}')
    print(f'  VRAM: {mem:.0f} GB')
    print(f'  CUDA: {torch.version.cuda}')
    print(f'  PyTorch: {torch.__version__}')
    print(f'  Arch: {platform.machine()}')
else:
    print('  WARNING: No GPU detected!')
    exit(1)
"

# Verify key imports
echo ""
echo "=== Dependency Check ==="
python3 -c "
import transformers, bitsandbytes, sklearn, matplotlib, pandas, numpy, tqdm, accelerate
print(f'  transformers:  {transformers.__version__}')
print(f'  bitsandbytes:  {bitsandbytes.__version__}')
print(f'  scikit-learn:  {sklearn.__version__}')
print(f'  accelerate:    {accelerate.__version__}')
print('  All dependencies OK')
"

# Create output directories
mkdir -p results/figures results/data results/comparison

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run experiments in terminal:"
echo "  source venv/bin/activate"
echo "  python run_experiment.py --model mistral --precision 4bit"
echo ""
echo "To run in Jupyter:"
echo "  source venv/bin/activate"
echo "  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser"
echo "  # Then open notebooks/run_pipeline.ipynb"
echo "  # Select kernel: 'Quantization Research'"
