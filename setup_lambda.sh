#!/bin/bash
# Setup script for Lambda Cloud A100 instances.
# Run: bash setup_lambda.sh

set -e

echo "=== Quantization & Alignment Research Pipeline Setup ==="
echo "Target: Lambda Cloud A100 GPU"

# Lambda instances come with CUDA, PyTorch, and Python pre-installed.
# Create a venv that inherits system packages (PyTorch, CUDA).
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv --system-site-packages venv
fi

source venv/bin/activate

echo "Installing project dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify GPU availability
echo ""
echo "=== GPU Check ==="
python3 -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'  GPU: {gpu}')
    print(f'  VRAM: {mem:.0f} GB')
    print(f'  CUDA: {torch.version.cuda}')
    print(f'  PyTorch: {torch.__version__}')
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
echo "To run experiments:"
echo "  source venv/bin/activate"
echo ""
echo "  # Single precision:"
echo "  python run_experiment.py --model mistral --precision 4bit"
echo ""
echo "  # Cross-precision comparison:"
echo "  python compare_results.py --model mistral --precisions fp16 8bit 4bit"
echo ""
echo "  # All models and precisions (use tmux for long runs):"
echo "  for model in mistral llama3 qwen; do"
echo "    python compare_results.py --model \$model --precisions fp16 8bit 4bit"
echo "  done"
