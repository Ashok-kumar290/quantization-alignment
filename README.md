# Quantization & Alignment Research Pipeline

A research framework for studying how model quantization affects alignment behaviors in large language models, combining LLM alignment evaluation, mechanistic interpretability, and quantized model analysis.

## Research Hypothesis

> "Quantization compresses internal representation space in transformer models and may distort alignment-related features, leading to increased sycophantic or unsafe behavior."

## Project Structure

```
quantization/
├── models/                    # Model loading + activation hooks
│   ├── model_loader.py        # HF model loading with bitsandbytes quantization
│   └── activation_collector.py # PyTorch hook-based activation extraction
├── datasets/                  # Evaluation datasets
│   ├── sycophancy_dataset.py  # 40 false-claim + 8 true-claim prompts
│   ├── truthfulness_dataset.py # 25 TruthfulQA-style questions
│   └── safety_dataset.py      # 24 harmful + 10 benign-sensitive prompts
├── experiments/
│   └── evaluator.py           # Behavioral evaluation (sycophancy, truth, safety)
├── probes/
│   └── linear_probe.py        # Layer-wise logistic regression probing
├── analysis/
│   └── geometric.py           # CKA, PCA, cosine similarity, activation norms
├── interventions/
│   └── steering.py            # Activation steering for causal validation
├── utils/
│   ├── visualization.py       # Publication-quality matplotlib plots
│   └── metrics.py             # Aggregate metric computation
├── run_experiment.py          # Single-precision pipeline
├── compare_results.py         # Cross-precision comparison
├── requirements.txt
└── results/                   # Output directory
    ├── figures/               # Generated plots
    └── data/                  # JSON results
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full experiment at one precision level
python run_experiment.py --model mistral --precision 4bit

# Skip expensive steps
python run_experiment.py --model mistral --precision 4bit --skip-intervention
python run_experiment.py --model llama3 --precision 8bit --skip-probing

# Compare across precision levels (runs fp16 → 8bit → 4bit sequentially)
python compare_results.py --model mistral --precisions fp16 8bit 4bit
```

## Deployment on Lambda Cloud (A100 GPU)

### 1. Launch Instance

Launch an A100 (40GB or 80GB) instance on Lambda Cloud. The 80GB variant can comfortably hold a 7B FP16 model; 40GB works for all quantized configs and FP16 with offloading.

### 2. Setup Environment

```bash
# SSH into your instance
ssh ubuntu@<your-lambda-ip>

# Clone / upload the project
git clone <your-repo-url> quantization
cd quantization

# Create a virtual environment (Lambda instances come with CUDA + PyTorch pre-installed)
python3 -m venv --system-site-packages venv
source venv/bin/activate

# Install project dependencies (PyTorch is already available system-wide)
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.0f}GB')"
```

### 3. Run Experiments

```bash
# Single precision run (~30-60 min on A100)
python run_experiment.py --model mistral --precision 4bit

# Full cross-precision comparison (~2-4 hours on A100)
python compare_results.py --model mistral --precisions fp16 8bit 4bit

# With all four precision levels
python compare_results.py --model mistral --precisions fp16 8bit 4bit 3bit

# Use screen/tmux for long runs
tmux new -s experiment
python compare_results.py --model mistral --precisions fp16 8bit 4bit
# Ctrl+B, D to detach; tmux attach -t experiment to reattach
```

### 4. Download Results

```bash
# From your local machine
scp -r ubuntu@<your-lambda-ip>:~/quantization/results/ ./results/
```

### Memory Requirements (A100)

| Config | Model | VRAM Usage |
|--------|-------|------------|
| FP16 | Mistral 7B | ~14 GB |
| 8-bit | Mistral 7B | ~8 GB |
| 4-bit | Mistral 7B | ~5 GB |
| 3-bit (NF4) | Mistral 7B | ~4 GB |
| FP16 | Llama 3 8B | ~16 GB |

A100 40GB handles all configurations including FP16. Activation collection adds ~2-4 GB overhead.

## Pipeline Stages

The pipeline executes these stages in order:

1. **Model Loading** — Load HF model with specified quantization via bitsandbytes
2. **Behavioral Evaluation** — Generate responses to sycophancy, truthfulness, and safety prompts; classify behavior with pattern-matching heuristics
3. **Activation Collection** — Use PyTorch forward hooks to capture residual stream outputs at every transformer layer
4. **Probe Training** — Train per-layer logistic regression probes to predict sycophantic/refusal behavior from hidden states; identify layers where alignment is encoded
5. **Geometric Analysis** — PCA projections, activation norms, CKA similarity between precision levels
6. **Causal Interventions** — Activation steering: inject scaled alignment direction vectors into the residual stream and measure behavior change
7. **Metrics & Visualization** — Aggregate all results, generate plots, save JSON

## Output Metrics

**Behavioral:**
- Sycophancy rate (% agreement with false claims)
- Excess sycophancy (controlled for baseline agreement)
- Truthfulness score
- Refusal rate on harmful prompts
- False refusal rate on benign prompts

**Representational:**
- Per-layer probe accuracy (where is alignment encoded?)
- Alignment direction cosine similarity across precisions
- CKA representational similarity
- Per-layer activation norm statistics

**Causal:**
- Intervention effect grid (layer × steering strength)
- Optimal steering configuration per behavior

## Supported Models

| Key | HuggingFace ID |
|-----|---------------|
| `mistral` | `mistralai/Mistral-7B-v0.1` |
| `llama3` | `meta-llama/Llama-3-8B` |
| `qwen` | `Qwen/Qwen2-7B` |

## Extending the Pipeline

**Add custom evaluation prompts:** Save a JSON file matching the dataset format and pass `--custom-prompts-path` (supported by all three dataset classes).

**Add a new model:** Add an entry to `ExperimentConfig.MODEL_CONFIGS` in `run_experiment.py` with the HF model name, hidden size, and layer count.

**Replace the truthfulness classifier:** The current `classify_truthfulness()` uses word-overlap with negation detection. For higher accuracy, swap in an NLI model (e.g., DeBERTa fine-tuned on MNLI).

## Scaling Suggestions

- Run on multiple model families (Llama, Mistral, Qwen) to test generality
- Use instruct-tuned variants to measure alignment strength differences
- Increase dataset sizes by loading TruthfulQA and BeaverTails from HuggingFace `datasets`
- Add GPTQ/AWQ quantization backends alongside bitsandbytes
- Train non-linear probes (MLP) as an upper bound on linear probing accuracy
- Extend activation steering to attention heads (per-head intervention)
