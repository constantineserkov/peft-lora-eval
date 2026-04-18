# PEFT-LoRA-Eval

**A full-stack PEFT experimentation framework** for LoRA, QLoRA, DoRA & QDoRA on causal LMs (Llama-3, Mistral, etc.).

Train → Eval → Benchmark (lm-eval) → Inference in one CLI command.  
Built for reproducible research with checkpoints, WandB, custom metrics, plots, and side-by-side comparisons.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Features
- Full pipeline orchestration (`train/eval/bench/inference`)
- 4 PEFT methods + `base`/`instruct` baselines
- 4-bit QLoRA/QDoRA with FlashAttention-2 auto-detection
- Resume from checkpoint + best-model saving
- lm-eval benchmarks (MMLU, HellaSwag, ARC, etc.)
- Live WandB logging + beautiful loss/perplexity/VRAM plots
- Custom causal-LM dataloader (Alpaca format) with length-sorted batches
- Inference server with latency, VRAM, energy tracking

## 🚀 Quick Start

```bash
git clone https://github.com/yourname/peft-lora-eval.git
cd peft-lora-eval
pip install -r requirements.txt

# Run full pipeline (LoRA only)
python -m src.runner.pipeline --methods lora --stages train/eval/bench --run_id 001
```
## 📁 Project Structure
peft-lora-eval/ 
├── configs/                  # one YAML per method
├── results/                  # metrics + plots (auto-generated)
├── runs/                     # checkpoints + metadata
├── src/
│   ├── runner/               # pipeline orchestrators
│   └── *.py                  # core modules
├── .github/
├── LICENSE
├── README.md
└── requirements.txt

## 📋 Module Overview
(See detailed explanations below)

## Configuration

All settings live in configs/{method}_config.yaml. Full example included.
## Results
(Insert your comparison tables and benchmark screenshots here)

## Contributing & Roadmap
(See CONTRIBUTING.md)

## License
MIT © 2026


---

### 2. Explanation of Every File (what it does + why it exists)

#### `src/runner/` – The brain of the project
| File                    | Purpose |
|-------------------------|--------|
| `runner/common.py`      | Heart of initialization. Parses CLI args, creates `runs/run_XXX/` folder, manages `metadata.json` (attempt counter, completed stages, wall-clock time, device, config). Sets up logging + WandB + HF auth. Decides which stages/methods still need to run (`resolve_stages`). |
| `runner/pipeline.py`    | **Main entry point**. Calls `init_run` → builds execution plan → loops through (method, stage) and dispatches to the right runner. Handles model reuse across stages. |
| `runner/train_runner.py` | Unpacks dataloaders → configures PEFT for training → calls `train_model`. |
| `runner/eval_runner.py`  | Unpacks test loader → configures model for eval → calls `evaluator`. |
| `runner/bench_runner.py` | Loads tokenizer + configures model → calls `run_benchmarks` (lm-eval). |
| `runner/inf_runner.py`   | Interactive inference loop with metrics collection. |

#### Core `src/` modules
| File                    | Purpose |
|-------------------------|--------|
| `utils.py`              | CLI argument parser, config loader + validation, seed setting, device detection, Colab/Kaggle path handling, attention backend auto-selection (FlashAttention-2 on A100/H100/4090). |
| `auth.py`               | Secure WandB + Hugging Face login (checks env vars, `~/.config`, falls back to interactive login). |
| `logger.py`             | Beautiful colored console + file logging (`runs/run_XXX/run.log`). Also initializes NVML once for VRAM tracking. |
| `dataloader.py`         | Loads Alpaca → formats instruction/response → tokenizes with proper `-100` masking for causal LM → length-sorted + custom padding collator. Returns train/val/test DataLoaders. |
| `model_utils.py`        | Loads base model (with 4-bit quant if needed) → applies LoRA/DoRA config → restores checkpoint → logs trainable %. Handles `configure_peft_model_for_training` and `..._for_eval` (including merge & unload). |
| `checkpoint.py`         | Saves/resumes optimizer + scheduler + scaler + PEFT adapter. Separate folders for "best" (adapter only) and "resume" (full state). Auto-cleans old checkpoints. |
| `trainer.py`            | Full training loop with gradient accumulation, mixed precision, linear scheduler, best-model checkpointing, validation every N steps, WandB logging, VRAM tracking. |
| `evaluator.py`          | Evaluation loop (no grad) → computes loss/perplexity/VRAM → saves JSON + generates 5 beautiful matplotlib plots (loss curve, perplexity, VRAM vs batch, VRAM vs loss, summary bar chart). |
| `metrics.py`            | Helper functions: `log_vram_usage`, `compute_training_metrics`, `compute_eval_metrics` (min/max/avg + perplexity). |
| `results.py`            | Saves metrics JSON (`results/metrics/` or `benchmarks/`) and calls plot generation. |
| `benchmark.py`          | Runs `lm_eval` on a list of tasks (ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, GSM8K). Extracts key scores and saves. (Note: currently has hardcoded `MODEL_PATHS` — easy to integrate with metadata.) |
| `inference.py`          | Interactive prompt loop ("STOP" to exit). Tracks latency (p50/p95), tokens/s, VRAM, system RAM, energy (via pynvml). Logs to WandB. |
| `compare.py`            | **(Currently a stub)** — planned to read all JSON results and build side-by-side tables (eval metrics + benchmarks). |
| `__init__.py` files     | Make packages importable. |

---

