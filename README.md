# PEFT LoRA Eval

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/constantineserkov/peft-lora-eval/blob/main/notebooks/colab_runner.ipynb)

`peft-lora-eval` is an experiment runner for comparing parameter-efficient fine-tuning methods on causal language models. The project is built around LLaMA-style instruction fine-tuning and currently supports:

- Base model evaluation and inference
- LoRA
- QLoRA
- DoRA
- QDoRA
- Train, eval, benchmark, and interactive inference stages
- Resume checkpoints and best-adapter checkpoints per method
- W&B logging
- Hugging Face authentication for gated models
- lm-eval benchmark integration
- Colab-friendly runs with persistent `run_id` metadata

The default target model is `meta-llama/Llama-3.2-3B`, and the default dataset is `yahma/alpaca-cleaned`.

## Current Status

This is a research/experiment runner, not a packaged training service. The core local/Colab GPU pipeline is the supported path today:

- Supported: CUDA GPU runs, Colab T4 runs, train/eval/bench/inf stages, PEFT checkpoints, W&B logging.
- Not supported yet: CPU-only execution, Kaggle, distributed training, automated hyperparameter search.

Tested target environment: Python 3.12, PyTorch/Transformers/PEFT stack, NVIDIA CUDA GPU. Colab T4 is the main low-cost target. Full-dataset runs and non-quantized methods may OOM on T4; start with a small `--data-subset`.

## Project Goals

This repository is meant to answer practical PEFT questions:

- How do LoRA, QLoRA, DoRA, and QDoRA compare on quality?
- How much VRAM does each method need?
- How long does each method take to train and evaluate?
- Which methods are worth using under Colab or single-GPU constraints?
- How reproducible are results when seed, dataset subset, and checkpoints are controlled?

The project emphasizes experiment hygiene: resolved configs, deterministic seeds, per-method metadata, resumable checkpoints, and comparison tables.

## Repository Layout

```text
configs/
  run.yaml                 # Main run config
  methods/
    base.yaml              # Base model config
    lora.yaml              # LoRA config
    qlora.yaml             # QLoRA config
    dora.yaml              # DoRA config
    qdora.yaml             # QDoRA config
scripts/
  run_pipeline.py          # Main entrypoint
notebooks/
  colab_runner.ipynb       # Google Colab T4 runner
src/
  runner/                  # Pipeline orchestration
  auth.py                  # Hugging Face and W&B auth
  benchmark.py             # lm-eval integration
  checkpoint.py            # Resume and best-adapter checkpoints
  dataloader.py            # Dataset loading, formatting, tokenization
  evaluator.py             # Test-set evaluation
  inference.py             # Interactive inference loop
  logger.py                # Console/file logging
  model_utils.py           # Model loading and PEFT setup
  trainer.py               # Training loop
  results.py               # Result persistence and plots
tests/                     # Unit and smoke tests
```

## Supported Methods

| Method | Adapter | Quantization | Notes |
| --- | --- | --- | --- |
| `base` | None | No | Loads the base model without PEFT. Useful for baseline eval, benchmark, and inference. |
| `lora` | LoRA | No | Standard low-rank adapter fine-tuning. |
| `qlora` | LoRA | 4-bit | Uses bitsandbytes 4-bit loading with NF4 by default. |
| `dora` | DoRA | No | Weight-decomposed LoRA via PEFT `use_dora=True`. |
| `qdora` | DoRA | 4-bit | DoRA with 4-bit base model loading. |

For efficient runs, group methods by model loading mode when possible. For example:

```bash
--methods base/lora/dora/qlora/qdora
```

This keeps non-quantized methods together and quantized methods together, reducing unnecessary model reloads.

## Quick Start

### Local GPU Run

The project targets Python 3.12+ and is intended for CUDA-capable NVIDIA GPUs.

Clone the repository:

```bash
git clone https://github.com/constantineserkov/peft-lora-eval.git
cd peft-lora-eval
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
```

On Linux/macOS, activate the virtual environment with:

```bash
source .venv/bin/activate
```

Install dependencies. `uv` is the recommended local path because `pyproject.toml` pins the CUDA PyTorch wheel index used by this project:

```bash
uv sync
```

If you are not using `uv`, install from the locked requirements file and then install the project in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

Copy the environment template and fill in tokens:

```bash
copy .env.example .env
```

On Linux/macOS:

```bash
cp .env.example .env
```

Check the run plan before loading the model:

```bash
python -m scripts.run_pipeline --run_id local_smoke --methods base --stages eval --data-subset 20 --dry-run
```

Run a small local smoke experiment:

```bash
python -m scripts.run_pipeline --run_id local_smoke --methods lora --stages train/eval --data-subset 20
```

Expected outputs:

```text
runs/run_local_smoke/metadata.json
runs/run_local_smoke/lora/checkpoints/best/
runs/run_local_smoke/lora/checkpoints/resume/
results/metrics/
results/wandb/
```

Generated dataset caches and run outputs are ignored by git. The repo-local `data/` directory is not source data; if it appears, it is safe to delete.

### Colab T4 Run

The repository includes a Colab notebook:

```text
notebooks/colab_runner.ipynb
```

Open directly in Colab:

[Open `notebooks/colab_runner.ipynb`](https://colab.research.google.com/github/constantineserkov/peft-lora-eval/blob/main/notebooks/colab_runner.ipynb)

Recommended Colab workflow:

1. Open `notebooks/colab_runner.ipynb` in Google Colab.
2. Set `Runtime -> Change runtime type -> T4 GPU`.
3. Add Colab Secrets named `HUGGINGFACE_HUB_TOKEN` and `WANDB_API_KEY`.
4. Run the notebook from top to bottom.
5. Start with a small `--data-subset`, then scale up once the run is stable.
6. Keep the same `--run_id` when resuming after OOM or runtime interruption.

The notebook mounts Google Drive, loads secrets, refreshes the repository under `/content/peft-lora-eval`, installs `requirements_colab.txt`, installs the project in editable mode, prints GPU diagnostics, and runs `scripts.run_pipeline`.

Use a T4-friendly first run:

```bash
python -m scripts.run_pipeline --run_id colab_t4_smoke --methods lora --stages train/eval --data-subset 20 --log-level INFO
```

Then run a larger PEFT comparison:

```bash
python -m scripts.run_pipeline --run_id colab_t4_exp_001 --methods lora/dora/qlora/qdora --stages train/eval/bench --data-subset 2000 --log-level INFO
```

Before launching a long run, verify the plan:

```bash
python -m scripts.run_pipeline --run_id colab_t4_exp_001 --methods lora/dora/qlora/qdora --stages train/eval/bench --data-subset 2000 --dry-run
```

Kaggle is not supported yet.

### Authentication

The default model is gated, so Hugging Face auth is usually required.

For local runs, set these environment variables when possible:

```bash
HUGGINGFACE_HUB_TOKEN=...
WANDB_API_KEY=...
```

For Colab, use Colab Secrets with the same names. If tokens are missing, the code tries the default Hugging Face/W&B login flows.

## Configuration

The main config is [configs/run.yaml](configs/run.yaml).

Important fields:

```yaml
model:
  model_name_or_path: "meta-llama/Llama-3.2-3B"

dataset:
  name: "yahma/alpaca-cleaned"
  subset: null
  val_ratio: 0.05
  test_ratio: 0.05

runtime:
  seed: 17
  merge: false
  use_small_model: false
```

`dataset.subset: null` means use the full train split. To use only the first N examples:

```yaml
dataset:
  subset: 2000
```

or from CLI:

```bash
--data-subset 2000
```

Each method has its own override file in `configs/methods/`. The runner deep-merges the main config with the selected method config before each stage.

## CLI

Main entrypoint:

```bash
python -m scripts.run_pipeline --run_id <id> --methods <methods> --stages <stages>
```

Common arguments:

| Argument | Meaning |
| --- | --- |
| `--run_id` | Required. Creates or resumes `runs/run_<id>`. |
| `--config` | Main config path. Defaults to `configs/run.yaml`. |
| `--methods` | Slash-separated methods, for example `lora/qlora`. |
| `--stages` | Slash-separated stages: `train`, `eval`, `bench`, `inf`. |
| `--wandb-project` | Override `logging.wandb_project`. |
| `--base-path` | Root path for configs, runs, results, and cache-relative outputs. Useful for Colab/Drive persistence. |
| `--data-subset` | Override `dataset.subset`. |
| `--seed` | Override `runtime.seed`. |
| `--log-level` | Logging level. Defaults to `INFO`. |
| `--merge` | Merge PEFT adapter into the base model for eval/inference stages. |
| `--use-small-model` | Switch model loading to `gpt2`. Experimental. |
| `--dry-run` | Print the resolved plan and exit before model/data loading. |

## Dry Run

Use dry run before expensive Colab jobs:

```bash
python -m scripts.run_pipeline --run_id test_1 --methods dora/qdora --stages train/eval/bench --dry-run
```

Example output:

```text
Dry run plan:
Run: run_test_1
Methods: dora, qdora
Requested stages: train, eval, bench
Will run: qdora_train, qdora_eval
Will skip: dora_train, dora_eval
Dataset: yahma/alpaca-cleaned, full train split
Seed: 17
```

Dry run is meant to catch stage, method, dataset, and seed mistakes before downloading or loading large models.

## Example Runs

Train and evaluate LoRA:

```bash
python -m scripts.run_pipeline --run_id exp_001 --methods lora --stages train/eval
```

Train and evaluate DoRA and QDoRA:

```bash
python -m scripts.run_pipeline --run_id exp_002 --methods dora/qdora --stages train/eval
```

Run benchmarks after training:

```bash
python -m scripts.run_pipeline --run_id exp_002 --methods dora/qdora --stages bench
```

Run interactive inference:

```bash
python -m scripts.run_pipeline --run_id exp_002 --methods dora --stages inf
```

During inference, type exactly:

```text
STOP
```

to log inference metrics and exit.

## Stage Behavior

Stages run in this order:

```text
train -> eval -> bench -> inf
```

If you request `eval`, `bench`, or `inf` for a PEFT method without explicitly requesting `train`, the runner still schedules `train` first unless the method is `base`.

The runner records completed stages in `runs/run_<id>/metadata.json`. If you rerun the same `run_id`, completed stages are skipped and logged:

```text
Skipping completed stage: dora_train
Skipping completed stage: dora_eval
```

## Checkpoints and Resume

Each method has its own checkpoint metadata:

```json
"checkpoints": {
  "dora": {
    "latest_resume": "checkpoint_epoch_1_step_100.pt",
    "latest_resume_path": ".../runs/run_exp/dora/checkpoints/resume/checkpoint_epoch_1_step_100.pt",
    "best_adapter": ".../runs/run_exp/dora/checkpoints/best"
  }
}
```

Checkpoint types:

- `checkpoints/resume/`: full training resume state, including adapter weights, optimizer, scheduler, scaler, epoch, step, and best loss.
- `checkpoints/best/`: PEFT adapter saved with `model.save_pretrained(...)` when validation loss improves.

This lets `dora` and `qdora` resume independently in the same run.

## Evaluation

The project has two evaluation paths.

### Test-set Evaluation

The `eval` stage computes loss/perplexity-style metrics on the held-out test split created from the configured dataset.

Results are saved under:

```text
results/metrics/
```

### lm-eval Benchmarks

The `bench` stage runs EleutherAI lm-eval through `src/benchmark.py`.

Benchmark config lives in `configs/run.yaml`:

```yaml
benchmark:
  batch_size: 32
  tasks:
    - "arc_challenge"
  num_fewshot: null
  limit: null
  use_cache: false
```

`benchmark.use_cache` accepts:

- `false` or `null`: disabled
- `true`: create a run-scoped lm-eval cache path
- string path: use that explicit cache path

lm-eval seeds are wired to the project seed by default.

Benchmark results are saved under:

```text
results/benchmarks/
```

## Metrics and Logging

The code currently logs:

- Training loss
- Validation loss
- Perplexity
- Learning rate
- VRAM usage
- Runtime and peak VRAM for benchmarks
- Inference latency and memory metrics
- W&B run config and selected metrics

The project notes call out future work around stronger experiment reporting: throughput, gradient norms, GPU utilization, adapter size, GPU-hours, multiple seeds, and cost-vs-quality plots.

## Outputs

Typical generated paths:

```text
data/
runs/run_<id>/metadata.json
runs/run_<id>/<method>/checkpoints/best/
runs/run_<id>/<method>/checkpoints/resume/
results/metrics/
results/benchmarks/
results/comparisons/
results/wandb/
```

Generated files are intentionally separate from source configs so runs can be resumed and compared. `data/`, `runs/`, and `results/` are local artifacts and should not be committed.

## Development

Run the test suite with:

```bash
pytest
```

Most unit tests are designed to avoid model downloads and GPU work. Full pipeline runs, benchmark stages, gated-model loading, and Colab experiments may require CUDA, network access, Hugging Face authentication, and W&B credentials.

## Colab Notes

Colab runs should use a stable `--run_id` and a persistent `runtime.base_path` if outputs need to survive runtime resets. Use the provided notebook at `notebooks/colab_runner.ipynb` for the standard T4 workflow.

Recommended workflow:

1. Authenticate with Hugging Face and W&B.
2. Run `--dry-run` to verify stages and dataset scope.
3. Start with a subset, for example `--data-subset 2000`.
4. Scale to `dataset.subset: null` only after the pipeline is validated.
5. Keep the same `--run_id` when resuming after OOM or runtime interruption.

## Known Limitations

- `--use-small-model` is experimental and currently not treated as a reliable smoke mode.
- Interactive inference is basic. It feeds raw prompts to the model and is not a full chat template runner.
- Base model inference uses a base pretrained model, not an instruction-tuned chat model.
- Hyperparameter search is not automated yet.
- Distributed training is not implemented yet.
- CPU-only execution is not supported yet.
- Kaggle is not supported yet.
- Some metrics in `inference.py` are placeholders or partial aggregates.

## Future Work

Planned improvements from the project notes:

- Automate hyperparameter tuning with repeatable trials per method.
- Tune LoRA first, then retune QLoRA/QDoRA instead of assuming the same parameters transfer cleanly.
- Add multiple configured adapters per method, for example `lora_1`, `lora_2`, `qdora_1`.
- Add stronger comparison tables with one row per method/run/seed.
- Add multiple-seed reporting with mean and standard deviation.
- Add gradient norm, throughput, GPU utilization, adapter size, checkpoint size, and GPU-hour metrics.
- Add latency benchmarks for merged and unmerged adapters.
- Add adapter switching to avoid reinitializing the base model when possible.
- Add distributed training for multiple GPUs.
- Add CPU-only support for tiny smoke tests and non-GPU validation.
- Investigate LoFTQ and quantization-aware training for LoRA/QLoRA/QDoRA.
- Investigate Liger Kernel for QLoRA-style runs.
- Refactor config layout into clearer base/model/task configs.
- Improve Colab persistence and resolved-config logging.
- Add LLM-judge or human-eval workflows for instruction-following quality.

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
