# ScaleDown Benchmarking Framework

Clean, modular, reproducible benchmarking framework to evaluate ScaleDown's context compression layer on downstream LLM workloads.

## Overview

This framework evaluates **accuracy**, **latency**, and **cost** when using ScaleDown compression vs. baseline LLMs on QA tasks with long contexts. Currently supports:

- **Datasets**: HotpotQA (multi-hop QA)
- **Models**: OpenAI , Gemini
- **Metrics**: Exact Match, Token-level F1
- **Tasks**: RAG-style QA with context compression

## Setup

### 1. Install dependencies

Using uv:
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Or standard venv:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file:
```bash
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
SCALEDOWN_API_KEY=your_scaledown_key
```

## Running Experiments

### Quick test (5-10 examples)
```bash
# Run single experiment
python3 run_benchmarks.py --experiment-id hotpotqa_baseline_openai

# Dry-run to validate config
python3 run_benchmarks.py --dry-run
```

### Full benchmark suite
```bash
# Run all experiments in config
python3 run_benchmarks.py

# Run with verbose logging
python3 run_benchmarks.py --verbose
```

### Experiment Configuration

Edit `config/experiments.yaml` to customize:
- **num_examples**: Number of samples per experiment (default: 10)
- **model**: Change model name or compression rate
- **pricing**: Override default pricing (per 1M tokens)
- **metrics**: Add/remove evaluation metrics
- **generation**: Optional model generation settings (under `model.generation`) — supported: `temperature`, `top_p`, `max_tokens` (forwarded to the model client)

Example:
```yaml
- id: my_experiment
  model:
    name: openai:gpt-4o
    mode: scaledown  # or 'baseline'
    pricing:  # Optional: override default pricing
      input_per_1m_tokens: 2.50
      output_per_1m_tokens: 10.00
    scaledown:
      rate: auto  # or 0.5 for 50% compression
      compression_model: gemini:2.5-flash  # Optional: model for compression (see supported models below)
  dataset:
    name: hotpotqa
    split: validation
    num_examples: 50  # omit or set to `null` to use the full split (all rows)
    # Note: `seed` is only used when `num_examples` is provided and smaller than the dataset size.
  task:
    name: rag_qa
  metrics:
    - exact_match
    - f1
```

#### ScaleDown Compression Models

When using `mode: scaledown`, ScaleDown uses a separate model for context compression. You can specify which model to use via `scaledown.compression_model`:

**Supported compression models**:
- **Gemini**: `gemini-2.5-flash` (default), `gemini-2.5-pro`, `gemini-2.5-flash-lite`, `gemini-2.0-flash`
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`

**Important**: The compression model is separate from the downstream generation model. For example, you can compress with `gemini-2.5-flash` and generate with `gpt-4.1`.

If `compression_model` is not specified, it defaults to `gemini-2.5-flash` (fast, cost-effective, widely supported).

**Example - compression with Gemini, generation with GPT:**
```yaml
- id: hotpotqa_scaledown_gpt41
  model:
    name: openai:gpt-4.1              # Downstream generation model (can be any supported model)
    mode: scaledown
    scaledown:
      rate: auto
      compression_model: gemini:2.5-flash  # Compression model (limited to supported models)
```

#### Default Model Pricing

The framework includes default pricing (per 1M tokens) for common models:
- **gpt-4o**: $2.50 input / $10.00 output
- **gpt-4o-mini**: $0.15 input / $0.60 output
- **gpt-4.1**: $2.00 input / $8.00 output
- **gpt-4.1-mini**: $0.40 input / $1.60 output
- **gpt-5-mini**: $0.25 input / $2.00 output
- **gpt-5.2**: $1.75 input / $14.00 output
- **gpt-5.2-pro**: $21.00 input / $168.00 output
- **gemini-2.5-flash**: $0.30 input / $2.50 output
- **gemini-2.5-flash-lite**: $0.10 input / $0.40 output
- **gemini-2.5-pro**: $1.25 input / $10.00 output

To use custom pricing, add the `pricing` field to your model config as shown above.

## Understanding Results

### Raw Results (`results/raw/`)

Per-example JSONL files with timestamp: `{experiment_id}_{timestamp}.jsonl`

Each line contains:
```json
{
  "id": "example_id",
  "model": "openai:gpt-4o",
  "pred": "model prediction",
  "gold": ["ground truth answer"],
  "input_tokens": 1250,
  "output_tokens": 8,
  "latency_ms": 1543.2,
  "cost_usd": 0.003205,
  "metrics": {"exact_match": 1.0, "f1": 1.0}
}
```

**Note**: `cost_usd` will be `null` if pricing is not configured for the model.

### Summary Statistics (`results/summaries/`)

Two files per experiment:

**CSV** (`{experiment_id}_{timestamp}_summary.csv`):
- Flat format for easy comparison
- Columns: experiment_id, model, n_examples, exact_match_mean, f1_mean, avg_latency_ms, avg_input_tokens, avg_output_tokens, total_cost_usd, avg_cost_usd, cost_p95_usd, compression_ratio

**JSON** (`{experiment_id}_{timestamp}_summary.json`):
- Nested format with full statistics
- Includes mean, std, min, max, p50, p95 for all metrics
- Example:
```json
{
  "experiment_id": "hotpotqa_baseline_openai_20251211_150028",
  "model": "openai:gpt-4o",
  "n_examples": 10,
  "metrics": {
    "exact_match": {"mean": 0.4, "std": 0.52, "p50": 0.0, "p95": 1.0},
    "f1": {"mean": 0.59, "std": 0.42, "p50": 0.6, "p95": 1.0}
  },
  "latency_ms": {"mean": 1135.0, "std": 495.0, "p50": 1199.0, "p95": 1859.5},
  "tokens": {"avg_input_tokens": 1327.6, "avg_output_tokens": 8.4},
  "cost": {
    "total_usd": 0.03356,
    "mean_usd": 0.003356,
    "std_usd": 0.000124,
    "p50_usd": 0.003312,
    "p95_usd": 0.003598
  }
}
```

**Note**: The `cost` field will be omitted if pricing is not configured.

### Key Metrics Explained

- **exact_match**: 1.0 if prediction exactly matches answer after normalization, else 0.0
- **f1**: Token-level F1 score (0.0-1.0), measuring token overlap between prediction and answer
- **avg_input_tokens**: Average input tokens sent to LLM (should be ~80-90% lower for ScaleDown)
- **avg_output_tokens**: Average tokens in model responses
- **avg_latency_ms**: Average end-to-end latency (includes compression time for ScaleDown)
- **avg_cost_usd**: Average cost per example in USD (based on token usage and model pricing)
- **total_cost_usd**: Total cost for all examples in the experiment
- **compression_ratio**: For ScaleDown only - compressed/original token ratio

### Comparing Baseline vs ScaleDown

Look for:
1. **Token reduction**: ScaleDown input tokens should be 80-90% lower
2. **Accuracy preservation**: Exact match and F1 should be similar (±5-10%)
3. **Latency trade-off**: ScaleDown adds compression overhead but may reduce generation time
4. **Cost savings**: ScaleDown should significantly reduce costs due to lower input token usage

## Project Structure

```
├── config/
│   └── experiments.yaml        # Experiment configurations
├── core/
│   ├── config.py              # Config dataclasses and YAML parsing
│   └── registry.py            # Component factories (models/datasets/tasks/metrics)
├── dataset/
│   ├── base.py                # Dataset ABC
│   └── hotpotqa.py            # HotpotQA loader
├── tasks/
│   ├── base.py                # Task ABC with normalization
│   └── rag_task.py            # RAG QA task with prompt formatting
├── models/
│   ├── base.py                # ModelClient interface
│   ├── openai_client.py       # OpenAI wrapper
│   ├── gemini_client.py       # Gemini wrapper
│   └── scaledown_client.py    # ScaleDown compression wrapper
├── metrics/
│   └── classification.py      # Exact match and F1 metrics
├── evaluation/
│   ├── runner.py              # Experiment execution engine
│   └── aggregation.py         # Results statistics computation
├── results/
│   ├── raw/                   # Per-example JSONL logs
│   └── summaries/             # Aggregated CSV/JSON summaries
└── run_benchmarks.py          # CLI entrypoint
```

## Extending the Framework

### Add a new dataset
1. Create `dataset/my_dataset.py` implementing `Dataset` ABC
2. Register in `core/registry.py`: `DATASET_REGISTRY["my_dataset"] = MyDataset`
3. Add to `config/experiments.yaml`

### Add a new metric
1. Create function in `metrics/` returning `{"metric_name": float}`
2. Register in `core/registry.py`: `METRIC_REGISTRY["my_metric"] = my_metric_fn`
3. Add to experiment config metrics list

### Add a new model
1. Create client in `models/` implementing `ModelClient` interface
2. Register in `core/registry.py`: `MODEL_REGISTRY["provider"] = MyClient`
3. Use format `provider:model_name` in config
