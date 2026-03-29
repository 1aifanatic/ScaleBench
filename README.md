[![PyPI version](https://badge.fury.io/py/scalebench.svg)](https://pypi.org/project/scalebench/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# ScaleBench

Open-source benchmarking framework for evaluating **context compression** on LLM workloads. Measure accuracy, latency, and cost when using ScaleDown compression versus baseline LLMs across a variety of QA, retrieval, and summarization tasks.

## Overview

ScaleBench runs structured experiments comparing a **baseline** (full context sent directly to the LLM) against a **ScaleDown** mode (context compressed before being sent), reporting:

- **Accuracy** — Exact Match, Token F1, BERTScore, ROUGE, Set metrics, Retrieval metrics
- **Latency** — End-to-end latency per example, compression overhead
- **Cost** — Token usage and USD cost per example and in aggregate

**Supported datasets:**
| Dataset | Task type |
|---|---|
| HotpotQA | Multi-hop QA |
| LongBench (4 tasks) | Long-context QA |
| ETHIC | Long-document list extraction |
| MS-MARCO v2.1 | Passage retrieval + QA |
| FinanceBench | Financial document QA |
| QMSum | Query-based meeting summarization |

**Supported models:** OpenAI (gpt-4o, gpt-4.1, gpt-5 variants), Gemini (2.5-flash, 2.5-pro, 2.0-flash), Claude, ScaleDown compression + summarization endpoints

**Supported metrics:** `exact_match`, `f1`, `bertscore`, `rouge1`, `rouge2`, `rougeL`, `set_exact_match`, `set_precision`, `set_recall`, `set_f1`, `retrieval_precision`, `retrieval_recall`, `retrieval_f1`

---

## Setup

### 1. Install dependencies

Using uv (recommended):
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

Copy `.env.example` to `.env` and fill in your values:
```bash
cp .env.example .env
```

Required keys depend on which experiments you run:
```bash
# Baseline experiments
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key

# ScaleDown compression mode
SCALEDOWN_API_KEY=your_scaledown_key

# ScaleDown summarization mode
SCALEDOWN_SUMMARIZE_URL=https://your-endpoint/summarization/abstractive

# Optional: LangSmith tracing
LANGSMITH_API_KEY=your_langsmith_key
```

---

## Running Experiments

All commands use `uv run` to ensure the correct environment is used:

### Quick start
```bash
# Dry-run to validate config without executing
uv run python scripts/run_benchmarks.py --dry-run

# Run a single experiment by ID
uv run python scripts/run_benchmarks.py --experiment-id hotpotqa_baseline_openai

# Run all experiments with a tag
uv run python scripts/run_benchmarks.py --tag qmsum
```

### Full benchmark suite
```bash
# Run all experiments in config (serial)
uv run python scripts/run_benchmarks.py

# Run with verbose logging
uv run python scripts/run_benchmarks.py --verbose
```

### Parallel execution
```bash
# Run multiple experiments concurrently (one thread per experiment)
uv run python scripts/run_benchmarks.py --parallel-experiments

# Combine with tag filtering
uv run python scripts/run_benchmarks.py --tag qmsum --parallel-experiments

# Force serial execution (overrides config default)
uv run python scripts/run_benchmarks.py --serial-experiments
```

### CLI flags reference

| Flag | Description |
|---|---|
| `--experiment-id <id>` | Run only the specified experiment |
| `--tag <tag>` | Run all experiments with the given tag |
| `--parallel-experiments` | Run experiments concurrently (one thread per experiment) |
| `--serial-experiments` | Run experiments one after another (default) |
| `--dry-run` | Validate config without running anything |
| `--verbose` | Enable debug logging |
| `--config <path>` | Path to config YAML (default: `config/experiments.yaml`) |
| `--mlflow` | Enable MLflow tracing (overrides config) |
| `--no-mlflow` | Disable MLflow tracing (overrides config) |
| `--mlflow-uri <uri>` | MLflow tracking URI (default: `http://localhost:5000`) |
| `--langsmith` | Enable LangSmith tracing (overrides config) |
| `--no-langsmith` | Disable LangSmith tracing (overrides config) |
| `--langsmith-project <name>` | LangSmith project name (default: `scalebench`) |
| `--upload-hf` | Upload results to HuggingFace after completion |
| `--no-upload-hf` | Disable HF upload (overrides config) |
| `--hf-token <token>` | HuggingFace token (falls back to `HF_TOKEN` env var) |

---

## Experiment Configuration

Edit `config/experiments.yaml` to define experiments. Example:

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
      compression_model: gemini-2.5-flash  # Optional: model for compression
  dataset:
    name: hotpotqa
    split: validation
    num_examples: 50  # omit or set to null to use the full split
    # Note: seed is only used when num_examples is smaller than the dataset size
  task:
    name: rag_qa
  metrics:
    - exact_match
    - f1
```

### ScaleDown Compression Models

When using `mode: scaledown`, ScaleDown uses a separate model for context compression. Specify via `scaledown.compression_model`:

**Supported compression models:**
- **Gemini**: `gemini-2.5-flash` (default), `gemini-2.5-pro`, `gemini-2.5-flash-lite`, `gemini-2.0-flash`
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`

The compression model is separate from the downstream generation model. For example, compress with `gemini-2.5-flash` and generate with `gpt-4.1`:

```yaml
- id: hotpotqa_scaledown_gpt41
  model:
    name: openai:gpt-4.1
    mode: scaledown
    scaledown:
      rate: auto
      compression_model: gemini-2.5-flash
```

### Default Model Pricing

The framework includes default pricing (per 1M tokens) for common models:
- **gpt-4o**: $2.50 input / $10.00 output
- **gpt-4o-mini**: $0.15 input / $0.60 output
- **gpt-4.1**: $2.00 input / $8.00 output
- **gpt-4.1-mini**: $0.40 input / $1.60 output
- **gpt-5-mini**: $0.25 input / $2.00 output
- **gemini-2.5-flash**: $0.30 input / $2.50 output
- **gemini-2.5-flash-lite**: $0.10 input / $0.40 output
- **gemini-2.5-pro**: $1.25 input / $10.00 output

To use custom pricing, add the `pricing` field to your model config as shown above.

---

## Understanding Results

### Raw Results (`results/raw/`)

Per-example JSONL files with timestamp: `{experiment_id}_{timestamp}.jsonl`

Each line contains:
```json
{
  "id": "example_id",
  "model": "openai:gpt-4o",
  "input_question": "What is the capital of France?",
  "input_context": "France is a country in Europe...",
  "pred": "Paris",
  "gold": ["Paris"],
  "input_tokens": 150,
  "output_tokens": 5,
  "latency_ms": 450.5,
  "cost_usd": 0.000375,
  "metrics": {"exact_match": 1.0, "f1": 1.0}
}
```

`input_context` contains the **compressed context** in ScaleDown mode, or the full context in baseline mode. `cost_usd` is `null` if pricing is not configured.

### Summary Statistics (`results/summaries/`)

Two files per experiment:

**CSV** (`{experiment_id}_{timestamp}_summary.csv`): flat format for easy comparison across runs.

**JSON** (`{experiment_id}_{timestamp}_summary.json`): nested format with full statistics:
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

### Key Metrics Explained

- **exact_match**: 1.0 if prediction exactly matches answer after normalization, else 0.0
- **f1**: Token-level F1 score (0.0–1.0), measuring token overlap between prediction and answer
- **avg_input_tokens**: Average input tokens sent to LLM (should be ~80–90% lower for ScaleDown)
- **avg_output_tokens**: Average tokens in model responses
- **avg_latency_ms**: Average end-to-end latency (includes compression time for ScaleDown)
- **avg_cost_usd**: Average cost per example in USD
- **total_cost_usd**: Total cost for all examples in the experiment
- **compression_ratio**: For ScaleDown only — compressed/original token ratio

### Comparing Baseline vs ScaleDown

Look for:
1. **Token reduction**: ScaleDown input tokens should be 80–90% lower
2. **Accuracy preservation**: Exact match and F1 should be similar (±5–10%)
3. **Latency trade-off**: ScaleDown adds compression overhead but may reduce generation time
4. **Cost savings**: ScaleDown should significantly reduce costs due to lower input token usage

---

## LongBench Dataset

LongBench is a benchmark for long-context evaluation with documents up to 40k+ tokens. ScaleBench supports **4 English QA tasks**:

- **narrativeqa**: Question answering over long narratives (stories, novels)
- **qasper**: Question answering over scientific papers
- **multifieldqa_en**: Multi-domain question answering
- **2wikimqa**: Multi-hop questions requiring information from multiple sources

### Single-Task Evaluation

```yaml
- id: longbench_narrativeqa_baseline_openai
  model:
    name: openai:gpt-4o
    mode: baseline
  dataset:
    name: longbench
    split: test
    task: narrativeqa  # One of: narrativeqa, qasper, multifieldqa_en, 2wikimqa
    num_examples: 20
    seed: 42
  task:
    name: rag_qa
  metrics:
    - f1
    - bertscore
```

### Multi-Task Evaluation

Evaluate all 4 QA tasks in a single experiment by setting `task: null`:

```yaml
- id: longbench_all_baseline_openai
  model:
    name: openai:gpt-4o
    mode: baseline
  dataset:
    name: longbench
    split: test
    task: null  # Runs all 4 QA tasks automatically
    num_examples: 20  # Per task (80 total examples)
    seed: 42
  task:
    name: rag_qa
  metrics:
    - f1
    - bertscore
```

**Multi-task results structure:**
- Separate raw JSONL files for each task: `{exp_id}_narrativeqa_{timestamp}.jsonl`, etc.
- Summary CSV includes overall row + per-task breakdown
- Summary JSON includes nested `per_task` structure with individual metrics

### BERTScore Metric

BERTScore evaluates **semantic similarity** using contextual embeddings, providing more nuanced evaluation than token-level metrics.

**Returns three scores:**
- `bertscore_precision`: How much of the prediction is semantically relevant
- `bertscore_recall`: How much of the reference is covered by the prediction
- `bertscore_f1`: Harmonic mean of precision and recall

**Model download**: On first use, BERTScore downloads `distilbert-base-uncased` (~268MB), cached for future runs.

---

## MLflow Integration

Optional MLflow integration provides experiment tracking and tracing.

### Quick Start

1. **Start MLflow server** (in a separate terminal):
   ```bash
   mlflow server --host 127.0.0.1 --port 5000 --allowed-hosts "*"
   ```
   > **Note**: MLflow 3.5.0+ validates Host headers. Use `--allowed-hosts "*"` for local development.

2. **Run benchmarks with MLflow enabled**:
   ```bash
   uv run python scripts/run_benchmarks.py --mlflow
   ```

3. **View results** at http://127.0.0.1:5000

### Configuration

Enable via `config/experiments.yaml`:
```yaml
mlflow:
  enabled: true
  tracking_uri: "http://127.0.0.1:5000"
  experiment_name: "scalebench"
```

### What Gets Logged

- **Traces**: Per-example execution with full prompts/contexts, auto-traced OpenAI and Gemini calls, ScaleDown compression spans
- **Parameters**: Experiment ID, dataset, model, mode, compression rate
- **Metrics**: Aggregated F1, exact match, latency, token counts, cost, compression ratio
- **Artifacts**: Raw JSONL result files, summary JSON/CSV

### Troubleshooting MLflow

**403 Error** — MLflow security middleware (added in v3.5.0+) is blocking the request. Restart with:
```bash
mlflow server --host 127.0.0.1 --port 5000 --allowed-hosts "localhost,127.0.0.1"
```

**404 Error: Experiment not found** — The experiment name in your config doesn't exist yet; the code creates it automatically on first run.

---

## LangSmith Integration

Optional LangSmith integration provides per-example tracing with full prompt visibility.

### Quick Start

1. **Get a LangSmith API key** at [smith.langchain.com](https://smith.langchain.com)
2. **Add to `.env`**: `LANGSMITH_API_KEY=your_key`
3. **Run with LangSmith enabled**:
   ```bash
   uv run python scripts/run_benchmarks.py --langsmith --langsmith-project MyProject
   ```

### What Gets Traced

Each benchmark run produces a trace hierarchy:
```
experiment run
  └─ example_<id>
       ├─ llm_<model>          (raw model call with full prompts)
       └─ feedback scores      (rouge1, f1, exact_match, etc.)
```

Per-example data captured: system/user prompts, context length, prediction, gold answer, token counts, latency, cost, metric scores.

---

## Project Structure

```
scalebench/
├── config/
│   └── experiments.yaml        # Experiment configurations
├── scripts/
│   └── run_benchmarks.py       # CLI entrypoint
├── src/
│   ├── core/
│   │   ├── config.py           # Config dataclasses and YAML parsing
│   │   └── registry.py         # Component factories (models/datasets/tasks/metrics)
│   ├── dataset/
│   │   ├── base.py             # Dataset ABC
│   │   ├── hotpotqa.py         # HotpotQA loader
│   │   ├── longbench.py        # LongBench loader (4 QA tasks)
│   │   ├── ethic.py            # ETHIC loader (list extraction)
│   │   ├── msmarco.py          # MS-MARCO v2.1 loader
│   │   ├── financebench.py     # FinanceBench loader
│   │   └── qmsum.py            # QMSum loader
│   ├── tasks/
│   │   ├── base.py             # Task ABC
│   │   ├── rag_task.py         # RAG QA task
│   │   ├── retrieval_task.py   # Retrieval task
│   │   └── summarization_task.py # Summarization task
│   ├── models/
│   │   ├── base.py             # ModelClient interface
│   │   ├── openai_client.py    # OpenAI wrapper
│   │   ├── gemini_client.py    # Gemini wrapper
│   │   ├── claude_client.py    # Claude wrapper
│   │   ├── scaledown_client.py # ScaleDown compression wrapper
│   │   └── scaledown_summarize_client.py # ScaleDown summarization client
│   ├── metrics/
│   │   ├── classification.py   # Exact match and F1
│   │   ├── generation.py       # BERTScore
│   │   ├── rouge_metrics.py    # ROUGE metrics
│   │   ├── set_metrics.py      # Set-based metrics (list outputs)
│   │   └── retrieval_metrics.py # Retrieval metrics
│   ├── evaluation/
│   │   ├── runner.py           # Experiment execution engine
│   │   └── aggregation.py      # Results statistics computation
│   ├── utils/
│   │   ├── mlflow_utils.py     # MLflow integration
│   │   ├── langsmith_utils.py  # LangSmith integration
│   │   ├── parallel.py         # Concurrent API request processing
│   │   └── cost.py             # Cost calculation
│   ├── huggingface/
│   │   └── uploader.py         # HuggingFace dataset upload
│   ├── hf_space/
│   │   └── app.py              # Gradio leaderboard UI
│   ├── load_testing/           # Locust-based load testing
│   ├── prompts/                # Prompt templates
│   └── settings.py             # Environment configuration
├── tests/                      # Test suite
├── .env.example                # Environment variable template
├── pyproject.toml
└── requirements.txt
```

---

## Extending the Framework

### Add a new dataset
1. Create `src/dataset/my_dataset.py` implementing the `Dataset` ABC (see `hotpotqa.py` as an example)
2. Register in `src/core/registry.py`: `DATASET_REGISTRY["my_dataset"] = MyDataset`
3. Add experiments to `config/experiments.yaml`

### Add a new metric
1. Create a function in `src/metrics/` returning `{"metric_name": float}`
2. Register in `src/core/registry.py`: `METRIC_REGISTRY["my_metric"] = my_metric_fn`
3. Add to experiment config `metrics` list

### Add a new model
1. Create a client in `src/models/` implementing the `ModelClient` interface
2. Register in `src/core/registry.py`: `MODEL_REGISTRY["provider"] = MyClient`
3. Use format `provider:model_name` in experiment config

---

## Load Testing

The framework includes a load testing module for evaluating ScaleDown's compression API under concurrent load.

### Quick Start

```bash
# Run with web UI
locust -f src/load_testing/locustfile.py

# Run headless
locust -f src/load_testing/locustfile.py --headless --users=50 --spawn-rate=5 --run-time=5m
```

### Features

- Realistic load testing using actual dataset examples (HotpotQA, etc.)
- Multiple compression configs (different rates and models)
- Custom metrics: compression ratios, token savings, latency
- Comprehensive outputs: Locust built-ins + custom JSON/CSV exports

**Web UI**: http://localhost:8089 (when running without `--headless`)

See [load_testing/README.md](src/load_testing/README.md) for full documentation.

---

## HuggingFace Leaderboard Integration

Upload and visualize benchmark results on HuggingFace.

### Usage

```bash
# Upload results during a benchmark run
uv run python scripts/run_benchmarks.py --upload-hf

# Upload existing results
python src/huggingface/upload_to_hf.py --all
python src/huggingface/upload_to_hf.py --file results/summaries/*hotpotqa*.json
```

### Architecture

```
Local Benchmarks ──▶ HF Dataset ──▶ HF Space (Gradio UI)
    └── src/huggingface/   └── Private     └── src/hf_space/
        uploader.py            Storage          app.py
```

---

## Contributing

Contributions are welcome! Please open an issue or pull request. When adding a new dataset, model, or metric, follow the patterns in the existing implementations and add corresponding tests.

## License

[MIT](LICENSE) — © 2025 ScaleDown
