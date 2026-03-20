---
title: ScaleDown Compression Leaderboard
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: false
license: mit
---

# ScaleDown Compression Benchmark Leaderboard

This Space displays benchmark results comparing **baseline** LLM inference vs **ScaleDown** context compression.

## Features

- **Dynamic metrics**: Automatically displays all metrics from uploaded results (exact_match, F1, BERTScore, etc.)
- **Filtering**: Filter by model, dataset, and mode (baseline/scaledown)
- **Comparison view**: Side-by-side baseline vs ScaleDown comparison with cost savings

## Data Source

Results are loaded from the private dataset: `scaledown/benchmark-results`

## Deploying Updates to HuggingFace Space

To push Space code changes to HuggingFace, run from the root directory:

```bash
# Deploy with default commit message
python deploy_space.py

# Deploy with custom message
python deploy_space.py --message "Fix: comparison tab reverse mapping"

# Use custom token if needed
python deploy_space.py --hf-token YOUR_TOKEN
```

This uploads the entire `hf_space/` directory to the live Space on HuggingFace Hub.

## Local Development

```bash
cd hf_space
pip install -r requirements.txt
python app.py
```
