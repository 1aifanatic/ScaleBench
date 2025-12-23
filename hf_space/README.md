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

## Local Development

```bash
cd hf_space
pip install -r requirements.txt
python app.py
```
