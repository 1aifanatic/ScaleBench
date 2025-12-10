# ScaleDown Benchmarking Framework

Clean, modular, reproducible benchmarking framework to evaluate ScaleDown's context compression layer on downstream LLM workloads.

# Setup

## Using uv
```
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Fallback: standard venv + pip
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```