"""Environment configuration for leaderboard Space."""
import os

from huggingface_hub import HfApi

# HuggingFace configuration
TOKEN = os.environ.get("HF_TOKEN")
RESULTS_DATASET = os.environ.get("HF_RESULTS_DATASET", "scaledown/benchmark-results")

# Cache path for downloaded results
CACHE_PATH = os.getenv("HF_HOME", ".")
RESULTS_CACHE_PATH = os.path.join(CACHE_PATH, "results-cache")

# API client
if TOKEN is not None:
    API = HfApi(token=TOKEN)
else:
    API = HfApi()
