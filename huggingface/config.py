"""HuggingFace configuration for leaderboard integration."""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_ORG = os.getenv("HF_ORG", "scaledown")
HF_RESULTS_DATASET = os.getenv("HF_RESULTS_DATASET", "scaledown/benchmark-results")
HF_LEADERBOARD_SPACE = os.getenv("HF_LEADERBOARD_SPACE", "scaledown/compression-leaderboard")
