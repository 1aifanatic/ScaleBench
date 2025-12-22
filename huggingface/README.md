# HuggingFace Integration Module

This module provides functionality to upload benchmark results to HuggingFace datasets for leaderboard display.

## Configuration

Set the following environment variables in `.env`:

```bash
HF_TOKEN=hf_xxx...  # HuggingFace token with write access
HF_ORG=scaledown
HF_RESULTS_DATASET=scaledown/benchmark-results
HF_LEADERBOARD_SPACE=scaledown/compression-leaderboard
```

## Usage

### Upload a Single Result

```python
from pathlib import Path
from huggingface.uploader import upload_result

# Upload result to HF dataset
success = upload_result(
    Path("results/summaries/experiment_summary.json")
)
```

### Upload Multiple Results

```python
from pathlib import Path
from huggingface.uploader import upload_results_batch

summary_files = list(Path("results/summaries").glob("*.json"))
stats = upload_results_batch(summary_files)

print(f"Uploaded: {stats['uploaded']}")
print(f"Skipped: {stats['skipped']}")
print(f"Failed: {stats['failed']}")
```

### Check Already Uploaded Results

```python
from huggingface.uploader import get_uploaded_results

uploaded_ids = get_uploaded_results()
print(f"Found {len(uploaded_ids)} results already uploaded")
```

### Using the ResultUploader Class

```python
from huggingface.uploader import ResultUploader

uploader = ResultUploader()

# Upload with custom options
uploader.upload_result(
    summary_path=Path("results/summaries/experiment.json"),
    force=True,  # Force re-upload even if exists
    skip_if_exists=False
)
```

## Schema Conversion

The module automatically converts benchmark summary JSONs to flat leaderboard rows:

```python
from huggingface.schema import summary_to_leaderboard_row
import json

with open("results/summaries/experiment.json") as f:
    summary = json.load(f)

row = summary_to_leaderboard_row(summary)
# row contains flattened metrics, latency, cost, tokens, etc.
```

### Dynamic Metric Handling

The schema converter **dynamically** extracts all metrics from your summary without hardcoding metric names:

- Supports any metrics: `exact_match`, `f1`, `bertscore_*`, custom metrics
- Automatically extracts mean and std for each metric
- Future-proof: new metrics work without code changes


## Design

- **Idempotent**: Automatically skips already-uploaded results
- **Dynamic**: Handles variable metrics without hardcoding
- **Robust**: Graceful error handling for missing fields
- **Metadata**: Preserves original summary for reference
