# Load Testing for ScaleDown API

Load testing module for evaluating ScaleDown's compression API performance under concurrent load.

## Quick Start

```bash
# Configure load test
vim load_testing/config.yaml

# Run load test with web UI
locust -f load_testing/locustfile.py --host=https://api.scaledown.xyz

# Run headless
locust -f load_testing/locustfile.py \
  --host=https://api.scaledown.xyz \
  --headless --users=50 --spawn-rate=5 --run-time=5m \
  --csv=results/loadtest --html=results/report.html
```

## Testing Scenarios

The "steady" scenario is the default behavior defined in locustfile.py.
Other scenarios can be achieved using Locust CLI flags:

Steady Load:
    Default behavior with constant number of users.
    
    Example:
        locust -f load_testing/locustfile.py --users=50 --spawn-rate=5
    
    This spawns 50 users at 5 users/second and maintains steady load.

Ramp Up:
    Gradually increase users over time using --run-time flag.
    
    Example:
        locust -f load_testing/locustfile.py --users=100 --spawn-rate=2 --run-time=10m
    
    This ramps up to 100 users at 2 users/second (takes 50 seconds to reach peak),
    then maintains load for the remaining time until 10 minutes total.

Spike Test:
    Rapid increase in users using high spawn-rate.
    
    Example:
        locust -f load_testing/locustfile.py --users=200 --spawn-rate=50 --run-time=5m
    
    This spikes to 200 users quickly (4 seconds) and maintains for 5 minutes.

Headless Mode:
    Run without web UI for automated testing.
    
    Example:
        locust -f load_testing/locustfile.py --headless --users=50 --spawn-rate=5 --run-time=5m

Web UI Mode:
    Interactive mode with real-time charts (default).
    
    Example:
        locust -f load_testing/locustfile.py
    
    Then open http://localhost:8089 in browser and configure test parameters.

Custom Outputs:
    Export results to CSV and HTML.
    
    Example:
        locust -f load_testing/locustfile.py --headless --users=50 --spawn-rate=5 \\
               --run-time=5m --csv=results --html=report.html

## Configuration

Edit `load_testing/config.yaml` to customize:

**ScaleDown API settings:**
- `api_url`: ScaleDown API endpoint
- `api_key`: API key (use env var `${SCALEDOWN_API_KEY}`)
- `timeout`: Request timeout in seconds
- `max_retries`: Number of retry attempts
- `retry_delay`: Initial retry delay (exponential backoff)

**Data sampling:**
- `dataset`: Dataset name (e.g., `hotpotqa`)
- `split`: Dataset split (`validation`, `test`, `train`)
- `num_samples`: Number of examples to pre-load
- `seed`: Random seed for reproducibility

**Load parameters:**
- `scenario`: Load pattern (`steady` - others via CLI)
- `default_users`: Default concurrent users
- `default_spawn_rate`: Default user spawn rate
- `default_duration`: Default test duration (seconds)

**Compression settings:**
- `rates`: List of compression rates (`auto`, `0.5`, `0.7`, etc.)
- `models`: List of compression models (`gemini-2.5-flash`, `gpt-4o`)

**Output options:**
- `raw_logs`: Enable per-request JSONL logging (default: false)
- `results_dir`: Output directory for results

## Results

**Locust built-in outputs:**
- Web UI: Real-time charts at http://localhost:8089
- CSV: `--csv=filename` creates `filename_stats.csv`, `filename_failures.csv`
- HTML: `--html=filename.html` creates comprehensive report

**Custom metrics (automatic):**
- Exported to `load_testing/results/summaries/` on test completion
- JSON: `load_test_metrics_<timestamp>.json` (machine-readable)
- CSV: `load_test_metrics_<timestamp>.csv` (spreadsheet-ready)

**Metrics tracked:**
- Total/successful/failed requests
- Success rate (%)
- Latency: mean, median, p95, p99 (ms)
- Compression ratio: mean, median, p95, p99
- Token counts: original, compressed, savings
- Context length statistics

## Interpreting Results

**Key metrics to watch:**
- **Success rate**: Should be >95% for healthy API
- **P95 latency**: 95% of requests complete within this time
- **Compression ratio**: Lower is better (more compression)
- **Token savings**: Average tokens saved per request

**Performance indicators:**
- High failure rate → API issues or rate limiting
- High latency → API overload or network issues
- Increasing latency over time → Resource exhaustion
- Low compression ratio → Context already concise

## Sample Test Scenarios

Sample test scenarios can be found in [`sample_scenarios.md`](sample_scenarios.md).

## Requirements

- Python 3.10+
- Locust 2.30.0+
- Valid ScaleDown API key
- Environment variables configured (see `config.yaml`)
