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

## Configuration

See `config.yaml` for available options:
- ScaleDown API settings (URL, API key, timeout)
- Data sampling (dataset, split, number of samples)
- Load parameters (users, spawn rate, duration)
- Compression settings (rates, models)
- Output options (raw logs, results directory)

## Documentation

Full documentation TBD.
