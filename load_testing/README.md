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

See `config.yaml` for available options:
- ScaleDown API settings (URL, API key, timeout)
- Data sampling (dataset, split, number of samples)
- Load parameters (users, spawn rate, duration)
- Compression settings (rates, models)
- Output options (raw logs, results directory)

## Documentation

Full documentation TBD.
