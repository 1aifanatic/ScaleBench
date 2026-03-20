## Testing Guide

Here are different ways to test the implementation:

### **1. Quick Validation Test (30 seconds)**

Verify all components work together:

```bash
cd scaledown-benchmarking

# Set required environment variables or create .env file
export SCALEDOWN_API_KEY="your_api_key_here"
export SCALEDOWN_API_URL="https://api.scaledown.xyz/compress/raw"

# Run a minimal test (2 users, 10 seconds)
locust -f load_testing/locustfile.py --headless --users=2 --spawn-rate=1 --run-time=10s
```

**Expected output:**
- Config loaded successfully
- DataSampler initialized with examples
- 2 users spawn and make requests
- Success/failure stats displayed
- Custom metrics exported to summaries

---

### **2. Web UI Mode Test (Interactive)**

Test with real-time monitoring:

```bash
# Start Locust with web UI
locust -f load_testing/locustfile.py

# Open browser to http://localhost:8089
# Configure:
# - Number of users: 5
# - Spawn rate: 1
# - Host: (leave empty as it's auto-filled from config)
# Click "Start swarming"
```

**What to verify:**
- Web UI loads correctly
- Can start/stop test from UI
- Charts update in real-time
- Statistics tab shows request metrics
- Failures tab (if any errors occur)
- Can download CSV/HTML reports from UI

---

### **3. Steady Load Test (5 minutes)**

Simulate realistic steady load:

```bash
locust -f load_testing/locustfile.py --headless \
  --users=10 --spawn-rate=2 --run-time=5m \
  --csv=load_testing/results/steady_test --html=load_testing/results/steady_test.html
```

**What to verify:**
- 10 users spawn at rate of 2/second
- Steady request rate maintained for 5 minutes
- Success rate >95% (if API is healthy)
- CSV files created: `results/steady_test_stats.csv`, `results/steady_test_failures.csv`
- HTML report created: `results/steady_test.html`
- Custom metrics JSON/CSV in summaries

---

### **4. Ramp-Up Test**

Test gradual load increase:

```bash
locust -f load_testing/locustfile.py --headless \
  --users=50 --spawn-rate=5 --run-time=3m
```

**What to verify:**
- Users ramp up gradually (50 users / 5 per second = 10 seconds to peak)
- System handles increasing load
- Latency remains stable as users increase
- No significant increase in failures during ramp-up

---

### **5. Spike Test**

Test sudden load burst:

```bash
locust -f load_testing/locustfile.py --headless \
  --users=100 --spawn-rate=50 --run-time=2m
```

**What to verify:**
- Rapid user spawn (100 users in 2 seconds)
- System handles spike without crashing
- Error rate and latency under stress
- Recovery after spike completes

---

### **6. Multiple Compression Configs Test**

Test with multiple rates and models:

```bash
# First, edit load_testing/config.yaml:
# compression:
#   rates:
#     - auto
#     - 0.5
#     - 0.7
#   models:
#     - gemini-2.5-flash
#     - gpt-4o

locust -f load_testing/locustfile.py --headless \
  --users=20 --spawn-rate=5 --run-time=2m
```

**What to verify:**
- Requests use different compression rates randomly
- Requests use different compression models randomly
- Locust stats show separate entries for each rate/model combo
- Custom metrics track all combinations

---

### **7. Dataset Integration Test**

Test with different dataset configurations:

```bash
# Edit load_testing/config.yaml to use different dataset settings:
# data:
#   dataset: hotpotqa
#   split: validation
#   num_samples: 50  # Try different sample sizes

locust -f load_testing/locustfile.py --headless \
  --users=10 --spawn-rate=5 --run-time=1m
```

**What to verify:**
- Different num_samples loads correctly
- Different splits work (if available)
- Context length range calculated correctly
- Random sampling provides variety

---

## Expected Results Summary

**Successful test indicators:**
- No Python errors or crashes
- Users spawn and make requests
- Success rate >95% (with valid API key)
- Locust web UI/stats display correctly
- Custom metrics files created in summaries
- Locust CSV/HTML reports generated (if flags used)
- Console shows test summary on completion

**Key metrics to check:**
- **Total requests**: Should match users × duration / wait_time
- **Success rate**: >95% indicates healthy API
- **Avg latency**: Depends on API, typically 200-2000ms
- **P95 latency**: Should be <2x average latency
- **Compression ratio**: Typically 0.1-0.3 (70-90% compression)
- **Token savings**: Positive number showing tokens saved

---

## Troubleshooting

**Issue**: "No module named 'locust'"  
**Solution**: `pip install locust>=2.30.0`

**Issue**: "No examples loaded from dataset"  
**Solution**: Check dataset name in config.yaml, verify it's registered in DATASET_REGISTRY

**Issue**: "SCALEDOWN_API_KEY not found"  
**Solution**: Set environment variable or update config.yaml with actual key

**Issue**: "All requests failing (403 Forbidden)"  
**Solution**: Invalid API key - check your SCALEDOWN_API_KEY

**Issue**: "High latency (>5000ms)"  
**Solution**: API may be overloaded, reduce number of users or check API status

**Issue**: "Compression ratio shows 0.0"  
**Solution**: All requests failed - check API connectivity and credentials
