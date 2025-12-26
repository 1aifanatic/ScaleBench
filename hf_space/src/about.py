"""About section content for leaderboard."""

TITLE = """<h1 align="center">ScaleDown Compression Benchmark</h1>"""

INTRODUCTION_TEXT = """
This leaderboard tracks benchmark results comparing **baseline** LLM performance vs **ScaleDown** context compression.

**Metrics displayed:** Quality metrics (exact_match, F1, BERTScore, etc.), latency, token usage, and cost.

**Modes:**
- **Baseline**: Standard LLM inference without compression
- **ScaleDown**: LLM inference with ScaleDown context compression

Select filters and columns below to explore results.
"""

ABOUT_TEXT = """
## Methodology

Benchmarks are run using the [scaledown-benchmarking](https://github.com/scaledown-team/scaledown-benchmarking) framework.

### Datasets
- **HotpotQA**: Multi-hop QA requiring reasoning across documents
- **MS MARCO**: Passage retrieval and QA
- **LongBench**: Long-context evaluation suite
- **ETHIC**: Ethics and reasoning benchmark

### Metrics
- **exact_match**: Exact string match accuracy
- **f1**: Token-level F1 score
- **bertscore_f1**: BERTScore F1 (semantic similarity)
- **latency_ms**: End-to-end inference time
- **cost_usd**: Estimated API cost

### Reproducibility
All experiments are deterministic with fixed seeds. Raw results and configurations are available in the results dataset.
"""

CITATION_BUTTON_LABEL = "Copy the following to cite these results"
CITATION_BUTTON_TEXT = r"""
@misc{scaledown-benchmark,
  title={ScaleDown Compression Benchmark},
  author={ScaleDown Team},
  year={2025},
  url={https://huggingface.co/spaces/scaledown/compression-leaderboard}
}
"""
