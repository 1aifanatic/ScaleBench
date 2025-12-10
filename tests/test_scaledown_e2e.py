"""End-to-end test for ScaleDown client with comprehensive output logging.

This test uses OpenAI (to avoid Gemini rate limits) and captures all metrics
to a text file for verification.

Run with: python tests/test_scaledown_e2e.py
"""
import os
import sys
from datetime import datetime
import wikipedia

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base import GenerationInput, PricingInfo
from models.openai_client import OpenAIClient
from models.scaledown_client import ScaleDownClient


def write_section(f, title):
    """Write a section header to the output file."""
    f.write("\n" + "=" * 80 + "\n")
    f.write(f"{title}\n")
    f.write("=" * 80 + "\n\n")


def main():
    # Check API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        return
    
    if not os.getenv("SCALEDOWN_API_KEY"):
        print("Error: SCALEDOWN_API_KEY not set")
        return
    
    # Create output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"tests/scaledown_e2e_output_{timestamp}.txt"
    
    print(f"Running end-to-end ScaleDown test...")
    print(f"Output will be written to: {output_file}")

    def wikipedia_context():
        """Fetch Wikipedia content for testing."""
        page = wikipedia.page('Taylor Swift', auto_suggest=False)
        return page.content
    
    with open(output_file, 'w') as f:
        f.write(f"ScaleDown End-to-End Test\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Base Model: OpenAI gpt-4o\n")
        
        # Test context - Wikipedia-like content
        test_context = wikipedia_context().strip()
        
        # Test question
        test_question = "Who is Taylor's brother?"
        
        write_section(f, "TEST CONFIGURATION")
        f.write(f"Context length: {len(test_context)} characters\n")
        f.write(f"Question: {test_question}\n")
        
        # Set up pricing for cost calculation
        pricing = PricingInfo(
            input_per_1k_tokens=0.0025,  # gpt-4o
            output_per_1k_tokens=0.01,
        )
        
        # Test 1: Baseline (no compression)
        write_section(f, "TEST 1: BASELINE (No Compression)")
        f.write("Creating OpenAI client without compression...\n\n")
        
        baseline_client = OpenAIClient(
            model_name="gpt-4o",
            pricing=pricing
        )
        
        baseline_input = GenerationInput(
            context=test_context,
            user_prompt=test_question,
            temperature=0.7,
        )
        
        f.write("Generating response...\n")
        baseline_output = baseline_client.generate(baseline_input)
        
        f.write(f"\nRESPONSE:\n{baseline_output.text}\n\n")
        f.write(f"METRICS:\n")
        f.write(f"  Input tokens: {baseline_output.input_tokens}\n")
        f.write(f"  Output tokens: {baseline_output.output_tokens}\n")
        f.write(f"  Total tokens: {baseline_output.input_tokens + baseline_output.output_tokens}\n")
        f.write(f"  Latency: {baseline_output.latency_ms:.0f}ms\n")
        f.write(f"  Cost: ${baseline_output.cost_usd:.6f}\n")
        
        # Test 2: ScaleDown with auto compression
        write_section(f, "TEST 2: SCALEDOWN (Auto Compression)")
        f.write("Creating ScaleDown client wrapping OpenAI...\n")
        f.write("Compression rate: auto\n\n")
        
        scaledown_client = ScaleDownClient(
            base_client=baseline_client,
            compression_rate="auto"
        )
        
        scaledown_input = GenerationInput(
            context=test_context,
            user_prompt=test_question,
            temperature=0.7,
        )
        
        f.write("Compressing context and generating response...\n")
        scaledown_output = scaledown_client.generate(scaledown_input)
        
        f.write(f"\nRESPONSE:\n{scaledown_output.text}\n\n")
        f.write(f"METRICS:\n")
        f.write(f"  Input tokens (compressed): {scaledown_output.input_tokens}\n")
        f.write(f"  Output tokens: {scaledown_output.output_tokens}\n")
        f.write(f"  Total tokens: {scaledown_output.input_tokens + scaledown_output.output_tokens}\n")
        f.write(f"  Total latency: {scaledown_output.latency_ms:.0f}ms\n")
        f.write(f"  Cost: ${scaledown_output.cost_usd:.6f}\n\n")
        
        f.write(f"COMPRESSION DETAILS:\n")
        metadata = scaledown_output.provider_metadata
        f.write(f"  Original tokens: {metadata.get('original_prompt_tokens', 'N/A')}\n")
        f.write(f"  Compressed tokens: {metadata.get('compressed_prompt_tokens', 'N/A')}\n")
        f.write(f"  Compression ratio: {metadata.get('compression_ratio', 0):.2%}\n")
        f.write(f"  Compression latency: {metadata.get('compression_latency_ms', 0):.0f}ms\n")
        f.write(f"  Generation latency: {metadata.get('generation_latency_ms', 0):.0f}ms\n")
        f.write(f"  ScaleDown rate: {metadata.get('scaledown_rate', 'N/A')}\n")
        f.write(f"  Base provider: {metadata.get('base_provider', 'N/A')}\n")
        f.write(f"  Base model: {metadata.get('base_model', 'N/A')}\n")
        
        # Test 3: ScaleDown with 0.5 compression rate
        write_section(f, "TEST 3: SCALEDOWN (50% Compression)")
        f.write("Creating ScaleDown client with compression_rate=0.5...\n\n")
        
        scaledown_50_client = ScaleDownClient(
            base_client=baseline_client,
            compression_rate=0.5
        )
        
        scaledown_50_input = GenerationInput(
            context=test_context,
            user_prompt=test_question,
            temperature=0.7,
        )
        
        f.write("Compressing context and generating response...\n")
        scaledown_50_output = scaledown_50_client.generate(scaledown_50_input)
        
        f.write(f"\nRESPONSE:\n{scaledown_50_output.text}\n\n")
        f.write(f"METRICS:\n")
        f.write(f"  Input tokens (compressed): {scaledown_50_output.input_tokens}\n")
        f.write(f"  Output tokens: {scaledown_50_output.output_tokens}\n")
        f.write(f"  Total latency: {scaledown_50_output.latency_ms:.0f}ms\n")
        f.write(f"  Cost: ${scaledown_50_output.cost_usd:.6f}\n\n")
        
        f.write(f"COMPRESSION DETAILS:\n")
        metadata_50 = scaledown_50_output.provider_metadata
        f.write(f"  Original tokens: {metadata_50.get('original_prompt_tokens', 'N/A')}\n")
        f.write(f"  Compressed tokens: {metadata_50.get('compressed_prompt_tokens', 'N/A')}\n")
        f.write(f"  Compression ratio: {metadata_50.get('compression_ratio', 0):.2%}\n")
        f.write(f"  Compression latency: {metadata_50.get('compression_latency_ms', 0):.0f}ms\n")
        
        # Comparison
        write_section(f, "COMPARISON SUMMARY")
        
        token_savings_auto = baseline_output.input_tokens - scaledown_output.input_tokens
        token_savings_50 = baseline_output.input_tokens - scaledown_50_output.input_tokens
        cost_savings_auto = baseline_output.cost_usd - scaledown_output.cost_usd
        cost_savings_50 = baseline_output.cost_usd - scaledown_50_output.cost_usd
        
        f.write("Baseline vs ScaleDown (auto):\n")
        f.write(f"  Token savings: {token_savings_auto} ({token_savings_auto/baseline_output.input_tokens*100:.1f}%)\n")
        f.write(f"  Cost savings: ${cost_savings_auto:.6f} ({cost_savings_auto/baseline_output.cost_usd*100:.1f}%)\n")
        f.write(f"  Latency difference: {scaledown_output.latency_ms - baseline_output.latency_ms:.0f}ms\n\n")
        
        f.write("Baseline vs ScaleDown (50%):\n")
        f.write(f"  Token savings: {token_savings_50} ({token_savings_50/baseline_output.input_tokens*100:.1f}%)\n")
        f.write(f"  Cost savings: ${cost_savings_50:.6f} ({cost_savings_50/baseline_output.cost_usd*100:.1f}%)\n")
        f.write(f"  Latency difference: {scaledown_50_output.latency_ms - baseline_output.latency_ms:.0f}ms\n\n")
        
        # Validation checks
        write_section(f, "VALIDATION CHECKS")
        
        checks = [
            ("Baseline response generated", baseline_output.text is not None and len(baseline_output.text) > 0),
            ("ScaleDown (auto) response generated", scaledown_output.text is not None and len(scaledown_output.text) > 0),
            ("ScaleDown (50%) response generated", scaledown_50_output.text is not None and len(scaledown_50_output.text) > 0),
            ("Compression reduced tokens (auto)", scaledown_output.input_tokens < baseline_output.input_tokens if 'compressed_prompt_tokens' in metadata else True),
            ("Compression reduced tokens (50%)", scaledown_50_output.input_tokens < baseline_output.input_tokens if 'compressed_prompt_tokens' in metadata_50 else True),
            ("Compression ratio valid (auto)", 0 < metadata.get('compression_ratio', 0) < 1 if 'compression_ratio' in metadata else True),
            ("Compression ratio valid (50%)", 0 < metadata_50.get('compression_ratio', 0) < 1 if 'compression_ratio' in metadata_50 else True),
            ("Pricing calculated (baseline)", baseline_output.cost_usd > 0),
            ("Pricing calculated (ScaleDown auto)", scaledown_output.cost_usd > 0),
            ("Pricing calculated (ScaleDown 50%)", scaledown_50_output.cost_usd > 0),
            ("ScaleDown cheaper than baseline (auto)", scaledown_output.cost_usd <= baseline_output.cost_usd),
            ("ScaleDown cheaper than baseline (50%)", scaledown_50_output.cost_usd <= baseline_output.cost_usd),
        ]
        
        all_passed = True
        for check_name, passed in checks:
            status = "✓ PASS" if passed else "✗ FAIL"
            f.write(f"{status}: {check_name}\n")
            if not passed:
                all_passed = False
        
        write_section(f, "TEST RESULT")
        if all_passed:
            f.write("✓ ALL TESTS PASSED\n")
            print(f"\n✓ All tests passed! Output saved to: {output_file}")
        else:
            f.write("✗ SOME TESTS FAILED\n")
            print(f"\n✗ Some tests failed. Check output: {output_file}")
        
        f.write(f"\nTest completed at: {datetime.now().isoformat()}\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
