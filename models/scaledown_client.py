"""ScaleDown client wrapper that injects context compression.

This module provides a wrapper around any ModelClient that adds ScaleDown's
context compression layer before generation. Key features:
- Wraps any ModelClient (OpenAI, Gemini, etc.)
- Calls ScaleDown API to compress long contexts
- Fallback to uncompressed baseline on errors
- Captures compression metrics and metadata
- Configurable compression rate
- MLflow tracing support for compression spans
"""
import logging
import os
import time
from typing import Optional, Union

import requests

from models.base import GenerationInput, ModelOutput, ModelClient
from settings import settings
import utils.mlflow_utils as mlflow_utils


logger = logging.getLogger(__name__)


# Models supported by ScaleDown API for compression (as of Dec 2025)
SCALEDOWN_SUPPORTED_MODELS = {
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gpt-4o",
    "gpt-4o-mini",
}


class ScaleDownClient(ModelClient):
    """Wrapper client that adds ScaleDown context compression.
    
    This client wraps any existing ModelClient and intercepts generation
    requests to apply context compression via the ScaleDown API before
    forwarding to the underlying model.
    
    Compression flow:
    1. If context is None → fallback to base client (log warning)
    2. Call ScaleDown /compress/raw endpoint
    3. Measure compression latency
    4. Extract compressed prompt from response
    5. Create new GenerationInput with compressed prompt
    6. Call base client's generate()
    7. Combine results: total_latency = compression + generation
    
    Args:
        base_client: The underlying ModelClient to use for generation
        compression_rate: Compression rate ("auto" or float between 0 and 1)
        api_key: ScaleDown API key (if None, uses SCALEDOWN_API_KEY env var)
        max_retries: Maximum retry attempts for compression (default: 2)
        retry_delay: Initial delay for exponential backoff (default: 2.0s)
        fallback_on_error: Whether to fallback to uncompressed on errors (default: True)
        pricing: Optional pricing override (if None, uses base_client.pricing)
        compression_model: Model to use for ScaleDown compression (if None, defaults to gemini-2.5-flash)
    
    Environment Variables:
        SCALEDOWN_API_KEY: Required API key for ScaleDown
        SCALEDOWN_API_URL: ScaleDown API endpoint (default: https://api.scaledown.xyz/compress/raw)
    """
    
    def __init__(
        self,
        base_client: ModelClient,
        compression_rate: Union[str, float] = "auto",
        api_key: Optional[str] = None,
        max_retries: int = 2,
        retry_delay: float = 2.0,
        fallback_on_error: bool = True,
        pricing: Optional['PricingInfo'] = None,
        compression_model: Optional[str] = None,
    ):
        # Initialize with base client's model name and pricing (allow override)
        super().__init__(
            model_name=base_client.model_name,
            pricing=pricing if pricing is not None else base_client.pricing,
        )
        
        self.base_client = base_client
        # Validate compression_rate
        if not (
            (isinstance(compression_rate, str) and compression_rate == "auto")
            or (isinstance(compression_rate, float) and 0.0 <= compression_rate <= 1.0)
        ):
            raise ValueError(
                f"compression_rate must be 'auto' or a float between 0 and 1, got {compression_rate!r}"
            )
        self.compression_rate = compression_rate
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.fallback_on_error = fallback_on_error
        self.provider = "scaledown"
        
        # Set compression model (default to gemini-2.5-flash if not provided)
        if compression_model:
            # Parse compression_model (handle "provider:model" format)
            if ":" in compression_model:
                parts = compression_model.split(":")
                self.compression_model_name = parts[1] if len(parts) > 1 else parts[0]
            else:
                self.compression_model_name = compression_model
            
            # Validate against supported models
            if self.compression_model_name not in SCALEDOWN_SUPPORTED_MODELS:
                logger.warning(
                    f"Compression model '{self.compression_model_name}' may not be supported by ScaleDown. "
                    f"Supported models: {SCALEDOWN_SUPPORTED_MODELS}"
                )
        else:
            # Default to gemini-2.5-flash (fast, cheap, widely supported)
            self.compression_model_name = "gemini-2.5-flash"
            logger.info(f"Using default compression model: {self.compression_model_name}")
        
        # Get API key and URL from settings or environment
        self.scaledown_api_key = api_key or os.getenv("SCALEDOWN_API_KEY") or settings.SCALEDOWN_API_KEY
        self.scaledown_api_url = os.getenv("SCALEDOWN_API_URL") or settings.SCALEDOWN_API_URL
        
        if not self.scaledown_api_key:
            raise ValueError("SCALEDOWN_API_KEY environment variable not set")
        
        logger.info(
            f"Initialized ScaleDownClient: compression_model={self.compression_model_name}, "
            f"generation_model={base_client.provider}/{base_client.model_name}, "
            f"compression_rate={compression_rate}"
        )
    
    def generate(self, inp: GenerationInput) -> ModelOutput:
        """Generate text with context compression.
        
        Args:
            inp: Generation input with prompt and context
        
        Returns:
            ModelOutput with generated text and compression metadata
        """
        # If no context, fallback to base client
        if not inp.context:
            logger.warning("No context provided, falling back to base client without compression")
            return self.base_client.generate(inp)
        
        # Try to compress context
        compression_result = self._compress_context(
            context=inp.context,
            prompt=inp.user_prompt,
        )
        
        # If compression failed and fallback is enabled, use base client
        if compression_result is None:
            if self.fallback_on_error:
                logger.warning("Compression failed, falling back to base client without compression")
                return self.base_client.generate(inp)
            else:
                raise RuntimeError("Context compression failed and fallback is disabled")
        
        # Extract compression results
        compressed_context = compression_result["compressed_prompt"]
        compression_latency_ms = compression_result["latency_ms"]
        compression_metadata = compression_result["metadata"]
        
        # Create new input with compressed context and original question
        compressed_input = GenerationInput(
            user_prompt=inp.user_prompt,
            context=compressed_context,
            system_prompt=inp.system_prompt,
            max_output_tokens=inp.max_output_tokens,
            temperature=inp.temperature,
            top_p=inp.top_p,
            stop=inp.stop,
        )
        
        # Generate with base client
        base_output = self.base_client.generate(compressed_input)
        
        # Combine latencies
        total_latency_ms = compression_latency_ms + base_output.latency_ms
        
        # Build enhanced provider metadata with compression stats
        provider_metadata = {
            "scaledown_rate": self.compression_rate,
            "compression_latency_ms": compression_latency_ms,
            "generation_latency_ms": base_output.latency_ms,
            "compression_ratio": compression_metadata["compression_ratio"],
            "original_prompt_tokens": compression_metadata["original_prompt_tokens"],
            "compressed_prompt_tokens": compression_metadata["compressed_prompt_tokens"],
            "base_provider": self.base_client.provider,
            "base_model": self.base_client.model_name,
        }
        
        # Merge with base client's provider metadata if present
        if base_output.provider_metadata:
            provider_metadata["base_provider_metadata"] = base_output.provider_metadata
        
        # Return enhanced output
        return ModelOutput(
            text=base_output.text,
            latency_ms=total_latency_ms,
            input_tokens=compression_metadata["compressed_prompt_tokens"],
            output_tokens=base_output.output_tokens,
            cost_usd=base_output.cost_usd,
            provider_metadata=provider_metadata,
            input_question=inp.user_prompt,
            input_context=compressed_context,
            input_system_prompt=inp.system_prompt,
        )
    
    def _compress_context(
        self,
        context: str,
        prompt: str,
    ) -> Optional[dict]:
        """Compress context using ScaleDown API.
        
        Args:
            context: Long context to compress
            prompt: User prompt/question
        
        Returns:
            Dictionary with compression results, or None if failed
        """
        # Start MLflow span for compression (if enabled)
        with mlflow_utils.start_span(name="scaledown_compress", span_type="TOOL") as span:
            # Set span inputs with actual context content for observability
            span.set_inputs({
                "context_length": len(context),
                "context": context, 
                "prompt": prompt,
                "compression_model": self.compression_model_name,
                "compression_rate": self.compression_rate,
            })
            
            payload = {
                "context": context,
                "prompt": prompt,
                "model": self.compression_model_name,
                "scaledown": {
                    "rate": self.compression_rate
                }
            }
            
            headers = {
                "x-api-key": self.scaledown_api_key,
                "Content-Type": "application/json"
            }
            
            # Retry logic

            for attempt in range(self.max_retries):
                try:
                    start_time = time.time()
                    
                    response = requests.post(
                        self.scaledown_api_url,
                        json=payload,
                        headers=headers,
                        timeout=30,
                    )
                    
                    latency_ms = (time.time() - start_time) * 1000
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    # Check if compression was successful
                    if not data.get("successful", False):
                        logger.error(f"ScaleDown API returned unsuccessful response: {data}")
                        if attempt < self.max_retries - 1:
                            delay = self.retry_delay * (2 ** attempt)
                            logger.warning(f"Retrying in {delay:.1f}s...")
                            time.sleep(delay)
                            continue
                        else:
                            return None
                    
                    # Extract compression results
                    results = data.get("results", {})
                    
                    compression_result = {
                        "compressed_prompt": results["compressed_prompt"],
                        "latency_ms": latency_ms,
                        "metadata": {
                            "compression_ratio": results["compression_ratio"],
                            "original_prompt_tokens": results["original_prompt_tokens"],
                            "compressed_prompt_tokens": results["compressed_prompt_tokens"],
                            "compression_time_ms": data.get("latency_ms", latency_ms),
                        }
                    }
                    
                    # Set span outputs for MLflow with full compressed content
                    span.set_outputs({
                        "compressed_context": results["compressed_prompt"], 
                        "compressed_context_length": len(results["compressed_prompt"]),
                        "compression_ratio": results["compression_ratio"],
                        "original_tokens": results["original_prompt_tokens"],
                        "compressed_tokens": results["compressed_prompt_tokens"],
                        "latency_ms": latency_ms,
                    })
                    
                    logger.info(
                        f"Compressed context: {results['original_prompt_tokens']} → "
                        f"{results['compressed_prompt_tokens']} tokens "
                        f"(ratio: {results['compression_ratio']:.2%}) in {latency_ms:.0f}ms"
                    )
                    
                    return compression_result
                    
                except requests.exceptions.RequestException as e:
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)
                        logger.warning(
                            f"ScaleDown API error (attempt {attempt + 1}/{self.max_retries}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"ScaleDown API error after {self.max_retries} attempts: {e}")
                
                except (KeyError, ValueError) as e:
                    logger.error(f"Invalid ScaleDown API response: {e}")
                    return None
            
            return None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using base client.
        
        Args:
            text: Input text to count tokens for
        
        Returns:
            Token count from base client
        """
        return self.base_client.count_tokens(text)
