"""Standalone ScaleDown compression API client for load testing.

This module provides a stateless function to call the ScaleDown compression
API directly, extracted from ScaleDownClient._compress_context() but made
suitable for Locust load testing (stateless, no class dependencies).
"""
import logging
import time
from typing import Optional, Union, Dict, Any

import requests


logger = logging.getLogger(__name__)


def compress_context_api(
    context: str,
    prompt: str,
    compression_rate: Union[str, float],
    compression_model: str,
    api_key: str,
    api_url: str,
    timeout: int = 30,
    max_retries: int = 2,
    retry_delay: float = 2.0,
) -> Optional[Dict[str, Any]]:
    """Call ScaleDown compression API (stateless function for load testing).
    
    This function is extracted from ScaleDownClient._compress_context()
    and made stateless for use in Locust load testing scenarios.
    
    Args:
        context: Long context to compress
        prompt: User prompt/question
        compression_rate: Compression rate ("auto" or float 0.0-1.0)
        compression_model: Model to use for compression (e.g., "gemini-2.5-flash")
        api_key: ScaleDown API key
        api_url: ScaleDown API endpoint URL
        timeout: Request timeout in seconds (default: 30)
        max_retries: Maximum retry attempts (default: 2)
        retry_delay: Initial delay for exponential backoff (default: 2.0)
        
    Returns:
        Dictionary with compression results:
        {
            "compressed_prompt": str,
            "latency_ms": float,
            "metadata": {
                "compression_ratio": float,
                "original_prompt_tokens": int,
                "compressed_prompt_tokens": int,
                "compression_time_ms": float,
            },
            "success": bool,
            "error": Optional[str],
        }
        Returns None if compression fails after all retries.
        
    Raises:
        Does not raise exceptions; returns error info in result dict instead.
    """
    # Build request payload
    payload = {
        "context": context,
        "prompt": prompt,
        "model": compression_model,
        "scaledown": {
            "rate": compression_rate
        }
    }
    
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    # Retry logic
    last_exception = None
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            
            response = requests.post(
                api_url,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            response.raise_for_status()
            data = response.json()
            
            # Check if compression was successful
            if not data.get("successful", False):
                error_msg = f"ScaleDown API returned unsuccessful response: {data}"
                logger.error(error_msg)
                last_exception = RuntimeError(error_msg)
                
                if attempt < max_retries - 1:
                    delay = retry_delay * (2 ** attempt)
                    logger.warning(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                else:
                    # Final attempt failed
                    return {
                        "success": False,
                        "error": error_msg,
                        "latency_ms": latency_ms,
                    }
            
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
                },
                "success": True,
                "error": None,
            }
            
            logger.debug(
                f"Compressed context: {results['original_prompt_tokens']} → "
                f"{results['compressed_prompt_tokens']} tokens "
                f"(ratio: {results['compression_ratio']:.2%}) in {latency_ms:.0f}ms"
            )
            
            return compression_result
            
        except requests.exceptions.RequestException as e:
            last_exception = e
            latency_ms = (time.time() - start_time) * 1000 if 'start_time' in locals() else 0
            
            if attempt < max_retries - 1:
                delay = retry_delay * (2 ** attempt)
                logger.warning(
                    f"ScaleDown API error (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(f"ScaleDown API error after {max_retries} attempts: {e}")
                return {
                    "success": False,
                    "error": f"Request failed: {str(e)}",
                    "latency_ms": latency_ms,
                }
        
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid ScaleDown API response: {e}")
            latency_ms = (time.time() - start_time) * 1000 if 'start_time' in locals() else 0
            return {
                "success": False,
                "error": f"Invalid response: {str(e)}",
                "latency_ms": latency_ms,
            }
    
    # Should not reach here, but just in case
    return {
        "success": False,
        "error": f"Failed after {max_retries} attempts: {str(last_exception)}",
        "latency_ms": 0,
    }
