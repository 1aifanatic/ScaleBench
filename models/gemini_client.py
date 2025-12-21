"""Gemini client implementation using google-genai SDK.

This module provides a client for Google's Gemini models with support for:
- Text generation with system instructions
- Token counting via API
- Temperature and output control
- Cost calculation (when pricing provided)
- Retry logic with exponential backoff
"""
import logging
import os
import time
from typing import Optional

from google import genai
from google.genai import types

from models.base import GenerationInput, ModelOutput, ModelClient, PricingInfo


logger = logging.getLogger(__name__)


class GeminiClient(ModelClient):
    """Client for Google Gemini models.
    
    Uses the google-genai SDK with the following features:
    - client.models.generate_content() for generation
    - client.models.count_tokens() for token counting
    - Configurable temperature, max_tokens, top_p, stop sequences
    - System instructions via GenerateContentConfig
    - Retry logic for rate limits and API errors
    - Cost calculation when pricing is provided
    
    Args:
        model_name: Gemini model name (e.g., "gemini-2.0-flash")
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Initial delay between retries in seconds (default: 2.0)
        pricing: Optional pricing information for cost calculation
        thinking_budget: Thinking budget for 2.5-flash models (default: 0 to disable)
    
    Environment Variables:
        GEMINI_API_KEY: Required API key for Gemini
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        pricing: Optional[PricingInfo] = None,
        thinking_budget: int = 0,
    ):
        super().__init__(model_name=model_name, pricing=pricing)
        self.provider = "gemini"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.thinking_budget = thinking_budget
        
        # Initialize Gemini client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        self.client = genai.Client(api_key=api_key)
        logger.info(f"Initialized GeminiClient with model: {model_name}")
    
    def generate(self, inp: GenerationInput) -> ModelOutput:
        """Generate text using Gemini API.
        
        Args:
            inp: Generation input with prompt, context, and optional parameters
        
        Returns:
            ModelOutput with generated text, tokens, latency, and cost
        
        Raises:
            Exception: If generation fails after all retries
        """
        # Build contents (user prompt + optional context)
        if inp.context:
            # Concatenate context and user prompt
            contents = f"{inp.context}\n\n{inp.user_prompt}"
        else:
            contents = inp.user_prompt
        
        # Build config with optional parameters
        config_kwargs = {}
        
        # System instruction
        if inp.system_prompt:
            config_kwargs["system_instruction"] = inp.system_prompt
        
        # Generation parameters (all optional)
        if inp.temperature is not None:
            config_kwargs["temperature"] = inp.temperature
        if inp.max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = inp.max_output_tokens
        if inp.top_p is not None:
            config_kwargs["top_p"] = inp.top_p
        if inp.stop is not None:
            config_kwargs["stop_sequences"] = inp.stop
        
        # Thinking budget for 2.5 models
        if "2.5" in self.model_name.lower():
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )
        
        config = types.GenerateContentConfig(**config_kwargs)
        
        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Extract text from response
                text = response.text
                
                # Extract token counts from usage_metadata
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
                
                # Calculate cost if pricing provided
                cost_usd = self._calculate_cost(input_tokens, output_tokens)
                
                # Build provider metadata
                provider_metadata = {
                    "finish_reason": (
                        response.candidates[0].finish_reason.name 
                        if response.candidates else "UNKNOWN"
                    ),
                    "model_used": self.model_name,
                    "usage_metadata": {
                        "prompt_token_count": input_tokens,
                        "candidates_token_count": output_tokens,
                        "total_token_count": response.usage_metadata.total_token_count,
                    },
                }
                
                # Add thinking tokens if present (for 2.5 models)
                if hasattr(response.usage_metadata, "thoughts_token_count"):
                    provider_metadata["usage_metadata"]["thoughts_token_count"] = (
                        response.usage_metadata.thoughts_token_count
                    )
                
                logger.info(
                    f"Generated {output_tokens} tokens in {latency_ms:.0f}ms "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                
                return ModelOutput(
                    text=text,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    provider_metadata=provider_metadata,
                    input_question=inp.user_prompt,
                    input_context=inp.context,
                    input_system_prompt=inp.system_prompt,
                )
            
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check if it's a retryable error
                is_retryable = any(
                    keyword in error_msg
                    for keyword in ["rate", "quota", "timeout", "unavailable", "503", "429"]
                )
                
                if is_retryable and attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Gemini API error (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Gemini API error after {attempt + 1} attempts: {e}")
                    raise
        
        raise RuntimeError("Failed to generate content after all retries")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using Gemini API.
        
        Args:
            text: Text to count tokens for
        
        Returns:
            Number of tokens in text
        """
        try:
            result = self.client.models.count_tokens(
                model=self.model_name,
                contents=text,
            )
            return result.total_tokens
        except Exception as e:
            logger.warning(f"Token counting failed: {e}. Using approximation (4 chars ≈ 1 token)")
            # Fallback: approximate as 4 characters per token
            return len(text) // 4
