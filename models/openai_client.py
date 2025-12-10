"""OpenAI model client."""
import logging
import time
from typing import Optional

import tiktoken
from openai import OpenAI, RateLimitError, APIError, APIConnectionError

from models.base import ModelClient, GenerationInput, ModelOutput, PricingInfo

logger = logging.getLogger(__name__)


class OpenAIClient(ModelClient):
    """Client for OpenAI models using Chat Completions API."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        pricing: Optional[PricingInfo] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """Initialize OpenAI client.
        
        Args:
            model_name: OpenAI model name (e.g., "gpt-4o", "gpt-4o-mini").
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            pricing: Optional pricing information for cost calculation.
            max_retries: Maximum number of retries for failed requests.
            retry_delay: Initial delay in seconds for exponential backoff.
        """
        super().__init__(model_name, api_key, pricing)
        self.provider = "openai"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        
        # Initialize tiktoken encoder for token counting
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.info(f"Model {model_name} not found in tiktoken, using o200k_base encoding")
            self.encoding = tiktoken.get_encoding("o200k_base")

    def generate(self, inp: GenerationInput) -> ModelOutput:
        """Generate text using OpenAI Chat Completions API.
        
        Args:
            inp: GenerationInput with prompt and configuration.
            
        Returns:
            ModelOutput with generated text and metadata.
        """
        # Build messages
        messages = []
        
        # Add system message if provided
        if inp.system_prompt:
            messages.append({"role": "system", "content": inp.system_prompt})
        
        # Build user message (context + user_prompt)
        user_content = inp.user_prompt
        if inp.context:
            user_content = f"{inp.context}\n\n{inp.user_prompt}"
        
        messages.append({"role": "user", "content": user_content})
        
        # Build API parameters
        params = {
            "model": self.model_name,
            "messages": messages,
        }
        
        # Add optional parameters if provided
        if inp.temperature is not None:
            params["temperature"] = inp.temperature
        if inp.max_output_tokens is not None:
            params["max_tokens"] = inp.max_output_tokens
        if inp.top_p is not None:
            params["top_p"] = inp.top_p
        if inp.stop is not None:
            params["stop"] = inp.stop
        
        # Retry logic with exponential backoff
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(**params)
                latency_ms = (time.time() - start_time) * 1000
                
                # Extract text from response
                text = response.choices[0].message.content or ""
                
                # Extract token counts from usage
                input_tokens = response.usage.prompt_tokens if response.usage else 0
                output_tokens = response.usage.completion_tokens if response.usage else 0
                
                # Calculate cost if pricing configured
                cost_usd = self._calculate_cost(input_tokens, output_tokens)
                
                return ModelOutput(
                    text=text,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                )
                
            except RateLimitError as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                else:
                    logger.error(f"Rate limit exceeded after {self.max_retries} retries")
                    
            except (APIError, APIConnectionError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"API error, retrying in {delay}s (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"API error after {self.max_retries} retries: {e}")
            except Exception as e:
                # Unexpected exception: raise immediately
                logger.error(f"Unexpected error in OpenAIClient.generate: {e}", exc_info=True)
                raise
        # If all retries failed, raise the last exception
        if last_exception is not None:
            raise last_exception
        else:
            raise RuntimeError("OpenAIClient.generate failed with an unexpected error and no exception was captured.")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.
        
        Args:
            text: Input text to count tokens for.
            
        Returns:
            Token count.
        """
        return len(self.encoding.encode(text))
