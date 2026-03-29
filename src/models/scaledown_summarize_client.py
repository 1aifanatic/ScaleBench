"""ScaleDown abstractive summarization client.

Standalone ModelClient that calls the ScaleDown /summarization/abstractive
endpoint directly. The endpoint takes a long text and an instruction (query)
and returns an abstractive summary — no downstream LLM call required.

API contract:
    POST /summarization/abstractive
    Body: {"text": str, "instructions": str, "max_tokens": int}
    Response: {"summary": str, "input_chars": int, "output_chars": int, "latency_ms": float}

GenerationInput mapping:
    inp.context      → "text"         (meeting transcript)
    inp.user_prompt  → "instructions" (query)
    inp.max_output_tokens or 512 → "max_tokens"
"""
import logging
import os
import time
from typing import Optional

import requests

from models.base import GenerationInput, ModelOutput, ModelClient, PricingInfo
from settings import settings


logger = logging.getLogger(__name__)

class ScaleDownSummarizeClient(ModelClient):
    """Standalone client for ScaleDown's abstractive summarization endpoint.

    Calls POST /summarization/abstractive directly. No downstream LLM is used;
    the ScaleDown service generates the summary.

    Args:
        model_name: Logical name for this client (e.g. "scaledown:summarize").
        api_key: ScaleDown API key (defaults to SCALEDOWN_API_KEY env var).
        max_retries: Number of retry attempts on transient errors (default: 2).
        retry_delay: Base delay for exponential backoff in seconds (default: 2.0).
        pricing: Optional PricingInfo for cost tracking.
    """

    def __init__(
        self,
        model_name: str = "scaledown:summarize",
        api_key: Optional[str] = None,
        max_retries: int = 2,
        retry_delay: float = 2.0,
        pricing: Optional[PricingInfo] = None,
    ):
        super().__init__(model_name=model_name, pricing=pricing)
        self.provider = "scaledown_summarize"
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.api_key = (
            api_key
            or os.getenv("SCALEDOWN_API_KEY")
            or settings.SCALEDOWN_API_KEY
        )
        if not self.api_key:
            raise ValueError("SCALEDOWN_API_KEY environment variable not set")

        self.api_url = settings.SCALEDOWN_SUMMARIZE_URL
        if not self.api_url:
            raise ValueError(
                "SCALEDOWN_SUMMARIZE_URL environment variable not set. "
                "Set it to your ScaleDown summarization endpoint URL."
            )

        logger.info(f"Initialized ScaleDownSummarizeClient: url={self.api_url}")

    def generate(self, inp: GenerationInput) -> ModelOutput:
        """Call the ScaleDown summarization endpoint.

        Args:
            inp: GenerationInput where context is the transcript and
                 user_prompt is the query/instruction.

        Returns:
            ModelOutput with the generated summary and latency/token metadata.
        """
        text = inp.context or inp.user_prompt
        instructions = inp.user_prompt if inp.context else ""
        max_tokens = inp.max_output_tokens or 512

        payload = {
            "text": text,
            "instructions": instructions,
            "max_tokens": max_tokens,
        }
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=60,
                )
                latency_ms = (time.time() - start_time) * 1000

                response.raise_for_status()
                data = response.json()

                summary = data.get("summary", "")
                input_chars = data.get("input_chars", len(text))
                output_chars = data.get("output_chars", len(summary))

                # Approximate token counts (char / 4)
                input_tokens = input_chars // 4
                output_tokens = output_chars // 4

                cost_usd = self._calculate_cost(input_tokens, output_tokens)

                logger.info(
                    f"ScaleDown summarization: {input_chars} chars → "
                    f"{output_chars} chars in {latency_ms:.0f}ms"
                )

                return ModelOutput(
                    text=summary,
                    latency_ms=data.get("latency_ms", latency_ms),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    provider_metadata={
                        "input_chars": input_chars,
                        "output_chars": output_chars,
                        "api_url": self.api_url,
                    },
                    input_question=inp.user_prompt,
                    input_context=inp.context,
                    input_system_prompt=inp.system_prompt,
                )

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"ScaleDown summarize error (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"ScaleDown summarize failed after {self.max_retries} attempts: {e}"
                    )
                    raise

            except (KeyError, ValueError) as e:
                logger.error(f"Invalid ScaleDown summarize response: {e}")
                raise

        raise RuntimeError("ScaleDown summarize: all retry attempts exhausted")

    def count_tokens(self, text: str) -> int:
        """Approximate token count using char/4 heuristic.

        Args:
            text: Input text.

        Returns:
            Approximate token count.
        """
        return len(text) // 4
