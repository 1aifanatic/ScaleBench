"""Summarization task for query-based meeting summarization (QMSum)."""
from typing import Dict, Any, Optional

from models.base import GenerationInput
from tasks.base import Task

_SYSTEM_PROMPT_TEMPLATE = (
    "Answer the following query about the meeting transcript "
    "with a concise abstractive summary (2-5 sentences). "
    "Use only information from the transcript. "
    "Paraphrase and synthesize — do not copy verbatim. "
    "No headers or bullets.\n\nQuery: {question}"
)


class SummarizationTask(Task):
    """Task for abstractive summarization conditioned on a query.

    Designed for QMSum and similar query-based summarization benchmarks.

    Prompt structure:
    - Standard LLM (GPT etc.): query embedded in system prompt,
      transcript passed as user message context.
    - ScaleDown summarize endpoint: no system prompt; query goes in
      user_prompt (-> "instructions"), transcript in context (-> "text").

    Key differences from RAGTask:
    - postprocess() returns the full output (not just first line).
    - normalize() only lowercases — does NOT strip articles/punctuation,
      since ROUGE is n-gram-based and those tokens affect scores.
    """

    def __init__(
        self,
        name: str = "summarization",
        dataset_name: Optional[str] = None,
    ):
        super().__init__(name)
        self.dataset_name = dataset_name

    def build_prompt(
        self, example: Dict[str, Any], **kwargs
    ) -> GenerationInput:
        """Build GenerationInput for query-based meeting summarization.

        Args:
            example: Example with 'context' (transcript) and
                     'question' (query).
            model_mode: Passed via kwargs; if 'scaledown_summarize',
                        uses ScaleDown prompt structure (no system prompt).
            **kwargs: Remaining generation parameters forwarded to
                      GenerationInput.

        Returns:
            GenerationInput configured for the target client.
        """
        model_mode = kwargs.pop("model_mode", None)
        question = example.get("question", "")
        context = example.get("context", "")

        if model_mode == "scaledown_summarize":
            # ScaleDown endpoint maps context -> "text", user_prompt ->
            # "instructions". No system prompt needed.
            return GenerationInput(
                system_prompt=None,
                context=context,
                user_prompt=question,
                **kwargs,
            )

        # Standard LLM: query as system prompt, transcript as user message.
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(question=question)
        return GenerationInput(
            system_prompt=system_prompt,
            context=f"Meeting Transcript:\n{context}",
            user_prompt="Summarize the transcript to answer the query.",
            **kwargs,
        )

    def postprocess(self, output: str) -> str:
        """Return full model output as the summary (no truncation)."""
        return output.strip()

    def normalize(self, text: str) -> str:
        """Lowercase only — preserve punctuation/articles for ROUGE."""
        return text.lower().strip()
