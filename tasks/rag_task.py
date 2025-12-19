"""RAG task (e.g. MS MARCO / HotpotQA RAG)."""
from typing import Dict, Any, Optional
from tasks.base import Task
from prompts.rag_prompts import RAG_PROMPTS, DEFAULT_RAG_PROMPT


class RAGTask(Task):
    """RAG-style QA task for multi-hop reasoning."""

    def __init__(self, name: str = "rag_qa", dataset_name: Optional[str] = None):
        """Initialize RAG task.
        
        Args:
            name: Task name.
            dataset_name: Optional dataset name to load specific prompts.
        """
        super().__init__(name)
        self.dataset_name = dataset_name
        self.system_prompt = self._get_system_prompt()

    def _get_system_prompt(self) -> str:
        """Get system prompt from prompts registry based on dataset name.
        
        Returns:
            System prompt string.
        """
        if not self.dataset_name:
            return DEFAULT_RAG_PROMPT

        # Extract base dataset name (e.g., 'hotpotqa:distractor' -> 'hotpotqa')
        base_dataset = self.dataset_name.split(":")[0].lower()
        
        return RAG_PROMPTS.get(base_dataset, DEFAULT_RAG_PROMPT)

    def build_prompt(self, example: Dict[str, Any], **kwargs) -> "GenerationInput":
        """Build GenerationInput with specific ordering: Instructions -> Context -> Question.
        
        Args:
            example: Example with 'context' and 'question' fields.
            **kwargs: Additional generation parameters.
            
        Returns:
            GenerationInput object.
        """
        from models.base import GenerationInput
        
        question = example.get("question", "")
        context = example.get("context", "")
        
        return GenerationInput(
            system_prompt=self.system_prompt,
            context=f"Context:\n{context}",
            user_prompt=f"Question: {question}\n\nAnswer:",
            **kwargs
        )

    def postprocess(self, output: str) -> str:
        """Extract answer from model output.
        
        Takes first non-empty line as the answer.
        
        Args:
            output: Raw model output text.
            
        Returns:
            Extracted answer string.
        """
        # Split by newlines and find first non-empty line
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line:
                return line
        
        # Fallback to full output if no lines found
        return output.strip()
