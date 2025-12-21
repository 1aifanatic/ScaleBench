"""RAG task (e.g. MS MARCO / HotpotQA RAG)."""
from typing import Dict, Any
from tasks.base import Task


class RAGTask(Task):
    """RAG-style QA task for multi-hop reasoning."""

    def __init__(self, name: str = "rag_qa"):
        """Initialize RAG task.
        
        Args:
            name: Task name.
        """
        super().__init__(name)

    def build_prompt(self, example: Dict[str, Any], **kwargs) -> "GenerationInput":
        """Build GenerationInput with specific ordering: Instructions -> Context -> Question.
        
        Args:
            example: Example with 'context' and 'question' fields.
            **kwargs: Additional generation parameters.
            
        Returns:
            GenerationInput object.
        """
        from models.base import GenerationInput
        
        instructions = """Instructions:
- Answer directly in just 1-3 words
- Use only information from the context provided below
- Do not add explanations or extra information"""
        
        question = example.get("question", "")
        context = example.get("context", "")
        
        return GenerationInput(
            system_prompt=instructions,
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
