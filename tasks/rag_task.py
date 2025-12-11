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

    def build_prompt(self, example: Dict[str, Any]) -> str:
        """Build prompt for RAG QA.
        
        Instructs model to answer concisely using only provided context.
        
        Args:
            example: Example with 'context' and 'question' fields.
            
        Returns:
            Formatted prompt string.
        """
        context = example.get("context", "")
        question = example.get("question", "")
        
        prompt = f"""Answer the following question using ONLY the information provided in the context below. 

Context:
{context}

Question: {question}

Instructions:
- Provide a brief, direct answer using minimum words
- Answer in a single line
- Use only information from the context
- Do not add explanations or extra information

Answer:"""
        
        return prompt

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
