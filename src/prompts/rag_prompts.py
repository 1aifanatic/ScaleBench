"""System prompts for RAG tasks."""

# Default prompt used if no dataset-specific prompt is found
DEFAULT_RAG_PROMPT = """Instructions:
- Answer directly in just 1-3 words
- Use only information from the context provided below
- Do not add explanations or extra information"""

# HotpotQA specific prompt
HOTPOTQA_RAG_PROMPT = """Instructions:
- Answer directly in just 1-3 words
- Use only information from the context provided below
- Do not add explanations or extra information"""

# LongBench specific prompt
LONGBENCH_RAG_PROMPT = """Instructions:
- Answer the question based on the provided context
- Keep your answer brief and concise
- Use only information from the context provided below
- Do not add explanations or extra information"""

# ETHIC specific prompt for set-based evaluation
ETHIC_RAG_PROMPT = """Instructions:
- Answer based only on the provided context
- Return each answer on the same line, comma-separated
- No numbering, bullets, or extra formatting
- Include only the requested items, nothing else
- Do not add explanations or surrounding text"""

# MS-MARCO specific prompt for structured output (answer + passage retrieval)
MSMARCO_RAG_PROMPT = """You are a question answering assistant.

Given a question and numbered passages, you must:
1. Answer the question using ONLY information from the provided passages
2. List the passage number(s) that contain the answer

IMPORTANT RULES:
- Use only information from the passages provided
- Provide a concise, straightforward answer directly addressing the question
- If multiple passages contain relevant information, list all of them
- Passage numbers are 0-indexed (first passage is 0)
- Return your response as valid JSON in this exact format:
{"answer": "<your concise answer>", "passages": [<passage numbers used>]}

Example response:
{"answer": "Paris", "passages": [2, 5]}"""

# FinanceBench specific prompt for financial document QA
FINANCEBENCH_RAG_PROMPT = """Instructions:
- Answer the financial question based only on the provided context
- Be precise with numbers, units, and currency values
- If the answer is a number, include the exact value with units
  (e.g., "$1.2 billion", "15.3%")
- Do not add explanations or extra information"""

# Mapping of dataset names to prompts
RAG_PROMPTS = {
    "hotpotqa": HOTPOTQA_RAG_PROMPT,
    "longbench": LONGBENCH_RAG_PROMPT,
    "ethic": ETHIC_RAG_PROMPT,
    "msmarco": MSMARCO_RAG_PROMPT,
    "financebench": FINANCEBENCH_RAG_PROMPT,
}
