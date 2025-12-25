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

# Mapping of dataset names to prompts
RAG_PROMPTS = {
    "hotpotqa": HOTPOTQA_RAG_PROMPT,
    "longbench": LONGBENCH_RAG_PROMPT,
    "ethic": ETHIC_RAG_PROMPT,
}
