import logging
import json
from typing import List, Dict, Any
from observability import log_stage
import ollama

logger = logging.getLogger(__name__)


class GroundedGenerator:
    """
    Grounded Generation with token usage tracking and structured logging.
    """

    def __init__(self, model: str = "qwen2.5:7b"):
        self.model = model
        logger.info(f"GroundedGenerator initialized with model: {model}")

    def generate(
        self,
        user_query: str,
        retrieved_chunks: List[Dict[str, Any]],
        max_context_chars: int = 8000
    ) -> Dict[str, Any]:

        if not retrieved_chunks:
            return {
                "answer": "I don't have enough relevant information in the provided documents to answer this question.",
                "citations": [],
                "grounded": False,
                "chunks_used": 0,
                "tokens": {"input": 0, "output": 0, "total": 0}
            }

        # Build context and citations
        context_parts = []
        citations = []

        for i, chunk in enumerate(retrieved_chunks, 1):
            text = chunk.get("text", "")[:900]
            source_file = chunk.get("source_file", "Unknown")
            page_num = chunk.get("page_number")

            citation_str = f"{source_file} (Page {page_num})" if page_num else source_file
            context_parts.append(f"[{i}] {citation_str}\n{text}")
            citations.append(citation_str)

        context = "\n\n".join(context_parts)[:max_context_chars]

        system_prompt = """You are a precise and trustworthy assistant.
Answer the user's question **only** using the provided context.
- If the answer is not in the context, clearly say so.
- Use citations like [1], [2] when making claims.
- Be concise and clear."""

        user_prompt = f"""Context:
{context}

Question: {user_query}

Answer:"""

        try:
            with log_stage("llm_generation", {
                "query": user_query,
                "num_chunks": len(retrieved_chunks),
                "context_length": len(context)
            }):
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    options={
                        "temperature": 0.2,
                        "num_predict": 900
                    }
                )

            answer = response['message']['content'].strip()

            # Try to extract token usage (Ollama sometimes provides this)
            tokens = {
                "input": response.get("prompt_eval_count", 0),
                "output": response.get("eval_count", 0),
                "total": response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
            }

            # Log token usage
            logger.info(json.dumps({
                "event": "generation_tokens",
                "model": self.model,
                "query": user_query,
                "tokens": tokens
            }))

            return {
                "answer": answer,
                "citations": citations,
                "grounded": True,
                "chunks_used": len(retrieved_chunks),
                "tokens": tokens
            }

        except Exception as e:
            logger.error(json.dumps({
                "event": "generation_failed",
                "query": user_query,
                "error": str(e)
            }))
            return {
                "answer": "An error occurred while generating the answer.",
                "citations": [],
                "grounded": False,
                "chunks_used": 0,
                "tokens": {"input": 0, "output": 0, "total": 0}
            }