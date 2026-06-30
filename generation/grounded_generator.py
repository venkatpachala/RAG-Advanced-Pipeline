import logging
from typing import List, Dict, Any, Optional

from observability import log_stage
import ollama

logger = logging.getLogger(__name__)


class GroundedGenerator:
    """
    Grounded Generation with proper observability, request tracing,
    and token usage tracking.
    """

    def __init__(self, model: str = "qwen2.5:7b"):
        self.model = model
        logger.info(f"GroundedGenerator initialized with model: {model}")

    def generate(
        self,
        user_query: str,
        retrieved_chunks: List[Dict[str, Any]],
        request_id: Optional[str] = None,
        max_context_chars: int = 8000
    ) -> Dict[str, Any]:
        """
        Generate a grounded answer with citations.
        Now properly integrated with observability layer.
        """
        if not retrieved_chunks:
            logger.warning(
                "No chunks retrieved for generation",
                extra={"request_id": request_id, "query": user_query}
            )
            return {
                "answer": "I don't have enough relevant information in the provided documents to answer this question.",
                "citations": [],
                "grounded": False,
                "chunks_used": 0,
                "tokens": {"input": 0, "output": 0, "total": 0}
            }

        # Build context + citations
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

        # Balanced system prompt (strict but not overly cautious)
        system_prompt = """You are a precise and trustworthy research assistant.

Your job is to answer the user's question using **only** the information provided in the Context.

Rules:
- Base your answer strictly on the provided Context.
- If the Context does not contain enough information, clearly state that.
- Always cite sources using the format shown in the context (e.g., filename (Page X)).
- Be concise and focused. Avoid adding unnecessary background information.
- Do not speculate or add information from outside the Context."""

        user_prompt = f"""Context:
{context}

Question: {user_query}

Answer:"""

        try:
            with log_stage(
                "llm_generation",
                request_id=request_id,
                query=user_query,
                num_chunks=len(retrieved_chunks),
                context_length=len(context)
            ):
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    options={
                        "temperature": 0.15,   # Slightly lower for more deterministic output
                        "num_predict": 900
                    }
                )

            answer = response["message"]["content"].strip()

            # Extract token usage (Ollama provides these fields)
            tokens = {
                "input": response.get("prompt_eval_count", 0),
                "output": response.get("eval_count", 0),
                "total": response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
            }

            # Log token usage as structured event
            logger.info(
                "Generation completed",
                extra={
                    "request_id": request_id,
                    "event": "generation_tokens",
                    "model": self.model,
                    "tokens": tokens
                }
            )

            return {
                "answer": answer,
                "citations": citations,
                "grounded": True,
                "chunks_used": len(retrieved_chunks),
                "tokens": tokens
            }

        except Exception as e:
            logger.error(
                "Generation failed",
                extra={
                    "request_id": request_id,
                    "event": "generation_failed",
                    "query": user_query,
                    "error": str(e)
                },
                exc_info=True
            )
            return {
                "answer": "An error occurred while generating the answer.",
                "citations": [],
                "grounded": False,
                "chunks_used": 0,
                "tokens": {"input": 0, "output": 0, "total": 0}
            }