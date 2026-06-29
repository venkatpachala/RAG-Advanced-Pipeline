import logging
from typing import List, Dict, Any
import ollama

logger = logging.getLogger(__name__)


class GroundedGenerator:
    """
    Generates answers strictly grounded in the retrieved context + adds citations.
    """

    def __init__(self, model: str = "qwen2.5:7b"):
        self.model = model
        logger.info(f"GroundedGenerator initialized with model: {model}")

    def generate(
        self,
        user_query: str,
        retrieved_chunks: List[Dict[str, Any]],
        max_context_chars: int = 6000
    ) -> Dict[str, Any]:
        """
        Generate a grounded answer with citations.
        """

        if not retrieved_chunks:
            return {
                "answer": "I don't have enough information in the provided documents to answer this question.",
                "citations": [],
                "grounded": False
            }

        # Prepare context with source information
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            text = chunk.get("text", "")[:800]
            source = chunk.get("source_file", "Unknown")
            context_parts.append(f"[{i}] Source: {source}\n{text}")

        context = "\n\n".join(context_parts)[:max_context_chars]

        system_prompt = """You are a precise and trustworthy AI assistant.
Your job is to answer the user's question **only** using the information provided in the context below.

Rules:
- Answer strictly based on the provided context. Do not use external knowledge.
- If the answer is not present in the context, clearly say "The provided documents do not contain sufficient information to answer this question."
- Use citations like [1], [2], etc. when making claims.
- Be concise and clear."""

        user_prompt = f"""Context:
{context}

Question: {user_query}

Answer:"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": 0.2,
                    "num_predict": 800
                }
            )

            answer = response['message']['content'].strip()

            return {
                "answer": answer,
                "citations": [chunk.get("source_file", "Unknown") for chunk in retrieved_chunks],
                "grounded": True,
                "context_used": len(retrieved_chunks)
            }

        except Exception as e:
            logger.error(f"Grounded generation failed: {e}")
            return {
                "answer": "An error occurred while generating the answer.",
                "citations": [],
                "grounded": False
            }