import logging
from typing import List
from ingestion.embedding.embedder import Embedder
from retrieval.retriever import Retriever
import ollama

logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    Advanced Grounded Query Rewriter
    - Uses retrieved context from documents to rewrite queries
    - Much more relevant to your ingested data
    """

    def __init__(self, model: str = "qwen2.5:7b"):
        self.model = model
        self.embedder = Embedder()
        self.retriever = Retriever()
        logger.info(f"Grounded QueryRewriter initialized with model: {model}")

    def rewrite_query(self, query: str) -> str:
        if not query or len(query.strip()) < 3:
            return query

        try:
            # Step 1: Get embedding of original query
            query_embedding = self.embedder.embed_texts([query])[0]["dense"]

            # Step 2: Light retrieval to get context from documents
            top_chunks = self.retriever.hybrid_search(
                query_dense=query_embedding,
                limit=5
            )

            # Step 3: Create context from top chunks
            context = "\n".join([
                chunk.get("text", "")[:600] for chunk in top_chunks[:3]
            ])

            # Step 4: Ask LLM to rewrite query using document context
            system_prompt = f"""You are an expert at rewriting queries for better retrieval from a specific document collection.

Here is relevant context from the documents:
{context}

Your task:
Rewrite the user's query to be more precise and better aligned with the above document context.
- Keep the original intent.
- Make it more specific to the domain of the documents.
- Output ONLY the rewritten query. Do not add any explanation or extra text."""

            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Original query: {query}"}
                ],
                options={
                    "temperature": 0.3,
                    "num_predict": 70
                }
            )

            rewritten = response['message']['content'].strip()

            # Clean output
            rewritten = rewritten.replace("Rewritten query:", "").strip()
            rewritten = rewritten.replace('"', '').strip()

            if len(rewritten) > 5:
                logger.info(f"Original : {query}")
                logger.info(f"Rewritten: {rewritten}")
                return rewritten
            else:
                return query

        except Exception as e:
            logger.warning(f"Grounded query rewriting failed: {e}. Using original query.")
            return query