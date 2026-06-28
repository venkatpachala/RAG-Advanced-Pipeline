import logging
from typing import List

logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    Simple Query Rewriter for improving user queries.
    (Can be upgraded later to use an LLM)
    """

    def rewrite_query(self, query: str) -> str:
        """
        Basic query rewriting.
        Currently does light expansion for short queries.
        """
        rewritten = query.strip()

        # Simple improvement: expand very short queries
        if len(rewritten.split()) < 6:
            rewritten = f"Explain in detail about: {rewritten}"

        logger.info(f"Original Query : {query}")
        logger.info(f"Rewritten Query: {rewritten}")

        return rewritten

    def generate_variations(self, query: str, num_variations: int = 3) -> List[str]:
        """Generate multiple versions of the query (optional feature)"""
        variations = [query]
        variations.append(f"What is {query}?")
        variations.append(f"Explain {query} clearly")

        return variations[:num_variations]