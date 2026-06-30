from observability import log_stage, generate_request_id

def ask(self, user_query: str, top_k: int = 8):
    request_id = generate_request_id()

    with log_stage("query_rewriting", {"request_id": request_id, "original_query": user_query}):
        rewritten_query = self.rewriter.rewrite_query(user_query)

    with log_stage("retrieval", {"request_id": request_id, "query": rewritten_query}):
        chunks = self.query_engine.retrieve(rewritten_query, top_k)

    with log_stage("generation", {"request_id": request_id}):
        result = self.generator.generate(rewritten_query, chunks)

    return result