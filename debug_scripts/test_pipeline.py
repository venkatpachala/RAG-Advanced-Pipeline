from ingestion.embedding.vector_store import VectorStore

vs = VectorStore()
vs.create_collection("rag_chunks_v1")
vs.create_payload_indexes("rag_chunks_v1")
print("Collection and indexes created successfully!")