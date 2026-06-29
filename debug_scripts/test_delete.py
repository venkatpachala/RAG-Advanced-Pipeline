from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

try:
    client.delete_collection("rag_chunks_v1")
    print("✅ Old collection deleted successfully!")
except Exception as e:
    print(f"Error: {e}")