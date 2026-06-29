from ingestion.embedding.embedder import Embedder

print("Downloading bge-m3 model... This may take 15-40 minutes on first run.")
embedder = Embedder(model_name="BAAI/bge-m3")
print("Model downloaded and loaded successfully!")