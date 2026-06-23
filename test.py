# test_loader.py
import os
from document_loader import EnterpriseDocumentLoader

# ====================== CONFIGURATION ======================
DATA_FOLDER = "test_data"                    # ← Change this to your folder path

# Set this to False if you don't have LlamaParse API key
USE_LLAMA_PARSE = True                      # ← Change to True later

# Optional: Put your key here or set as environment variable
LLAMA_API_KEY="llx-itdagRKTP9JNkKypdda6jaA50vcSS0EQkLiTLO3ITK5B0aCm"                         # or "llx-xxx..."
# ===========================================================

def main():
    print("=" * 60)
    print("Testing EnterpriseDocumentLoader")
    print("=" * 60)

    # Create loader
    loader = EnterpriseDocumentLoader(
        data_dir=DATA_FOLDER,
        use_llama_parse=USE_LLAMA_PARSE,
        llama_parse_api_key=LLAMA_API_KEY
    )

    # Load documents
    print(f"\nLoading documents from: {DATA_FOLDER}")
    documents = loader.load_documents()

    # Print results
    print(f"\n{'='*60}")
    print(f"TOTAL DOCUMENTS LOADED: {len(documents)}")
    print(f"{'='*60}\n")

    for i, doc in enumerate(documents, 1):
        print(f"--- Document {i} ---")
        print(f"Source     : {doc.metadata.get('source', 'N/A')}")
        print(f"File Type  : {doc.metadata.get('file_type', 'N/A')}")
        print(f"Timestamp  : {doc.metadata.get('ingestion_timestamp', 'N/A')}")
        print(f"Text Length: {len(doc.text)} characters")
        print(f"Preview    : {doc.text[:300]}...")   # First 300 chars
        print()
    for i, doc in enumerate(documents[:3]):
        print("\n====================")
        print("METADATA")
        print(doc.metadata)

        print("\nDOC ID")
        print(getattr(doc, "doc_id", None))

        print("\nTEXT LENGTH")
        print(len(doc.text))

if __name__ == "__main__":
    main()

