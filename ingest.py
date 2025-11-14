# ingest.py
import os
from typing import List, Dict

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI

from parsers import parse_file

client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"
QDRANT_PATH = "qdrant_db"
COLLECTION_NAME = "team_kb"
DOCS_ROOT = "kb_docs"


def chunk_text(text: str, max_tokens: int = 500, overlap: int = 100) -> List[str]:
    approx_chars = max_tokens * 4
    step = approx_chars - overlap * 4
    chunks = []
    start = 0
    while start < len(text):
        end = start + approx_chars
        chunks.append(text[start:end])
        start += max(step, 1)
    return chunks


def build_qdrant_client():
    return QdrantClient(path=QDRANT_PATH)


def ingest_local_folder(
    docs_root: str = DOCS_ROOT,
    collection_name: str = COLLECTION_NAME,
):
    # Validate docs folder exists
    if not os.path.exists(docs_root):
        raise FileNotFoundError(f"Documents folder '{docs_root}' not found.")
    
    qdrant = build_qdrant_client()
    
    # Create or recreate collection
    try:
        qdrant.delete_collection(collection_name)
    except:
        pass  # Collection doesn't exist yet
    
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1536,  # Size for OpenAI's text-embedding-3-small
            distance=Distance.COSINE,
        ),
    )

    doc_ids = []
    texts = []
    metadatas: List[Dict] = []

    supported_exts = {".pdf", ".docx", ".pptx", ".txt", ".md", ".markdown", ".csv"}

    for root, dirs, files in os.walk(docs_root):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in supported_exts:
                continue

            path = os.path.join(root, fname)
            rel_path = os.path.relpath(path, docs_root)

            print(f"Parsing {rel_path}")
            text = parse_file(path)
            if not text or not text.strip():
                print(f"Skipping empty/unsupported file: {rel_path}")
                continue

            chunks = chunk_text(text, max_tokens=500, overlap=100)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{rel_path}__{i}"
                doc_ids.append(chunk_id)
                texts.append(chunk)
                metadatas.append(
                    {
                        "rel_path": rel_path,
                        "file_name": fname,
                        "ext": ext,
                    }
                )

    if not texts:
        print("No chunks to index.")
        return

    print(f"Embedding {len(texts)} chunks...")
    embeddings = []
    batch_size = 128
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        embeddings.extend([d.embedding for d in resp.data])

    # Create points for Qdrant
    points = []
    for i, (doc_id, text, embedding, metadata) in enumerate(
        zip(doc_ids, texts, embeddings, metadatas)
    ):
        points.append(
            PointStruct(
                id=i,
                vector=embedding,
                payload={"text": text, "metadata": metadata, "doc_id": doc_id},
            )
        )

    # Upsert points to Qdrant
    qdrant.upsert(
        collection_name=collection_name,
        points=points,
    )

    print("Ingestion complete.")


if __name__ == "__main__":
    ingest_local_folder()
