# ingest.py
import os
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from parsers import parse_file

client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"
PERSIST_DIR = "chroma_db"
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


def build_chroma_client():
    return chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=PERSIST_DIR,
        )
    )


def ingest_local_folder(
    docs_root: str = DOCS_ROOT,
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
):
    db = build_chroma_client()
    collection = db.get_or_create_collection(name=collection_name)

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

    collection.delete(where={})
    collection.add(
        ids=doc_ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    db.persist()
    print("Ingestion complete.")


if __name__ == "__main__":
    ingest_local_folder()
