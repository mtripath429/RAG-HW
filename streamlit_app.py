# streamlit_app.py
import os
import textwrap

import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from openai import OpenAI

from ingest import ingest_local_folder

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
QDRANT_PATH = "qdrant_db"
COLLECTION_NAME = "team_kb"

client = OpenAI()


@st.cache_resource
def get_qdrant_client():
    return QdrantClient(path=QDRANT_PATH)


def embed_query(query: str):
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    return resp.data[0].embedding


def build_system_prompt():
    return (
        "You are an assistant answering questions based only on the provided internal documents. "
        "Use ONLY the provided context. If insufficient, say so clearly."
    )


def build_user_prompt(query: str, context_chunks):
    context_texts = []
    for i, chunk in enumerate(context_chunks):
        meta = chunk["metadata"]
        header = f"[Source {i+1}] {meta.get('rel_path')}"
        body = chunk["document"]
        context_texts.append(header + "\n" + body)

    context_block = "\n\n---\n\n".join(context_texts)

    return f"""Use the following document excerpts to answer the question.

Context:
{context_block}

Question: {query}

Rules:
- ONLY answer using the context above.
- If the context does not contain the answer, say you don't know.
- Cite the specific [Source X] you used.
"""


def answer_query(query: str, k: int = 5):
    qdrant = get_qdrant_client()
    q_emb = embed_query(query)
    
    try:
        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=q_emb,
            limit=k,
        )
    except Exception as e:
        raise Exception(f"Error searching Qdrant: {str(e)}")

    chunks = []
    for result in results:
        chunks.append(
            {
                "id": result.id,
                "document": result.payload.get("text", ""),
                "metadata": result.payload.get("metadata", {}),
                "distance": result.score,
            }
        )

    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(query, chunks)},
    ]

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content, chunks


def critique_answer(answer: str, chunks):
    context_snips = []
    for i, c in enumerate(chunks):
        meta = c["metadata"]
        snippet = textwrap.shorten(c["document"], width=400, placeholder=" [...]")
        context_snips.append(f"[Source {i+1}] {meta.get('rel_path')}:\n{snippet}")

    context_text = "\n\n".join(context_snips)

    critique_prompt = f"""
Evaluate the answer based on the retrieved context.

Context:
{context_text}

Answer:
{answer}

Critique (3-5 bullet points):
- Are claims grounded in the context?
- Any hallucinations?
- Is it complete?
- Are the right chunks used?
"""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a strict RAG evaluator."},
            {"role": "user", "content": critique_prompt},
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content


def main():
    st.title("Team Knowledge Base Q&A (Local Folder RAG)")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("❌ OPENAI_API_KEY environment variable not set. Please set it and restart.")
        return

    st.sidebar.header("Settings")
    k = st.sidebar.slider("Top-k retrieved chunks", 3, 15, 5)
    show_chunks = st.sidebar.checkbox("Show retrieved chunks", value=True)
    enable_critique = st.sidebar.checkbox("Show model self-critique", value=False)

    if st.sidebar.button("Rebuild Index (kb_docs/)"):
        if not os.path.exists("kb_docs"):
            st.sidebar.error("kb_docs folder not found. Please create it and add documents.")
            return
        with st.spinner("Indexing and embedding documents..."):
            try:
                ingest_local_folder()
                st.sidebar.success("Index rebuilt. Ask a new question.")
            except Exception as e:
                st.sidebar.error(f"Error building index: {str(e)}")
                return

    query = st.text_area("Ask a question about your documents:")

    if st.button("Ask"):
        if not query.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Retrieving context and generating answer..."):
            answer, chunks = answer_query(query, k=k)

        st.subheader("Answer")
        st.write(answer)

        if enable_critique:
            st.subheader("Self-Critique")
            st.write(critique_answer(answer, chunks))

        if show_chunks:
            st.subheader("Retrieved Context")
            for i, c in enumerate(chunks):
                meta = c["metadata"]
                with st.expander(f"Source {i+1}: {meta.get('rel_path')}"):
                    st.write(f"📄 File: {meta['file_name']} ({meta['ext']})")
                    st.text(
                        textwrap.shorten(
                            c["document"], width=1200, placeholder=" [...]"
                        )
                    )


if __name__ == "__main__":
    main()
