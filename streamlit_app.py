# streamlit_app.py

import os
import textwrap

import streamlit as st

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.chains import RetrievalQA


# ------------------------------------------------------------------------------
# OpenAI key from Streamlit secrets
# ------------------------------------------------------------------------------

# Streamlit Cloud: set this in "Secrets" as OPENAI_API_KEY = "sk-..."
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


DOCS_PATH = "docs"


# ------------------------------------------------------------------------------
# Load and index documents (cached)
# ------------------------------------------------------------------------------

@st.cache_resource
def load_vectorstore_and_chain():
    # --- Load documents of multiple types from docs/ ---
    documents = []

    # PDFs
    pdf_loader = DirectoryLoader(
        DOCS_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    documents.extend(pdf_loader.load())

    # Plain text
    txt_loader = DirectoryLoader(
        DOCS_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
    )
    documents.extend(txt_loader.load())

    # Markdown
    md_loader = DirectoryLoader(
        DOCS_PATH,
        glob="**/*.md",
        loader_cls=TextLoader,
    )
    documents.extend(md_loader.load())

    # DOCX
    docx_loader = DirectoryLoader(
        DOCS_PATH,
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
    )
    documents.extend(docx_loader.load())

    # PPTX
    pptx_loader = DirectoryLoader(
        DOCS_PATH,
        glob="**/*.pptx",
        loader_cls=UnstructuredPowerPointLoader,
    )
    documents.extend(pptx_loader.load())

    # CSV
    csv_loader = DirectoryLoader(
        DOCS_PATH,
        glob="**/*.csv",
        loader_cls=CSVLoader,
    )
    documents.extend(csv_loader.load())

    print(f"Loaded {len(documents)} documents from {DOCS_PATH}")

    # --- Chunking ---
    chunk_size_value = 1000
    chunk_overlap = 100

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_value,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    texts = text_splitter.split_documents(documents)

    # --- Build FAISS index ---
    vectorstore = FAISS.from_documents(texts, OpenAIEmbeddings())

    # --- Simple QA chain using LangChain's standard approach ---
    prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Helpful Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    # Use a simpler chain approach for modern langchain
    chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.2),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    return vectorstore, chain


def get_answer(query: str, k: int = 2):
    """
    Run similarity search on FAISS, then QA chain over top-k docs.
    Returns: (answer_text, reference_text)
    """
    vectorstore, chain = load_vectorstore_and_chain()

    # Run QA chain with retrieval
    results = chain({"query": query})

    # Extract answer and sources
    answer_text = results.get("result", "No answer generated")
    source_docs = results.get("source_documents", [])
    reference_text = "\n\n".join(d.page_content for d in source_docs)

    return answer_text, reference_text


# ------------------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------------------

def main():
    st.title("Document Q&A (FAISS + LangChain + OpenAI)")

    st.markdown(
        "This app loads documents from the `docs/` folder (PDF, DOCX, PPTX, TXT, MD, CSV), "
        "indexes them with FAISS, and answers questions using a RAG-style pipeline."
    )

    st.sidebar.header("Settings")
    k = st.sidebar.slider("Top-k chunks to use", min_value=1, max_value=10, value=2, step=1)
    show_refs = st.sidebar.checkbox("Show reference text", value=True)

    # Trigger loading index (so user sees spinner)
    with st.spinner("Loading documents and building index (only runs once)..."):
        _ = load_vectorstore_and_chain()

    query = st.text_area("Ask a question about your documents:")

    if st.button("Get Answer"):
        if not query.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Searching and generating answer..."):
            answer_text, reference_text = get_answer(query, k=k)

        st.subheader("Answer")
        st.write(answer_text)

        if show_refs:
            st.subheader("Reference Chunks (text used by the model)")
            st.text(
                textwrap.shorten(
                    reference_text,
                    width=4000,
                    placeholder="\n\n[... truncated ...]",
                )
            )


if __name__ == "__main__":
    main()
