"""
RAG Streamlit App - Document Q&A with FAISS and LangChain
"""

import os
import streamlit as st
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_core.prompts import PromptTemplate

# Set OpenAI API key from Streamlit secrets
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

DOCS_PATH = "docs"


def load_documents():
    """Load documents from docs folder - only PDF and TXT files."""
    documents = []
    docs_dir = Path(DOCS_PATH)
    
    if not docs_dir.exists():
        st.warning(f"Docs directory '{DOCS_PATH}' not found!")
        return documents
    
    # Load PDFs
    try:
        pdf_files = list(docs_dir.glob("**/*.pdf"))
        if pdf_files:
            for pdf_file in pdf_files:
                try:
                    loader = PyPDFLoader(str(pdf_file))
                    documents.extend(loader.load())
                except Exception as e:
                    st.warning(f"Failed to load {pdf_file.name}: {str(e)}")
    except Exception as e:
        st.warning(f"Error loading PDFs: {str(e)}")
    
    # Load TXT files
    try:
        txt_files = list(docs_dir.glob("**/*.txt"))
        if txt_files:
            for txt_file in txt_files:
                try:
                    loader = TextLoader(str(txt_file))
                    documents.extend(loader.load())
                except Exception as e:
                    st.warning(f"Failed to load {txt_file.name}: {str(e)}")
    except Exception as e:
        st.warning(f"Error loading TXT files: {str(e)}")
    
    return documents


@st.cache_resource
def load_vectorstore():
    """Load documents and create FAISS vectorstore."""
    documents = load_documents()
    
    if not documents:
        st.error("No documents loaded. Please add PDF or TXT files to the docs/ folder.")
        st.stop()
    
    st.info(f"Loaded {len(documents)} documents")
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)
    st.info(f"Split into {len(chunks)} chunks")
    
    # Create FAISS vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore


def get_answer(query: str):
    """Get answer from RAG pipeline."""
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Get relevant docs
    relevant_docs = retriever.invoke(query)
    
    if not relevant_docs:
        return "No relevant documents found.", ""
    
    # Format context
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Create prompt
    prompt_template = """Answer the question based on the following context. If you don't know the answer, say so.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )
    
    # Get answer
    llm = OpenAI(temperature=0.3)
    formatted_prompt = prompt.format(context=context, question=query)
    answer = llm(formatted_prompt)
    
    return answer, context


def main():
    st.set_page_config(page_title="RAG Q&A", layout="wide")
    st.title("Document Q&A with RAG")
    
    st.markdown("""
    This app lets you ask questions about documents in the `docs/` folder.
    - Supported formats: PDF, TXT
    - Uses FAISS for semantic search
    - Powered by OpenAI
    """)
    
    # Load vectorstore
    try:
        with st.spinner("Loading documents..."):
            vectorstore = load_vectorstore()
        st.success("Documents loaded!")
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        st.stop()
    
    # Query input
    query = st.text_input("Ask a question about your documents:")
    
    if query:
        try:
            with st.spinner("Searching and generating answer..."):
                answer, context = get_answer(query)
            
            st.subheader("Answer")
            st.write(answer)
            
            with st.expander("View context"):
                st.text(context[:2000] + "..." if len(context) > 2000 else context)
        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
