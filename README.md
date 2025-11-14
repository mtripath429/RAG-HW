# Team Knowledge Base RAG (Streamlit + OpenAI + Chroma)

This app lets you ask questions over a local folder of documents (`kb_docs/`)
using a vector-based RAG pipeline (OpenAI embeddings + Chroma + Streamlit).

## How it works

- Put documents in `kb_docs/` (PDF, DOCX, PPTX, CSV, TXT, MD)
- Click "Rebuild Index" in the Streamlit app → creates embeddings
- Ask questions → model retrieves relevant chunks + answers based on them
- Optional: enable self-critique mode

## Running Locally

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
streamlit run streamlit_app.py
