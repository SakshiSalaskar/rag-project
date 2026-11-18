import os
import streamlit as st
from src.ingest import load_pdf, chunk_pages
from src.retriever import Retriever
from src.rag import answer

st.set_page_config(page_title="DocChat", page_icon="📄")
st.title("DocChat")
st.caption("Ask questions about your document. Answers come from a local model.")

PDF_PATH = "data/doc.pdf"


@st.cache_resource
def build_retriever(path):
    pages = load_pdf(path)
    chunks = chunk_pages(pages, source=os.path.basename(path))
    return Retriever(chunks), len(pages), len(chunks)


if not os.path.exists(PDF_PATH):
    st.error(f"Put a PDF at {PDF_PATH} and reload.")
    st.stop()

with st.spinner("Loading the document and building the search index..."):
    retriever, n_pages, n_chunks = build_retriever(PDF_PATH)

with st.sidebar:
    st.subheader("Document")
    st.write(os.path.basename(PDF_PATH))
    st.caption(f"{n_pages} pages, {n_chunks} chunks")
    st.subheader("Settings")
    k = st.slider("Chunks retrieved", min_value=2, max_value=6, value=4)

question = st.text_input("Ask a question about your document")

if st.button("Ask") and question.strip():
    with st.spinner("Thinking..."):
        text, sources = answer(question, retriever, k=k)

    st.markdown("### Answer")
    st.write(text)

    st.markdown("### Sources")
    for i, s in enumerate(sources, 1):
        with st.expander(f"Source {i}: page {s['page']} (match {s['score']:.2f})"):
            st.write(s["text"])