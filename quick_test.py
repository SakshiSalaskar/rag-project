from src.ingest import load_pdf, chunk_pages
from src.retriever import Retriever
from src.rag import answer

pages = load_pdf("data/doc.pdf")
chunks = chunk_pages(pages)
retriever = Retriever(chunks)
print(f"{len(pages)} pages, {len(chunks)} chunks")

for q in [
    "What is self-attention?",
    "How many attention heads does the base model use?",
    "Who won the 2018 World Cup?",
]:
    text, sources = answer(q, retriever)
    print("\nQ:", q)
    print(text)
    print("Pages retrieved:", [s["page"] for s in sources])