# Project notes

Document Q&A (RAG) over PDFs. Local embeddings + FAISS + local LLM via Ollama.

## Setup
- Python 3.12, venv, sentence-transformers (all-MiniLM-L6-v2), FAISS, PyMuPDF
- LLM: qwen2.5:1.5b via Ollama (laptop has 7.7 GB RAM, so a small model)
- Test document: "Attention Is All You Need" (15 pages, 58 chunks at 150 words, 30 overlap)

## Day 1: Retrieval findings
- Page 1 is mostly author names and emails, which adds noise to the index.
- Fixed-size chunking cuts across section boundaries (one chunk mixed the end
  of positional encoding with the start of Section 4).
- Chunking page by page splits text that continues across a page break.
- "What is self-attention?" missed the best definition chunk (page 2) in the
  top 5; figure-caption chunks from pages 13 and 14 ranked higher.
- Likely fixes: sentence/section-aware chunking, hybrid search (BM25 + embeddings),
  reranking.

## Day 2: Generation findings
- Prompt v1 (rules only): answers were correct but had no source citations.
- Prompt v2 (rules + unrelated example of the format): citations appeared on
  3 of 4 answerable questions, and each cited page matched the retrieved chunks.
- Out-of-scope question ("Who won the 2018 World Cup?") returned the exact
  refusal: "I could not find this in the document."
- Small-model quirks: copies source text on "why" questions; changed
  "transduction" to "transcription" (PDF hyphenation artifact).

## Test results (5 questions)
| Question | Correct | Cited | Notes |
|---|---|---|---|
| What is self-attention? | Partly | No | Conflated with multi-head attention |
| Why is attention better than recurrence? | Partly | Yes | Only mentions parallelization |
| How many attention heads? | Yes | Yes | 8 |
| What optimizer was used? | Yes | Yes | Adam, β1=0.9, β2=0.98, ε=1e-9 |
| Who won the 2018 World Cup? | Yes (refused) | n/a | Exact refusal |

## To do
- Experiment with chunk sizes (80, 150, 300) and record the rank of the best chunk
- Build a 50-question evaluation set with known answers and pages
- Compare retrieval: embeddings only vs hybrid vs hybrid + reranker
- Compare models: qwen2.5:1.5b vs a hosted model
- Streamlit app with source cards

## Day 3: Refactor
- code now lives in src/ (ingest.py, retriever.py, rag.py) with a Streamlit UI
- 5-question UI test: 3 correct, 1 vague, 1 failed (“why” question copied an irrelevant fragment)
- Idea: refuse when the top retrieval score is under a threshold instead of relying on the model