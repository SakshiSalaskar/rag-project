import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, chunks, model_name="all-MiniLM-L6-v2"):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        vectors = self.model.encode(
            [c["text"] for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.array(vectors, dtype="float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query, k=4):
        q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        q = np.array(q, dtype="float32")
        scores, ids = self.index.search(q, k)
        return [{"score": float(s), **self.chunks[i]} for s, i in zip(scores[0], ids[0])]