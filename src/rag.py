import ollama

PROMPT = """Answer the question using ONLY the context below.

Rules:
- Write 2 to 4 complete sentences.
- After each sentence, add the source tag it came from, like [Source 2].
- If the context does not contain the answer, reply exactly: I could not find this in the document.

Example of the format (unrelated topic):
Water boils at 100 degrees Celsius at sea level. [Source 2] The boiling point drops at higher altitudes. [Source 1]

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def answer(question, retriever, k=4, model_name="qwen2.5:1.5b"):
    results = retriever.search(question, k)

    context = "\n\n".join(
        f"[Source {i+1}, page {r['page']}]\n{r['text']}"
        for i, r in enumerate(results)
    )

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": PROMPT.format(context=context, question=question)}],
    )
    return response["message"]["content"], results