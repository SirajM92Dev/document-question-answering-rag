from openai import OpenAI


class LLMError(RuntimeError):
    """Raised when an LLM answer cannot be generated."""


class OpenAIAnswerGenerator:
    def __init__(self, api_key: str | None, model: str):
        if not api_key:
            raise LLMError("OPENAI_API_KEY is required to generate answers")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def answer(self, question: str, citations: list) -> str:
        context = "\n\n".join(
            f"[{citation.filename}, page {citation.page_number}]\n{citation.text}"
            for citation in citations
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from the supplied context. If the context does not contain "
                        "the answer, say that the documents do not provide enough information. "
                        "Do not invent facts. Cite sources inline as [filename, page N]."
                    ),
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise LLMError("The LLM returned an empty answer")
        return content.strip()
