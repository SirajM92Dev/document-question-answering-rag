import re

from .models import Citation, EvaluationResponse


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def evaluate_answer(question: str, answer: str, citations: list[Citation]) -> EvaluationResponse:
    question_tokens = _tokens(question)
    answer_tokens = _tokens(answer)
    context_tokens = set().union(*(_tokens(citation.text) for citation in citations)) if citations else set()
    overlap = len(answer_tokens & context_tokens) / max(len(answer_tokens), 1)
    relevance = len(question_tokens & answer_tokens) / max(len(question_tokens), 1)
    coverage = min(1.0, len(citations) / 3) if citations else 0.0
    notes = []
    if not citations:
        notes.append("No retrieved passages were available.")
    if overlap < 0.5:
        notes.append("The answer has limited lexical overlap with retrieved context; review grounding.")
    if relevance < 0.1:
        notes.append("The answer may not directly address the question.")
    return EvaluationResponse(
        retrieval_recall=coverage,
        answer_relevance=round(relevance, 4),
        citation_coverage=coverage,
        groundedness=round(overlap, 4),
        notes=notes or ["The answer has reasonable lexical support in the retrieved context."],
    )
