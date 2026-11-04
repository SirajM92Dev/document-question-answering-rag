from rag_app.evaluation import evaluate_answer
from rag_app.models import Citation


def test_grounded_answer_scores_higher():
    citation = Citation(chunk_id="1", filename="a.pdf", page_number=1, text="Paris is in France.", score=0.9)
    result = evaluate_answer("Where is Paris?", "Paris is in France.", [citation])
    assert result.groundedness > 0.5
    assert result.citation_coverage > 0
