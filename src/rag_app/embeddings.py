from functools import lru_cache


@lru_cache
def get_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class Embedder:
    def __init__(self, model_name: str):
        self.model = get_embedding_model(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()
