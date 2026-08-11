import numpy as np
from sentence_transformers import CrossEncoder
from config.settings import RERANKER_MODEL, RERANKER_TOP_N, RERANKER_ENABLE

_reranker_model = None


def _get_reranker() -> CrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(RERANKER_MODEL, local_files_only=True)
    return _reranker_model

def rerank(query: str, results: list[tuple], top_n=None) -> list[tuple]:
    if not RERANKER_ENABLE or not results:
        return results

    n = top_n or RERANKER_TOP_N

    pairs = [(query, doc.page_content.strip()) for doc, _ in results]
    scores = _get_reranker().predict(pairs) 

    mn, mx = float(scores.min()), float(scores.max())
    normalized = [(s - mn) / (mx - mn) for s in scores] if mx > mn else [1.0] * len(scores)

    reranked = sorted(
        [(doc, ns) for (doc, _), ns in zip(results, normalized)],
        key=lambda x: x[1],
        reverse=True,
    )
    return reranked[:n]