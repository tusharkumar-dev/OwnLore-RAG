from chromadb import PersistentClient
from config.settings import COLLECTION_NAME, DB_PATH, SCORE_THRESHOLD, TOP_K
from retrieval.reranker import rerank
from retrieval.retriever import retrieve_with_scores

_MIN_CHUNK_LENGTH = 100     
_SIMILARITY_CUTOFF = 0.3   
_DEDUP_OVERLAP_RATIO = 0.85


def search_chunks(query: str, selected_pdf=None, top_k=None) -> str:
    if not query.strip():
        return "No query provided."

    try:
        col = PersistentClient(path=DB_PATH).get_collection(COLLECTION_NAME)
        if col.count() == 0:
            return "DB is empty. Please upload a PDF first."
    except Exception:
        return "No documents ingested yet. Please upload a PDF first."

    actual_k = top_k or TOP_K
    retrieve_k = actual_k * 2 

    raw = retrieve_with_scores(query=query, top_k=retrieve_k, filter_source=selected_pdf)

    if not raw:
        return "No relevant chunks found."

    filtered = _filter_noise(raw)
    deduped = _deduplicate(filtered)
    reranked = rerank(query, deduped, top_n=actual_k)

    if not reranked:
        return "Nothing left after filtering."

    return _format_results(query, reranked)


def _filter_noise(results):
    # drop TOC/header chunks and low-score noise
    cleaned = []
    for doc, score in results:
        text = doc.page_content.strip()
        if len(text) < _MIN_CHUNK_LENGTH:
            continue
        if score < _SIMILARITY_CUTOFF:
            continue
        cleaned.append((doc, score))
    return cleaned


def _deduplicate(results):

    seen = []
    unique = []

    for doc, score in results:
        text = doc.page_content.strip()
        words = set(text.lower().split())

        is_dup = False
        for s in seen:
            sw = set(s.lower().split())
            if not sw:
                continue
            if len(words & sw) / len(words | sw) > _DEDUP_OVERLAP_RATIO:
                is_dup = True
                break

        if not is_dup:
            seen.append(text)
            unique.append((doc, score))

    return unique


def _format_results(query: str, results) -> str:
    total = len(results)
    out = f"### Search Results — `{query}`\n"
    out += f"**{total} relevant chunk(s) found**\n\n"

    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown").strip()
        page = doc.metadata.get("page", "?")
        chunk = doc.page_content.strip()

        score_pct = round(score * 100)
        
        if score >= 0.7:
            badge = f"High — {score_pct}%"
        elif score >= 0.5:
            badge = f"Medium — {score_pct}%"
        else:
            badge = f"Low — {score_pct}%"

        preview = chunk[:400] + ("..." if len(chunk) > 400 else "")

        out += f"---\n"
        out += f"**[{i}]** `{source}` — Page **{page}** — {badge}\n\n"
        out += f"{preview}\n\n"

    return out