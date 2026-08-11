from typing import Generator

from chromadb import PersistentClient
from langchain_core.messages import HumanMessage, SystemMessage

from chat.source_formatter import format_sources
from config.settings import COLLECTION_NAME, DB_PATH, TOP_K
from llm.llm_provider import load_llm, get_chunk_text
from llm.llm_provider import load_llm
from retrieval.reranker import rerank
from retrieval.retriever import build_retriever


_SYSTEM_PROMPT_TEMPLATE = """
You are a helpful and knowledgeable assistant.
Answer the user's question ONLY using the provided context.
Do NOT use any external knowledge.
If the answer is not present in the context, respond:
"I don't know based on the provided document."
Keep your answer:
- Clear and concise
- Well-structured (use bullet points if helpful)

Context:
{context}
"""

_MAX_CONTEXT_CHARS = 4000

_llm_cache: dict = {}


def _get_llm(provider=None):
    key = provider or "default"
    if key not in _llm_cache:
        _llm_cache[key] = load_llm(provider=provider)
    return _llm_cache[key]


def _normalize_pdf(selected_pdf=None):
    # "All PDFs" comes from the dropdown — treat it same as None
    if not selected_pdf or selected_pdf.strip() == "All PDFs":
        return None
    return selected_pdf.strip()


def _check_db() -> tuple[bool, str]:
    try:
        client = PersistentClient(path=DB_PATH)
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() == 0:
            return False, "DB is empty. Please upload a PDF first."
        return True, ""
    except ValueError:
        return False, "No documents ingested yet. Please upload a PDF first."
    except Exception as e:
        return False, f"DB error: {str(e)}"


def _build_context(docs) -> str:
    parts = []
    for doc in docs:
        chunk = doc.page_content.strip()
        if len(chunk) > _MAX_CONTEXT_CHARS:
            chunk = chunk[:_MAX_CONTEXT_CHARS] + "..."
        parts.append(chunk)
    return "\n\n".join(parts)


def answer_question(
    question: str,
    history,
    selected_pdf=None,
    provider=None,
    top_k=None,
) -> Generator[tuple[str, str], None, None]:

    db_ok, db_err = _check_db()
    if not db_ok:
        yield db_err, ""
        return

    pdf_filter = _normalize_pdf(selected_pdf)
    actual_k = top_k or TOP_K
    retrieve_k = actual_k * 2  # oversample so reranker has something to work with

    try:
        retriever = build_retriever(top_k=retrieve_k, filter_source=pdf_filter)
        docs = retriever.invoke(question)
    except Exception as e:
        yield f"Retrieval error: {e}", ""
        return

    if not docs:
        msg = f"No answer found in '{pdf_filter}'." if pdf_filter else "No relevant answer found."
        yield msg, ""
        return

    # rerank — if it fails just fall back to top-k slice, not worth crashing over
    try:
        reranked = rerank(question, [(doc, 1.0) for doc in docs], top_n=actual_k)
        docs = [doc for doc, _ in reranked]
    except Exception:
        docs = docs[:actual_k]

    if not docs:
        yield "No relevant answer found.", ""
        return

    # build context and kick off the stream
    sources_md = format_sources(docs)
    context = _build_context(docs)
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(context=context)

    llm = _get_llm(provider)
    acc = ""

    # history is currently ignored.

    try:
        for chunk in llm.stream([
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]):
            acc += get_chunk_text(chunk)
            yield acc, sources_md
            # print("chunk:", chunk.content)

    except KeyboardInterrupt:
        if acc:
            yield acc, sources_md

    except Exception as e:
        # Partial responses are lost if streaming stops unexpectedly.
        yield acc + f"\n\nStream error: {e}", sources_md