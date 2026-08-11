from typing import Generator
from langchain_core.messages import HumanMessage, SystemMessage
from chromadb import PersistentClient

from llm.llm_provider import load_llm, get_chunk_text
from config.settings import (
    DB_PATH,
    COLLECTION_NAME,
    SUMMARY_MAX_CHUNKS,
    SUMMARY_MAX_CHARS,
)

_SUMMARY_PROMPT = """
You are an expert document analyst.
Analyze the following document chunks and provide a structured summary.

Your response MUST follow this exact format:

QUICK OVERVIEW
Write 3-5 sentences giving a high-level overview of what this document is about.

DETAILED BREAKDOWN
For each major topic or section found in the chunks, write:
[Topic/Section Name]
2-3 sentences describing what this section covers.

KEY CONCEPTS
List 5-7 most important concepts or takeaways from this document.

Keep the summary clear, informative, and well-structured.
Base your summary ONLY on the provided chunks.

Document Chunks:
{chunks}
"""

def summarize_pdf(doc_name: str, provider: str = None) -> Generator[str, None, None]:
    if not doc_name:
        yield "Please select a document first."
        return

    try:
        collection = PersistentClient(path=DB_PATH).get_collection(COLLECTION_NAME)
        if collection.count() == 0:
            yield "DB is empty. Please upload a document first."
            return
    except ValueError:
        yield "No documents ingested yet. Please upload first."
        return
    except Exception as e:
        yield f"DB error: {str(e)}"
        return

    try:
        all_docs = collection.get(
            where={"source": doc_name.strip()},
            include=["documents", "metadatas"],
        )

        if not all_docs["documents"]:
            yield f"No chunks found for '{doc_name}'. Try re-ingesting."
            return

        # sort by page order before slicing
        paired = sorted(
            zip(all_docs["documents"], all_docs["metadatas"]),
            key=lambda x: int(x[1].get("page", 0))
            if str(x[1].get("page", "0")).isdigit()
            else 0,
        )

        total_chunks = len(paired)
        paired = paired[:SUMMARY_MAX_CHUNKS] 

        chunks_text = "\n\n---\n\n".join(
            f"[Page {meta.get('page', '?')}]\n{doc[:SUMMARY_MAX_CHARS]}"
            for doc, meta in paired
        )

    except Exception as e:
        yield f"Error retrieving chunks: {str(e)}"
        return

    try:
        llm = load_llm(provider=provider)
    except Exception as e:
        yield f"LLM load error: {str(e)}"
        return

    header = f"## Summary — {doc_name}\n"
    header += f"*{total_chunks} total chunks — top {len(paired)} analyzed*\n\n"
    yield header

    # stream the actual summary
    prompt = _SUMMARY_PROMPT.format(chunks=chunks_text)
    messages = [
        SystemMessage(content="You are an expert document analyst. Be concise and structured."),
        HumanMessage(content=prompt),
    ]

    acc = header
    try:
        for chunk in llm.stream(messages):
            acc += get_chunk_text(chunk)
            yield acc

    except Exception as e:
        yield acc + f"\n\nStream error: {str(e)}"