from langchain_core.documents import Document

# Keep previews short enough to stay readable.
CHUNK_PREVIEW_LENGTH = 400


def format_sources(docs: list[Document]) -> str:
    if not docs:
        return "No sources found."

    seen   = set()
    result = "### Sources & Similar Chunks\n\n"

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown").strip()
        page   = doc.metadata.get("page", "?")
        chunk  = doc.page_content.strip()

        key = f"{source}_p{page}_{i}"
        if key in seen:
            continue

        seen.add(key)
        result += f"---\n**[{i}] {source}** — Page {page}\n\n"
        result += f"```\n{chunk[:CHUNK_PREVIEW_LENGTH]}\n```\n\n"

    return result