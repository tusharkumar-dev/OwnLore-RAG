from langchain_chroma import Chroma
from langchain_core.documents import Document as LangChainDoc
from embeddings.embeddings import get_embedding_model
from config.settings import DB_PATH, COLLECTION_NAME

_CHROMA_BATCH_SIZE = 5000


def _get_vectorstore() -> Chroma:
    # fresh client each time
    import chromadb
    client = chromadb.PersistentClient(path=DB_PATH)
    client.get_or_create_collection(COLLECTION_NAME)

    return Chroma(
        persist_directory=DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_model(),
    )


def _convert_to_langchain(llamaindex_docs) -> list[LangChainDoc]:
    # LlamaIndex Documents → LangChain Documents
    langchain_docs = []
    for doc in llamaindex_docs:
        if isinstance(doc, str):
            text = doc.strip()
        else:
            text = (doc.text or "").strip()

        if not text:
            continue

        meta = doc.metadata if hasattr(doc, "metadata") else {}
        langchain_docs.append(LangChainDoc(
            page_content=text,
            metadata={
                "source": meta.get("file_name", "Unknown"),
                "page": str(meta.get("page_label", "?")),
            }
        ))
    return langchain_docs


def _get_stored_model() -> str | None:
    try:
        import chromadb
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(COLLECTION_NAME)

        if collection.count() == 0:
            return None  # empty collection, skip model check

        return (collection.metadata or {}).get("embedding_model")
    except Exception:
        return None


def create_vectorstore(chunks):
    import config.settings as s

    existing_model = _get_stored_model()
    if existing_model and existing_model != s.EMBEDDING_MODEL:
        raise ValueError(
            f"Embedding model mismatch!\n"
            f"DB has: {existing_model}\n"
            f"Current: {s.EMBEDDING_MODEL}\n"
            f"Delete chroma_db and re-ingest."
        )

    vectorstore = _get_vectorstore()

    try:
        vectorstore._collection.modify(metadata={"embedding_model": s.EMBEDDING_MODEL})
    except Exception:
        pass

    langchain_docs = _convert_to_langchain(chunks)
    if not langchain_docs:
        print("[VECTORSTORE] No valid chunks found.")
        return vectorstore

    total = len(langchain_docs)
    for i in range(0, total, _CHROMA_BATCH_SIZE):
        batch = langchain_docs[i: i + _CHROMA_BATCH_SIZE]
        vectorstore.add_documents(batch)
        print(f"[VECTORSTORE] Batch {i // _CHROMA_BATCH_SIZE + 1}: {len(batch)} chunks")

    print(f"[VECTORSTORE] Done. {total} chunks saved.")
    return vectorstore