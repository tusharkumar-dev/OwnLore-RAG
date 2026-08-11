from langchain_chroma import Chroma
from embeddings.embeddings import get_embedding_model
from config.settings import DB_PATH, SEARCH_TYPE, TOP_K, COLLECTION_NAME

_vectorstore = None


def reset_vectorstore_cache():
    global _vectorstore
    _vectorstore = None

def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=DB_PATH,
            collection_name=COLLECTION_NAME,
            embedding_function=get_embedding_model(),
        )
    return _vectorstore

def build_retriever(top_k: int = None, filter_source: str = None):
    search_kwargs = {"k": top_k or TOP_K}
    if filter_source:
        search_kwargs["filter"] = {"source": filter_source}
    return _get_vectorstore().as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs=search_kwargs,
    )

def retrieve_with_scores(
    query:         str,
    top_k:         int = None,
    filter_source: str = None,
) -> list[tuple]:
    k     = top_k or TOP_K
    where = None
    if filter_source and filter_source != "All PDFs":
        where = {"source": filter_source.strip()}
    return _get_vectorstore().similarity_search_with_relevance_scores(
        query=query,
        k=k,
        filter=where,
    )


get_retriever = build_retriever