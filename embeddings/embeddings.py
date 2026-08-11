from langchain_huggingface import HuggingFaceEmbeddings
import config.settings as s

_embedding_model = None

def _sync_model_from_db():
   
    try:
        import chromadb
        client     = chromadb.PersistentClient(path=s.DB_PATH)
        collection = client.get_collection(s.COLLECTION_NAME)
        stored     = (collection.metadata or {}).get("embedding_model")
        if stored and stored != s.EMBEDDING_MODEL:
            print(f"[EMBEDDINGS] DB model: {stored} — settings update ho raha hai")
            s.EMBEDDING_MODEL = stored
    except Exception:
        pass


def reset_embedding_cache():
    global _embedding_model
    _embedding_model = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        _sync_model_from_db()
        _embedding_model = HuggingFaceEmbeddings(
            model_name=s.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_model