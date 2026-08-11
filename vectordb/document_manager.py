import os
import gc
import re
import shutil
import time
from pathlib import Path

import chromadb
from config.settings import DB_PATH, COLLECTION_NAME


def _get_client() -> chromadb.PersistentClient:
    # fresh client every time — module-level client goes stale after deletes
    return chromadb.PersistentClient(path=DB_PATH)


def list_all_pdfs() -> list[list[str]]:
    try:
        collection = _get_client().get_collection(COLLECTION_NAME)
        data = collection.get(include=["metadatas"])

        names: set[str] = set()
        for meta in data["metadatas"]:
            if meta and "source" in meta:
                names.add(meta["source"].strip())

        return [[name] for name in sorted(names)]

    except Exception:
        return []


def is_pdf_already_ingested(pdf_name: str) -> bool:
    try:
        collection = _get_client().get_collection(COLLECTION_NAME)
        data = collection.get(
            where={"source": pdf_name.strip()},
            include=["metadatas"],
        )
        return len(data["ids"]) > 0
    except Exception:
        return False


def get_db_info() -> dict:
    try:
        collection = _get_client().get_collection(COLLECTION_NAME)
        meta = collection.metadata or {}
        return {
            "embedding_model": meta.get("embedding_model", "Unknown"),
            "total_chunks": collection.count(),
        }
    except Exception:
        return {"embedding_model": "Unknown", "total_chunks": 0}


def delete_pdf(selected_pdf: str) -> tuple[list[list[str]], str]:
    if not selected_pdf:
        return list_all_pdfs(), "No file selected."

    selected_pdf = selected_pdf.strip()

    try:
        collection = _get_client().get_collection(COLLECTION_NAME)
        data = collection.get(include=["metadatas"])

        ids_to_delete = [
            doc_id
            for doc_id, meta in zip(data["ids"], data["metadatas"])
            if meta and "source" in meta
            and meta["source"].strip() == selected_pdf
        ]

        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            return list_all_pdfs(), f"Deleted: {selected_pdf}"

        return list_all_pdfs(), f"File not found: {selected_pdf}"

    except Exception as e:
        return list_all_pdfs(), f"Delete error: {str(e)}"


def _is_uuid_folder(name: str) -> bool:
    return bool(re.match(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        name.lower()
    ))


def delete_all_pdfs() -> tuple[list, str]:
    from retrieval.retriever import reset_vectorstore_cache
    from embeddings.embeddings import reset_embedding_cache

    def _force_close_chroma():
        # clears chromadb singleton connections
        try:
            import chromadb.api
            chromadb.api.client.SharedSystemClient.clear_system_cache()
        except Exception:
            pass

    def _try_delete_folder(folder_path: Path, retries: int = 5, delay: float = 1.0) -> bool:
        # retry loop for Windows — file locks don't release instantly after chroma closes
        for attempt in range(retries):
            try:
                shutil.rmtree(folder_path)
                print(f"[DB] deleted: {folder_path.name}")
                return True
            except PermissionError:
                print(f"[DB] attempt {attempt + 1}/{retries} failed, waiting {delay}s...")
                gc.collect()
                time.sleep(delay)
        print(f"[DB] gave up after {retries} attempts: {folder_path.name}")
        return False

    try:
        reset_vectorstore_cache()
        reset_embedding_cache()

        client = chromadb.PersistentClient(path=DB_PATH)
        try:
            for col in client.list_collections():
                client.delete_collection(col.name)
                print(f"[DB] collection deleted: {col.name}")
        except Exception as e:
            print(f"[DB] collection delete error: {e}")

        # shut down client before touching the filesystem — order matters here
        try:
            client._system.stop()
        except Exception:
            pass

        client = None
        _force_close_chroma()
        gc.collect()
        time.sleep(1.5)

        db_path = Path(DB_PATH)
        if db_path.exists():
            for item in db_path.iterdir():
                if item.is_dir() and _is_uuid_folder(item.name):
                    _try_delete_folder(item, retries=5, delay=1.0)

        time.sleep(0.3)

        fresh = chromadb.PersistentClient(path=DB_PATH)
        fresh.get_or_create_collection(COLLECTION_NAME)
        fresh = None

        reset_vectorstore_cache()
        reset_embedding_cache()

        return [], "All documents deleted."

    except Exception as e:
        return [], f"Error during delete: {str(e)}"