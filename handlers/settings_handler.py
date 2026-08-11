import gradio as gr
import config.settings as s

from config.state import set_basic_prompt
from config.prompt_config import _load_saved_prompt, _save_prompt_to_disk
from vectordb.document_manager import get_db_info


def _get_existing_model_ui() -> str | None:
    try:
        return get_db_info().get("embedding_model")
    except Exception:
        return None


def sync_embedding_ui():
    info = get_db_info()
    stored = info.get("embedding_model", "Unknown")
    if stored and stored != "Unknown":
        return gr.update(value=stored)
    return gr.update(value=s.EMBEDDING_MODEL)


def handle_settings_save(chunk_size, chunk_overlap, prompt, embedding_model):
    from embeddings.embeddings import reset_embedding_cache
    from retrieval.retriever import reset_vectorstore_cache

    existing_model = _get_existing_model_ui()
    warning = ""
    if existing_model and existing_model != embedding_model:
        warning = (
            f"⚠️ WARNING: DB was embedded with `{existing_model}`.\n"
            f"New model `{embedding_model}` selected.\n"
            f"Run Delete ALL and re-ingest — otherwise results will be wrong!"
        )

    s.CHUNK_SIZE = int(chunk_size)
    s.CHUNK_OVERLAP = int(chunk_overlap)
    s.EMBEDDING_MODEL = embedding_model
    set_basic_prompt(prompt)
    _save_prompt_to_disk(prompt)  # persist across restarts
    reset_embedding_cache()
    reset_vectorstore_cache()

    status_msg = warning if warning else "✅ Settings saved!"
    return gr.update(visible=False), gr.update(visible=True, value=status_msg), gr.update(value="")


def toggle_settings_panel(is_open: bool, saved: dict):
    new_state = not is_open
    if new_state:
        # opening: restore controls to last saved values — discard any unsaved edits
        return (
            gr.update(visible=True),
            new_state,
            gr.update(value=saved["chunk_size"]),
            gr.update(value=saved["chunk_overlap"]),
            gr.update(value=saved["prompt"]),
            gr.update(value=saved["embedding"]),
        )
    return gr.update(visible=False), new_state, gr.update(), gr.update(), gr.update(), gr.update()


def _cancel_settings(saved: dict):
    # close panel and revert controls to last saved snapshot
    return (
        gr.update(visible=False),
        False,
        gr.update(value=saved["chunk_size"]),
        gr.update(value=saved["chunk_overlap"]),
        gr.update(value=saved["prompt"]),
        gr.update(value=saved["embedding"]),
    )


def sync_embedding_and_state():
    info = get_db_info()
    stored = info.get("embedding_model", "Unknown")
    model = stored if (stored and stored != "Unknown") else s.EMBEDDING_MODEL
    saved_prompt = _load_saved_prompt()
    return gr.update(value=model), gr.update(value={
        "chunk_size": s.CHUNK_SIZE,
        "chunk_overlap": s.CHUNK_OVERLAP,
        "prompt": saved_prompt,
        "embedding": model,
    })