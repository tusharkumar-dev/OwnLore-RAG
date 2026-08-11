import threading
import gradio as gr
from pathlib import Path

from storage.vector_store import create_vectorstore
from vectordb.document_manager import (
    list_all_pdfs, delete_pdf, delete_all_pdfs,
    is_pdf_already_ingested,
)
from ingestion.file_ingestor import extract_documents
from ingestion.rag_chunking_framework import chunk_documents_as_docs

# module-level state shared with chat_handler
_selected_pdf = None


def get_selected_pdf():
    return _selected_pdf


def set_selected_pdf(value):
    global _selected_pdf
    _selected_pdf = value


def handle_select(select_data: gr.SelectData):
    global _selected_pdf
    _selected_pdf = select_data.value.strip()
    short = (_selected_pdf[:48] + "…") if len(_selected_pdf) > 50 else _selected_pdf
    return (
        f"## 💬 Chat\n"
        f"<div title='{_selected_pdf}' style='"
        f"display:inline-flex;align-items:center;gap:6px;"
        f"background:#1565c0;color:#ffffff;"
        f"padding:3px 10px;border-radius:12px;"
        f"font-size:0.82em;font-weight:600;margin-top:2px;"
        f"max-width:100%;overflow:hidden;white-space:nowrap;'>"
        f"📄 {short}"
        f"</div>"
    )


def handle_deselect():
    global _selected_pdf
    _selected_pdf = None
    return "## 💬 Chat\n📂 **All Documents**"


def _progress_bar(current: int, total: int, width: int = 18) -> str:
    if total == 0:
        return "░" * width + "  0% • 0 / 0 chunks"
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    pct = int(100 * current / total)
    return f"`{bar}`  **{pct}%** • {current} / {total} chunks"


def _run_vectorstore_with_progress(chunks):
    import time
    total = max(len(chunks), 1)
    exc_holder = {}

    def _worker():
        try:
            create_vectorstore(chunks)
        except Exception as e:
            exc_holder["err"] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    secs_per_chunk = 0.05
    start = time.time()
    while t.is_alive():
        elapsed = time.time() - start
        estimated = min(int(elapsed / secs_per_chunk), int(total * 0.95))
        yield _progress_bar(estimated, total)
        time.sleep(0.4)

    t.join()
    if "err" in exc_holder:
        raise exc_holder["err"]
    yield _progress_bar(total, total)


def handle_ingest(files):
    skipped, ingested = [], []

    for file in files:
        doc_name = Path(file).name

        if is_pdf_already_ingested(doc_name):
            skipped.append(doc_name)
            yield gr.update(), f"⚠️ **{doc_name}** already exists — skipped", gr.update()
            continue

        yield gr.update(), f"📄 **{doc_name}**\n━━━━━━━━━━━━\n✅ Uploaded\n⏳ Reading PDF...", "🟡 Extracting Text..."

        try:
            docs = extract_documents(doc_name, Path(file))
        except Exception as e:
            yield list_all_pdfs(), f"❌ Extraction failed\n\n{e}", ""
            return

        yield gr.update(), f"📄 **{doc_name}**\n━━━━━━━━━━━━\n✅ Uploaded\n✅ Text Extracted\n⏳ Creating Chunks...", "🟡 Chunking..."

        try:
            chunks = chunk_documents_as_docs(docs)
        except Exception as e:
            yield list_all_pdfs(), f"❌ Chunking failed\n\n{e}", ""
            return

        n = len(chunks)
        stage_prefix = f"📄 **{doc_name}**\n━━━━━━━━━━━━\n✅ Uploaded\n✅ Text Extracted\n✅ Chunked ({n} chunks)"
        yield gr.update(), stage_prefix, "🟡 Creating Embeddings..."

        try:
            for bar in _run_vectorstore_with_progress(chunks):
                yield (
                    gr.update(),
                    f"{stage_prefix}\n🧠 Creating Embeddings\n{bar}\n□ Saving to Vector DB",
                    "🟡 Saving to Vector DB...",
                )
        except Exception as e:
            yield list_all_pdfs(), f"❌ Embedding failed\n\n{e}", ""
            return

        yield (
            gr.update(),
            f"{stage_prefix}\n✅ Embeddings Created\n✅ Saved to Vector DB",
            "",
        )
        ingested.append(doc_name)

    parts = []
    if ingested:
        for doc_name in ingested:
            short = (doc_name[:38] + "…") if len(doc_name) > 40 else doc_name
            parts.append(
                f"<div title='{doc_name}' style='background:#e8f5e9;border-left:3px solid #43a047;"
                f"padding:4px 10px;border-radius:3px;margin:2px 0;font-size:0.84em;"
                f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#1b5e20;'>"
                f"✅ Indexed: {short}</div>"
            )
    if skipped:
        for doc_name in skipped:
            short = (doc_name[:35] + "…") if len(doc_name) > 37 else doc_name
            parts.append(
                f"<div title='{doc_name}' style='background:#fff8e1;border-left:3px solid #f9a825;"
                f"padding:4px 10px;border-radius:3px;margin:2px 0;font-size:0.84em;"
                f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#5d4037;'>"
                f"⚠️ Already exists: {short}</div>"
            )

    inner = "\n".join(parts)
    final_notice = (
        f"<div style='animation:fadeOutNotice 0.5s ease 4s forwards;'>"
        f"{inner}"
        f"</div>"
        f"<style>"
        f"@keyframes fadeOutNotice{{0%{{opacity:1;max-height:120px;}}"
        f"100%{{opacity:0;max-height:0;overflow:hidden;padding:0;margin:0;}}}}"
        f"</style>"
    )
    yield list_all_pdfs(), final_notice, ""


def handle_delete_selected():
    global _selected_pdf
    result, msg = delete_pdf(_selected_pdf)
    _selected_pdf = None
    return result, msg or "", "", "## 💬 Chat\n📂 **All Documents**"


def handle_delete_all():
    global _selected_pdf
    _selected_pdf = None
    result, msg = delete_all_pdfs()
    return result, msg or "", "", "## 💬 Chat\n📂 **All Documents**"