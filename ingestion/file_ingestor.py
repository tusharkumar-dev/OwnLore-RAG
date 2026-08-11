from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Optional, Type

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.string_iterable import StringIterableReader

from llama_index.readers.file import (
    DocxReader,
    EpubReader,
    IPYNBReader,
    ImageReader,
    MboxReader,
    MarkdownReader,
    PDFReader,
    PandasCSVReader,
    PptxReader,
    VideoAudioReader,
)
from llama_index.readers.json import JSONReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingestion_helper")


# extension -> reader class
FILE_READER_CLS: dict[str, Type[BaseReader]] = {
    ".pdf": PDFReader,
    ".docx": DocxReader,
    ".pptx": PptxReader,
    ".ppt": PptxReader,
    ".pptm": PptxReader,
    ".hwp": DocxReader, 
    ".jpg": ImageReader,
    ".jpeg": ImageReader,
    ".png": ImageReader,
    ".csv": PandasCSVReader,
    ".json": JSONReader,
    ".md": MarkdownReader,
    ".epub": EpubReader,
    ".mbox": MboxReader,
    ".ipynb": IPYNBReader,
    ".mp3": VideoAudioReader,
    ".mp4": VideoAudioReader,
}

DOC_TYPE_MAP: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "word",
    ".pptx": "powerpoint",
    ".ppt": "powerpoint",
    ".pptm": "powerpoint",
    ".hwp": "hwp",
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".csv": "csv",
    ".json": "json",
    ".md": "markdown",
    ".epub": "epub",
    ".mbox": "mbox",
    ".ipynb": "notebook",
    ".mp3": "audio",
    ".mp4": "video",
}

EXCLUDED_EMBED_META = ["doc_id", "file_name", "source", "document_type", "word_count", "char_count"]
EXCLUDED_LLM_META = ["doc_id", "file_name", "source", "word_count", "char_count"]


def _process_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _enrich_metadata(doc: Document, file_name: str, file_data: Path, doc_id: str) -> Document:
    ext = file_data.suffix.lower()
    existing = doc.metadata or {}

    page_label = str(existing.get("page_label", ""))
    try:
        page_no = int(page_label)
    except (ValueError, TypeError):
        page_no = 0

    txt = doc.get_content() or ""
    wc = len(txt.split())
    cc = len(txt)

    doc.metadata = {
        **existing,
        "doc_id": doc_id,
        "file_name": file_name,
        "source": str(file_data.resolve()),
        "document_type": DOC_TYPE_MAP.get(ext, "unknown"),
        "page_no": page_no,
        "page_label": page_label,
        "word_count": wc,
        "char_count": cc,
    }

    doc.excluded_embed_metadata_keys = EXCLUDED_EMBED_META.copy()
    doc.excluded_llm_metadata_keys = EXCLUDED_LLM_META.copy()

    return doc


def _get_reader(ext: str) -> BaseReader:
    cls = FILE_READER_CLS.get(ext)
    if cls is None:
        logger.warning("No reader for '%s', falling back to StringIterableReader", ext)
        return StringIterableReader()
    logger.info("Using %s for '%s'", cls.__name__, ext)
    return cls()


def _load_documents(reader: BaseReader, file_data: Path) -> list[Document]:
    try:
        if isinstance(reader, StringIterableReader):
            raw = file_data.read_text(encoding="utf-8", errors="replace")
            docs = reader.load_data(texts=[raw])
        else:
            docs = reader.load_data(file=file_data)

        logger.info("%s loaded %d doc(s) from '%s'", reader.__class__.__name__, len(docs), file_data.name)
        return docs

    except Exception as exc:
        raise RuntimeError(
            "Reader '{}' failed on '{}': {}".format(reader.__class__.__name__, file_data.name, exc)
        ) from exc


def _validate_file(file_name: str, file_data: Path) -> None:
    if not file_data.exists():
        raise FileNotFoundError(f"File not found: '{file_data}'")
    if not file_data.is_file():
        raise ValueError(f"Not a regular file: '{file_data}'")
    ext = file_data.suffix.lower()
    if ext not in FILE_READER_CLS:
        logger.warning("'%s' has unsupported extension '%s', will attempt fallback", file_name, ext)


def extract_documents(file_name: str, file_data: Path) -> list[Document]:
   
    file_data = Path(file_data).resolve()
    ext = file_data.suffix.lower()

    logger.info("Starting ingestion: '%s'", file_name)
    _validate_file(file_name, file_data)

    reader = _get_reader(ext)

    try:
        raw_docs = _load_documents(reader, file_data)
    except RuntimeError as exc:
        logger.error("Load failed for '%s': %s", file_name, exc)
        raise

    if not raw_docs:
        logger.warning("Nothing extracted from '%s'", file_name)
        return []

    doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(file_data)))
    out: list[Document] = []

    for doc in raw_docs:
        orig = doc.get_content() or ""
        cleaned = _process_text(orig)

        if not cleaned:
            logger.debug("Skipping empty chunk in '%s'", file_name)
            continue

        logger.debug("Cleaned: %d -> %d chars", len(orig), len(cleaned))

        clean_doc = Document(text=cleaned, metadata=doc.metadata or {})
        clean_doc = _enrich_metadata(clean_doc, file_name, file_data, doc_id)
        out.append(clean_doc)

    logger.info("Done '%s' — %d doc(s) returned", file_name, len(out))
    return out