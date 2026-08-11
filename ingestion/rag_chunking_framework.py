from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from llama_index.core.schema import Document, TextNode
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from llama_index.core.node_parser import (
    SentenceSplitter,
    MarkdownElementNodeParser,
    SemanticSplitterNodeParser,
    CodeSplitter,
    JSONNodeParser,
    HierarchicalNodeParser,
    get_leaf_nodes,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  [%(levelname)s]  %(message)s")
logger = logging.getLogger("RAGChunker")

class ChunkStrategy(str, Enum):
    SENTENCE = "SentenceSplitter"
    MARKDOWN = "MarkdownElementNodeParser"
    HIERARCHICAL = "HierarchicalNodeParser"
    SEMANTIC = "SemanticSplitterNodeParser"
    CODE = "CodeSplitter"
    JSON = "JSONNodeParser"
    SENTENCE_FIXED = "SentenceSplitter_FixedSize"

class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    MARKDOWN = "markdown"
    TXT = "txt"
    JSON = "json"
    CSV = "csv"
    UNKNOWN = "unknown"


@dataclass
class ChunkConfig:
    chunk_size: int = -1  
    chunk_overlap: int = -1

    def __post_init__(self):
        import config.settings as s
        if self.chunk_size == -1:
            self.chunk_size = s.CHUNK_SIZE
        if self.chunk_overlap == -1:
            self.chunk_overlap = s.CHUNK_OVERLAP

    hier_chunk_sizes: list[int] = field(default_factory=lambda: [2048, 1024, 650])
    default_code_language: str = "python"
    semantic_buffer_size: int = 1
    semantic_breakpoint_threshold: int = 95
    include_summary: bool = True

class StrategyRouter:

    _BOOK_HINTS = re.compile(
        r"\b(chapter|section|part\s+\d|table of contents|appendix)\b",
        re.IGNORECASE,
    )
    _CODE_HINTS = re.compile(r"(def |class |import |#include|function |<\?php|public static)")

    @classmethod
    def route(cls, doc: Document, force_strategy: ChunkStrategy | None = None, use_semantic: bool = False) -> ChunkStrategy:
        if force_strategy:
            return force_strategy

        dtype = cls._get_doc_type(doc)
        text = (doc.text or "").strip()

        if dtype == DocumentType.JSON:
            return ChunkStrategy.JSON
        if dtype == DocumentType.CSV:
            return ChunkStrategy.SENTENCE_FIXED
        if dtype in (DocumentType.MARKDOWN, DocumentType.PPTX):
            return ChunkStrategy.MARKDOWN
        if cls._CODE_HINTS.search(text):
            return ChunkStrategy.CODE

        if dtype == DocumentType.PDF:
            wc = doc.metadata.get("word_count", 0)
            if wc > 5000 and cls._BOOK_HINTS.search(text):
                return ChunkStrategy.HIERARCHICAL
            if use_semantic:
                return ChunkStrategy.SEMANTIC

        return ChunkStrategy.SENTENCE

    @staticmethod
    def _get_doc_type(doc: Document) -> DocumentType:
        raw = (doc.metadata.get("document_type", "") or doc.metadata.get("file_name", "")).lower()

        # File type detection currently relies on the filename extension.
        mapping = {
            "pdf": DocumentType.PDF,
            "docx": DocumentType.DOCX,
            "pptx": DocumentType.PPTX,
            "markdown": DocumentType.MARKDOWN,
            "md": DocumentType.MARKDOWN,
            "txt": DocumentType.TXT,
            "json": DocumentType.JSON,
            "csv": DocumentType.CSV,
        }
        for key, dtype in mapping.items():
            if key in raw:
                return dtype
        return DocumentType.UNKNOWN

class SplitterFactory:

    @staticmethod
    def build(strategy: ChunkStrategy, cfg: ChunkConfig, embed_model=None, code_language: str | None = None):
        if strategy == ChunkStrategy.SENTENCE:
            return SentenceSplitter(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)

        elif strategy == ChunkStrategy.SENTENCE_FIXED:
         
            return SentenceSplitter(chunk_size=min(cfg.chunk_size, 512), chunk_overlap=cfg.chunk_overlap)

        elif strategy == ChunkStrategy.MARKDOWN:
            return MarkdownElementNodeParser()

        elif strategy == ChunkStrategy.HIERARCHICAL:
            return HierarchicalNodeParser.from_defaults(chunk_sizes=cfg.hier_chunk_sizes)

        elif strategy == ChunkStrategy.SEMANTIC:
            if embed_model is None:
                logger.warning("No embed_model provided. Falling back to SentenceSplitter.")
                return SentenceSplitter(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
            return SemanticSplitterNodeParser(
                embed_model=embed_model,
                buffer_size=cfg.semantic_buffer_size,
                breakpoint_percentile_threshold=cfg.semantic_breakpoint_threshold,
            )

        elif strategy == ChunkStrategy.CODE:
            return CodeSplitter(
                language=code_language or cfg.default_code_language,
                chunk_lines=50,
                chunk_lines_overlap=8,
                max_chars=cfg.chunk_size * 4,
            )

        elif strategy == ChunkStrategy.JSON:
            return JSONNodeParser()

        raise ValueError(f"Unknown strategy: {strategy}")

def _enrich_chunk(node: TextNode, chunk_index: int, strategy: ChunkStrategy, cfg: ChunkConfig, parent_doc: Document) -> dict[str, Any]:
    text = (node.text or "").strip()

    metadata = dict(parent_doc.metadata)
    if node.metadata.get("page_label"):
        metadata["page_label"] = node.metadata["page_label"]
    if node.metadata.get("page_no"):
        metadata["page_no"] = node.metadata["page_no"]

    return {
        "chunk_id": chunk_index,
        "chunk_uuid": str(uuid.uuid4()),
        "text": text,
        "metadata": metadata,
    }

def _validate_chunks(chunks: list[dict], cfg: ChunkConfig) -> list[dict]:
    seen: set[str] = set()
    clean: list[dict] = []

    for c in chunks:
        txt = c["text"]
        if not txt or not txt.strip():
            continue

        h = str(hash(txt.strip()))
        if h in seen:
            logger.debug("Removing duplicate chunk: %r", txt[:60])
            continue
        seen.add(h)
        clean.append(c)

    return clean


class RAGChunker:

    _BLANK_PAGE = re.compile(r"^[\s\W_]*$")

    def __init__(self, config: ChunkConfig | None = None):
        self.cfg = config or ChunkConfig()

    def chunk(
        self,
        documents: list[Document],
        force_strategy: ChunkStrategy | None = None,
        embed_model=None,
        use_semantic: bool = False,
    ) -> list[dict[str, Any]]:
        if not documents:
            return []

        all_chunks: list[dict] = []
        chunk_counter = 0

        for doc in documents:
            text = (doc.text or "").strip()

            # drop blank / symbol-only pages
            if not text or self._BLANK_PAGE.match(text):
                logger.debug("Skipping blank page: %s", doc.metadata.get("page_label"))
                continue

            strategy = StrategyRouter.route(doc, force_strategy, use_semantic)
            nodes = self._parse(doc, strategy, embed_model)

            for node in nodes:
                node_text = (node.text or "").strip()
                if not node_text:
                    continue  # only empty nodes are skipped here.

                chunk_counter += 1
                all_chunks.append(_enrich_chunk(node, chunk_counter, strategy, self.cfg, doc))

        return all_chunks

    def _parse(self, doc: Document, strategy: ChunkStrategy, embed_model) -> list[TextNode]:
        try:
            splitter = SplitterFactory.build(strategy, self.cfg, embed_model=embed_model)
        except Exception:
    
            splitter = SentenceSplitter(chunk_size=self.cfg.chunk_size, chunk_overlap=self.cfg.chunk_overlap)

        try:
            if strategy == ChunkStrategy.HIERARCHICAL:
                return get_leaf_nodes(splitter.get_nodes_from_documents([doc]))
            elif strategy == ChunkStrategy.MARKDOWN:
        
                return [n for n in splitter.get_nodes_from_documents([doc]) if isinstance(n, TextNode)]
            else:
                return splitter.get_nodes_from_documents([doc])
        except Exception:

            return SentenceSplitter(
                chunk_size=self.cfg.chunk_size,
                chunk_overlap=self.cfg.chunk_overlap,
            ).get_nodes_from_documents([doc])

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3) 


def chunk_documents_as_docs(
    documents: list[Document],
    config: ChunkConfig | None = None,
    force_strategy: ChunkStrategy | None = None,
    embed_model=None,
    use_semantic: bool = False,
    run_validation: bool = True,
) -> list[Document]:
    import config.settings as s
    cfg = config or ChunkConfig(
        chunk_size=s.CHUNK_SIZE,
        chunk_overlap=s.CHUNK_OVERLAP,
    )

    chunks = RAGChunker(config=cfg).chunk(
        documents,
        force_strategy=force_strategy,
        embed_model=embed_model,
        use_semantic=use_semantic,
    )

    if run_validation:
        chunks = _validate_chunks(chunks, cfg)

    parent_map = {doc.metadata.get("doc_id"): doc for doc in documents}

    result = []
    for chunk in chunks:
        parent_id = chunk["metadata"].get("doc_id", "")
        parent_doc = parent_map.get(parent_id, documents[0])
        result.append(
            Document(
                id_=chunk["chunk_uuid"],
                text=chunk["text"],
                metadata=chunk["metadata"],
                excluded_embed_metadata_keys=parent_doc.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=parent_doc.excluded_llm_metadata_keys,
            )
        )
    return result


def load_and_chunk(
    file_name: str,
    file_data: Path,
    config: ChunkConfig | None = None,
    force_strategy: ChunkStrategy | None = None,
    embed_model=None,
    use_semantic: bool = False,
) -> list[Document]:
    
    from file_ingestor import extract_documents

    docs = extract_documents(file_name=file_name, file_data=Path(file_data))

    if not docs:
        logger.warning("load_and_chunk: no documents loaded from '%s'", file_name)
        return []

    return chunk_documents_as_docs(
        documents=docs,
        config=config,
        force_strategy=force_strategy,
        embed_model=embed_model,
        use_semantic=use_semantic,
    )