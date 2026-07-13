from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

from app.agent.config import AgentConfig
from app.agent.schemas import DocumentChunk

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]*")


def normalize_condition_tag(condition: str | None) -> str:
    if not condition:
        return "general"
    normalized = condition.lower().replace(" ", "_")
    if "diabetic" in normalized or "dfu" in normalized or "wagner" in normalized:
        return "dfu"
    if "pressure" in normalized or "npuap" in normalized or "epuap" in normalized:
        return "pi"
    return "general"


def document_tag_from_path(path: Path) -> str:
    return normalize_condition_tag(path.name)


def normalize_doc_id(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]+", "_", value).lower()).strip("_")


def source_doc_id(source: str | None) -> str:
    if not source:
        return ""
    return Path(source).stem


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def read_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ImportError("pypdf is required to ingest PDF clinical documents.") from e
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix in {".txt", ".md"}:
        return path.read_text(errors="ignore")
    return ""


class HashEmbeddingFunction:
    """
    Deterministic local embeddings for ChromaDB demos without an external model.

    This is not a semantic embedding model. It is a portable retrieval fallback
    until OpenAI embeddings or sentence-transformers are configured.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def __call__(self, input: list[str]) -> list[list[float]]:  # Chroma expects this name.
        return [self.embed_text(text) for text in input]

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class ClinicalMemory:
    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig.from_env()
        self.collection_name = "wound_clinical_docs"
        self.index_path = self.config.chroma_db_path / "wound_clinical_docs.json"
        self.embedding_function = HashEmbeddingFunction()
        self._chroma_collection = None
        self.backend = "json"
        self._init_chroma_if_available()

    def _init_chroma_if_available(self) -> None:
        try:
            import chromadb
        except ImportError:
            return

        self.config.chroma_db_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.config.chroma_db_path))
        self._chroma_collection = client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
        )
        self.backend = "chroma"

    def ensure_index(self) -> dict:
        if self.backend == "chroma" and self._chroma_collection is not None:
            if self._chroma_collection.count() > 0:
                return {"backend": self.backend, "chunks": self._chroma_collection.count()}
        elif self.index_path.exists():
            data = json.loads(self.index_path.read_text())
            if data.get("chunks"):
                return {"backend": self.backend, "chunks": len(data["chunks"])}

        return self.ingest_documents(self.config.clinical_docs_path)

    def ingest_documents(self, docs_dir: Path | None = None) -> dict:
        docs_dir = docs_dir or self.config.clinical_docs_path
        docs_dir.mkdir(parents=True, exist_ok=True)
        docs = sorted(
            path
            for path in docs_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"}
        )

        chunks: list[dict] = []
        for path in docs:
            text = read_document_text(path)
            condition_tag = document_tag_from_path(path)
            for index, chunk in enumerate(chunk_text(text), start=1):
                chunks.append(
                    {
                        "id": f"{path.stem}-{index}",
                        "text": chunk,
                        "source": path.name,
                        "doc_id": path.stem,
                        "section": f"chunk-{index}",
                        "condition_tag": condition_tag,
                    }
                )

        self.config.chroma_db_path.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps({"chunks": chunks}, indent=2))

        if self.backend == "chroma" and self._chroma_collection is not None and chunks:
            existing = set(self._chroma_collection.get().get("ids", []))
            new_chunks = [chunk for chunk in chunks if chunk["id"] not in existing]
            if new_chunks:
                self._chroma_collection.add(
                    ids=[chunk["id"] for chunk in new_chunks],
                    documents=[chunk["text"] for chunk in new_chunks],
                    metadatas=[
                        {
                            "source": chunk["source"],
                            "doc_id": chunk["doc_id"],
                            "section": chunk["section"],
                            "condition_tag": chunk["condition_tag"],
                        }
                        for chunk in new_chunks
                    ],
                )

        return {
            "backend": self.backend,
            "documents": len(docs),
            "chunks": len(chunks),
            "docs_dir": str(docs_dir),
            "index_path": str(self.index_path),
        }

    def retrieve(
        self,
        query: str,
        condition: str | None = None,
        top_k: int = 4,
        allowed_doc_ids: Sequence[str] | None = None,
    ) -> list[DocumentChunk]:
        self.ensure_index()
        condition_tag = normalize_condition_tag(condition)
        doc_id_filter = (
            {normalize_doc_id(doc_id) for doc_id in allowed_doc_ids if doc_id}
            if allowed_doc_ids is not None
            else None
        )
        if allowed_doc_ids is not None and not doc_id_filter:
            return []

        if self.backend == "chroma" and self._chroma_collection is not None:
            where = {"condition_tag": condition_tag} if condition_tag != "general" else None
            result = self._chroma_collection.query(
                query_texts=[query],
                n_results=max(top_k, min(top_k * 5, 20)),
                where=where,
            )
            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]
            if docs:
                chunks = []
                for doc, meta, distance in zip(docs, metas, distances):
                    meta = meta or {}
                    doc_id = meta.get("doc_id") or source_doc_id(meta.get("source"))
                    if doc_id_filter is not None and normalize_doc_id(doc_id) not in doc_id_filter:
                        continue
                    chunks.append(
                        DocumentChunk(
                            text=doc,
                            source=meta.get("source", "unknown"),
                            doc_id=doc_id or None,
                            section=meta.get("section"),
                            condition_tag=meta.get("condition_tag", "general"),
                            score=float(1.0 - distance) if distance is not None else None,
                        )
                    )
                if chunks:
                    return chunks[:top_k]

        chunks = json.loads(self.index_path.read_text()).get("chunks", [])
        scored = self._score_chunks(query, condition_tag, chunks, doc_id_filter)
        if not scored and condition_tag != "general" and doc_id_filter is None:
            scored = self._score_chunks(query, "general", chunks)
        return [chunk for _, chunk in scored[:top_k]]

    def _score_chunks(
        self,
        query: str,
        condition_tag: str,
        chunks: Iterable[dict],
        doc_id_filter: set[str] | None = None,
    ) -> list[tuple[float, DocumentChunk]]:
        query_tokens = set(tokenize(query))
        scored: list[tuple[float, DocumentChunk]] = []
        for chunk in chunks:
            doc_id = chunk.get("doc_id") or source_doc_id(chunk.get("source"))
            if doc_id_filter is not None and normalize_doc_id(doc_id) not in doc_id_filter:
                continue
            tag = chunk.get("condition_tag", "general")
            if condition_tag != "general" and tag not in {condition_tag, "general"}:
                continue
            text = chunk.get("text", "")
            text_tokens = set(tokenize(text))
            if not text_tokens:
                continue
            overlap = len(query_tokens & text_tokens)
            score = overlap / max(len(query_tokens), 1)
            if condition_tag != "general" and tag == condition_tag:
                score += 0.25
            source_name = chunk.get("source", "").lower()
            query_lower = query.lower()
            text_lower = text.lower()
            if "grade" in query_lower and "wagner" in source_name:
                score += 0.35
            if "grade" in query_lower and "wagner" in text_lower and "grade 0" in text_lower:
                score += 0.80
            for grade in re.findall(r"grade\s+\d+", query_lower):
                if grade in text_lower:
                    score += 0.60
            if "stage" in query_lower and any(key in source_name for key in ["epuap", "npiap", "pressure"]):
                score += 0.35
            for stage in re.findall(r"stage\s+\d+", query_lower):
                stage_number = stage.split()[-1]
                roman = {"1": "i", "2": "ii", "3": "iii", "4": "iv", "5": "v"}.get(stage_number)
                if stage in text_lower or (roman and f"stage {roman}" in text_lower):
                    score += 0.60
            if "classification" in source_name:
                score += 0.10
            if "project_writeup" in source_name:
                score -= 0.10
            if score <= 0:
                continue
            scored.append(
                (
                    score,
                    DocumentChunk(
                        text=text,
                        source=chunk.get("source", "unknown"),
                        doc_id=doc_id or None,
                        section=chunk.get("section"),
                        condition_tag=tag,
                        score=score,
                    ),
                )
            )
        return sorted(scored, key=lambda item: item[0], reverse=True)
