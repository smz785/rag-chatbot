from langchain_core.documents import Document
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from app.config import load_settings
from app.providers import get_embeddings
from ingests.loaders import load_pdfs
from ingests.chunking import chunk_docs
from ingests.loaders import _source_id


def build_index() -> None:
    load_dotenv()
    settings = load_settings()
    embeddings = get_embeddings(settings)

    pages = load_pdfs(settings.pdf_dir)

    # 1) Chunk index (for retrieval)
    chunks = chunk_docs(
        pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunk_db = FAISS.from_documents(chunks, embedding=embeddings)
    chunk_db.save_local(settings.index_dir)
    print(f"Saved CHUNK FAISS index to {settings.index_dir}")
    print(f"Chunks indexed: {len(chunks)}")

    for chunk in chunks:
        src = chunk.metadata.get("source", "")
        chunk.metadata["source_id"] = _source_id(src)

    chunk_db = FAISS.from_documents(chunks, embedding=embeddings)
    chunk_db.save_local(settings.index_dir)
    print(f"Saved CHUNK FAISS index to {settings.index_dir}")
    print(f"Chunks indexed: {len(chunks)}")

    # 2) Doc routing index (use small routing chunks to avoid Ollama context overflow)
    doc_docs = build_doc_documents(
        pages,
        max_chars=settings.doc_text_max_chars,
        max_pages=3,
        routing_chunk_chars=900,
        routing_overlap_chars=100,
    )
    doc_db = FAISS.from_documents(doc_docs, embedding=embeddings)
    doc_db.save_local(settings.doc_index_dir)
    print(f"Saved DOC FAISS index to {settings.doc_index_dir}")
    print(f"Docs indexed: {len(doc_docs)}")


def build_doc_documents(
    pages: list[Document],
    *,
    max_chars: int = 12000,
    max_pages: int = 3,
    routing_chunk_chars: int = 900,
    routing_overlap_chars: int = 100,
) -> list[Document]:
    """
    Build routing documents per PDF, but split into small chunks so embeddings never
    exceed the Ollama model context window.
    """

    def normalize(text: str) -> str:
        text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def strip_references(text: str) -> str:
        m = re.search(r"^\s*references\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
        return text[: m.start()].strip() if m else text

    def is_junk_line(line: str) -> bool:
        l = re.sub(r"\s+", " ", line).strip().lower()
        if not l:
            return True
        junk = [
            "annual conference on",
            "proceedings of",
            "acm",
            "ieee",
            "copyright",
            "all rights reserved",
            "permission to",
            "provided that you attribute",
            "personal and corporate web sites",
            "this work is licensed",
            "doi:",
            "arxiv:",
        ]
        return any(j in l for j in junk)

    def cleanup(text: str) -> str:
        lines = [ln.strip() for ln in text.splitlines()]
        kept: list[str] = []
        for ln in lines:
            if is_junk_line(ln):
                continue
            if re.match(r"^\s*\[page\s+\d+\]\s*$", ln, flags=re.IGNORECASE):
                continue
            kept.append(ln)
        cleaned = "\n".join(kept)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    def split_with_overlap(text: str, size: int, overlap: int) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        if size <= 0:
            return [text]
        overlap = max(0, min(overlap, size - 1)) if size > 1 else 0
        step = max(1, size - overlap)
        parts: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + size].strip()
            if chunk:
                parts.append(chunk)
            if start + size >= len(text):
                break
        return parts

    by_source: dict[str, list[Document]] = {}
    for d in pages:
        src = d.metadata.get("source", "unknown")
        by_source.setdefault(src, []).append(d)

    doc_docs: list[Document] = []
    for src, ds in by_source.items():
        ds = sorted(ds, key=lambda x: x.metadata.get("page", 0))
        ds = ds[:max_pages]

        merged = "\n\n".join(normalize(x.page_content) for x in ds).strip()
        if not merged:
            continue

        merged = strip_references(merged)
        merged = cleanup(merged)

        merged = merged[:max_chars].strip()
        if not merged:
            continue

        routing_parts = split_with_overlap(
            merged,
            size=routing_chunk_chars,
            overlap=routing_overlap_chars,
        )

        for i, part in enumerate(routing_parts):
            doc_docs.append(
                Document(
                    page_content=part,
                    metadata={
                        "source": src,  # <-- ADD THIS
                        "source_id": _source_id(src),  # canonical routing key
                        "route_pages_used": len(ds),
                        "route_part": i,
                        "route_parts_total": len(routing_parts),
                    },
                )
            )

    return doc_docs


if __name__ == "__main__":
    build_index()
