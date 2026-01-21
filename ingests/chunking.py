import re
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------- Heuristics (cheap but effective) ----------

_HEADING_RE = re.compile(r"^\s*(\d+(\.\d+)*)\s+([A-Z][^\n]{3,80})\s*$")  # e.g., "4.1 Semantic data graphs"
_CAPTION_RE = re.compile(r"^\s*(Figure|Fig\.|Table)\s+\d+[:.]\s+.*$", re.IGNORECASE)
_CODE_HINT_RE = re.compile(r"^\s*(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b", re.IGNORECASE)
_EQUATION_HINT_RE = re.compile(r"[\=\+\-\*/]\s*[\w\(\)\[\]]+|\\(frac|sum|int|alpha|beta|theta|lambda)", re.IGNORECASE)


def _normalize(text: str) -> str:
    # normalize whitespace but keep newlines for block detection
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_junk_page(text: str) -> bool:
    t = re.sub(r"\s+", " ", text.lower())
    junk_phrases = [
        "annual conference on innovative data systems research",
        "cidr",
        "provided that you attribute",
        "personal and corporate web sites",
        "copyright",
        "all rights reserved",
    ]
    if any(p in t for p in junk_phrases):
        return True
    # very short pages are usually noise (cover, footer-only)
    return len(t) < 200


def _strip_references(text: str) -> str:
    # crude but effective: drop everything after "References" heading
    m = re.search(r"^\s*references\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    return text[: m.start()].strip() if m else text


def _classify_block(block: str) -> str:
    b = block.strip()
    if not b:
        return "empty"
    if _CAPTION_RE.match(b):
        return "caption"
    # table-ish: many lines with multiple columns or pipe separators
    lines = b.splitlines()
    tabular_lines = sum(1 for ln in lines if ("|" in ln) or (len(re.findall(r"\s{2,}", ln)) >= 2))
    if tabular_lines >= max(3, int(0.5 * len(lines))):
        return "table"
    # code-ish: lots of braces/semicolons or SQL keywords near line starts
    code_lines = sum(1 for ln in lines if _CODE_HINT_RE.match(ln) or ln.strip().endswith(";"))
    if code_lines >= 2:
        return "code"
    # equation-ish
    if _EQUATION_HINT_RE.search(b) and len(b) < 800:
        return "equation"
    return "text"


def _split_into_blocks(text: str) -> List[str]:
    """
    Split into coarse blocks by blank lines.
    We'll later keep certain block types intact (tables/code/equations/captions).
    """
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _group_by_headings(blocks: List[str]) -> List[Tuple[str, List[str]]]:
    """
    Group blocks into sections by detecting numbered headings.
    Returns list of (section_title, blocks_in_section).
    """
    sections: List[Tuple[str, List[str]]] = []
    current_title = "Unknown"
    current_blocks: List[str] = []

    for b in blocks:
        # heading detection: single-line block that matches heading regex
        if "\n" not in b and _HEADING_RE.match(b):
            # flush previous
            if current_blocks:
                sections.append((current_title, current_blocks))
            current_title = b.strip()
            current_blocks = []
        else:
            current_blocks.append(b)

    if current_blocks:
        sections.append((current_title, current_blocks))
    return sections


def chunk_docs(docs: list[Document], chunk_size: int = 800, chunk_overlap: int = 120) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    global_id = 0

    for doc in docs:
        text = _normalize(doc.page_content)
        if not text:
            continue
        # if _is_junk_page(text):
        #     continue
        # text = _strip_references(text)

        # split into coarse blocks by blank lines
        blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

        for b in blocks:
            block_type = _classify_block(b)
            base_meta = {
                **(doc.metadata or {}),
                "chunk_type": block_type,
            }

            if block_type in {"table", "code", "equation", "caption"}:
                chunks.append(Document(page_content=b, metadata={**base_meta, "chunk_id": global_id}))
                global_id += 1
                continue

            # text blocks: further split by size
            split_docs = splitter.split_documents([Document(page_content=b, metadata=base_meta)])
            for sd in split_docs:
                sd.metadata = {**sd.metadata, "chunk_id": global_id}
                chunks.append(sd)
                global_id += 1

    return chunks