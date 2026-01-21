from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import re

def _source_id(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def load_pdfs(pdf_dir: str | Path) -> list[Document]:
    pdf_path = Path(pdf_dir)

    if not pdf_path.exists():
        raise RuntimeError(f"PDF directory does not exist: {pdf_path}")

    docs: list[Document] = []
    for file in sorted(pdf_path.glob('*.pdf')):
        loader = PyPDFLoader(str(file))
        pages = loader.load()

        sid = _source_id(file.name)

        for page_idx, d in enumerate(pages):
            d.metadata = {
                "source": file.name,
                "source_id": sid,      # <-- ADD THIS
                "page": page_idx
            }
            docs.append(d)

    if not docs:
        raise RuntimeError(f"No pdfs found in: {pdf_path}")
    return docs
