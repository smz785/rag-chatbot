from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_pdfs(pdf_dir: str | Path) -> list[Document]:
    pdf_path = Path(pdf_dir)

    if not pdf_path.exists():
        raise RuntimeError(f"PDF directory does not exist: {pdf_path}")

    docs: list[Document] = []
    for file in sorted(pdf_path.glob('*.pdf')):
        loader = PyPDFLoader(str(file))
        pages = loader.load()

        for page_idx, d in enumerate(pages):
            d.metadata = {
                "source": file.name,
                "page": page_idx
            }
            docs.append(d)
    if not docs:
        raise RuntimeError(f"No pdfs found in: {pdf_path}")
    return docs
