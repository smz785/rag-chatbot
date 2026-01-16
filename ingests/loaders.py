from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_pdfs(pdf_dir: str) -> list[Document]:
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        raise RuntimeError(f"The file {pdf_path} does not exist")
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
        raise RuntimeError(f"No pdfs found in: {pdf_dir}")
    return docs

