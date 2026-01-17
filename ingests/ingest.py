from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from app.config import load_settings
from app.providers import get_embeddings
from ingests.loaders import load_pdfs
from ingests.chunking import chunk_docs

def build_index() -> None:
    load_dotenv()
    settings = load_settings()
    docs = load_pdfs(settings.pdf_dir)
    chunks = chunk_docs(docs, chunk_size=settings.chunk_size,
                        chunk_overlap=settings.chunk_overlap)

    embeddings = get_embeddings(settings)
    db = FAISS.from_documents(chunks, embedding=embeddings)

    db.save_local(settings.index_dir)
    print(f"Save FAISS index to {settings.index_dir}")
    print(f"chunks indexed: {len(chunks)}")

if __name__ == "__main__":
    build_index()
