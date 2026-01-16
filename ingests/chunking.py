from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_docs(docs: list[Document],
               chunk_size: int = 800, chunk_overlap: int = 1200 ) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=['\n\n', '\n', '. ', ' ', '']
    )

    chunks: list[Document] = []
    for doc in docs:
        split = splitter.split_documents([doc])
        for j, c in enumerate(split):
            c.metadata = {**doc.metadata, 'chunk_id': j}
            chunks.append(c)
    return chunks

