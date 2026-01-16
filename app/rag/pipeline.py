from __future__ import annotations
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from app.config import load_settings, Settings
from app.providers import get_embeddings, get_chatbot_llm
from app.rag.prompts import SYSTEM_PROMPT

class RAGPipeline:
    def __init__(self, settings: Settings | None = None):
        load_dotenv()
        self.settings = settings or load_settings()
        self.embeddings = get_embeddings(self.settings)
        self.db = FAISS.load_local(
            self.settings.index_dir,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.llm = get_chatbot_llm(self.settings)


    def ask(self, question: str, k: int | None = None) -> dict:
        top_k = k or self.settings.rag_top_k
        docs = self.db.similarity_search(question, k =top_k)
        context_blocks: list[str] = []
        citations: list[dict] = []

        for d in docs:
            source = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", "?")
            chunk_id = d.metadat.get("chunk_id", None)
            context_blocks.append(f"[{source}p.{page}] {d.page_content}")
            citations.append({"source": source, "page": page, "chunk_id": chunk_id})

        context = "\n\n".join(context_blocks)
        user_prompt=f"""CONTEXT:
{context}
QUESTION: {question}
Write the answer now following the rules."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": user_prompt},
        ]

        answer = self.llm.invoke(messages).content
        return {"answer": answer, "citations": citations}


