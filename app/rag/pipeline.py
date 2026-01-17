from __future__ import annotations
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from app.config import load_settings, Settings
from app.providers import get_embeddings, get_chatbot_llm
from app.rag.prompts import SYSTEM_PROMPT
import re
import json

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
        docs = self.db.similarity_search(question, k=top_k)

        # Deduplicate by (source, page) to reduce repetition
        seen = set()
        deduped = []
        for d in docs:
            key = (d.metadata.get("source"), d.metadata.get("page"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(d)

        # Build labeled context C1..Cn
        context_lines = []
        sources = []
        max_chars = 12000  # keep small for slow local models
        current = 0

        for i, d in enumerate(deduped, start=1):
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", "?")
            chunk_id = d.metadata.get("chunk_id", None)

            text = (d.page_content or "").strip()
            block = f"C{i} ({src} p.{page}):\n{text}\n"
            if current + len(block) > max_chars:
                break
            current += len(block)

            context_lines.append(block)
            sources.append({"cid": f"C{i}", "source": src, "page": page, "chunk_id": chunk_id})

        context = "\n".join(context_lines)

        system = """You are a QA assistant.
        Use ONLY the provided CONTEXT.
        If the answer is not in the CONTEXT, output exactly this JSON and nothing else:
        {"definition":"I don't know based on the documents.","why_it_matters":"","components":[],"cited_chunks":[]}

        Rules:
        - Return VALID JSON ONLY. No markdown. No extra text.
        - definition: 1-2 sentences.
        - why_it_matters: exactly 1 sentence (or empty if unknown).
        - components: MUST be 2-5 short strings (<=12 words each) when answer is known; otherwise [].
        - cited_chunks: MUST contain 1-3 chunk ids when answer is known; otherwise [].
        - Choose the MINIMUM chunk ids necessary to support the answer.
        - Only cite chunk ids present in the CONTEXT (C1..Cn)."""

        user = f"""CONTEXT:
        {context}

        QUESTION:
        {question}

        Return the JSON object now."""

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        raw = self.llm.invoke(messages).content

        # --- Robust JSON extraction & validation ---
        # Some models accidentally include leading/trailing text; extract first {...} block.
        json_text = None
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            json_text = m.group(0)
        else:
            json_text = raw.strip()

        fallback = {
            "definition": "I don't know based on the documents.",
            "why_it_matters": "",
            "components": [],
            "cited_chunks": [],
        }

        try:
            data = json.loads(json_text)
        except Exception:
            data = fallback

        # Validate schema
        definition = data.get("definition", fallback["definition"])
        why = data.get("why_it_matters", fallback["why_it_matters"])
        components = data.get("components", fallback["components"])
        cited = data.get("cited_chunks", fallback["cited_chunks"])

        if not isinstance(definition, str):
            definition = fallback["definition"]
        if not isinstance(why, str):
            why = fallback["why_it_matters"]
        if not isinstance(components, list) or not all(isinstance(x, str) for x in components):
            components = []
        if not isinstance(cited, list) or not all(isinstance(x, str) for x in cited):
            cited = []

        # Enforce chunk-id validity: only allow C1..C{len(sources)}
        valid_cids = {s["cid"] for s in sources}
        cited = [c for c in cited if c in valid_cids]

        # If model "answered" but provided no citations, that’s suspicious — clamp to "I don't know"
        # (You can relax this later, but it keeps demos safe.)
        # If answer is known, components must not be empty
        known = "I don't know based on the documents." not in definition
        if known and len(cited) > 3:
            definition = fallback["definition"]
            why = ""
            components = []
            cited = []
        # Enforce citation selectivity: keep at most 3
        cited = cited[:3]

        if definition.strip() and "I don't know based on the documents." not in definition and len(cited) == 0:
            definition = fallback["definition"]
            why = ""
            components = []
            cited = []

        return {
            "definition": definition.strip(),
            "why_it_matters": why.strip(),
            "components": components[:5],
            "cited_chunks": cited,
            "sources": sources,
        }
