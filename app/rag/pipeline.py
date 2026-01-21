from __future__ import annotations
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from app.config import load_settings, Settings
from app.providers import get_embeddings, get_chatbot_llm
from app.rag.prompts import SYSTEM_PROMPT
import re

class RAGPipeline:
    def __init__(self, settings: Settings | None = None):
        load_dotenv()
        self.settings = load_settings()
        self.llm = get_chatbot_llm(self.settings)
        self.embeddings = get_embeddings(self.settings)

        # chunk index
        self.db = FAISS.load_local(
            self.settings.index_dir,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        # doc routing index
        self.doc_db = FAISS.load_local(
            self.settings.doc_index_dir,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        # DEBUG: What files are actually in the chunk index?
        print("\n=== CHUNK INDEX CONTENTS ===")
        chunk_sources = set()
        for doc_id, doc in self.db.docstore._dict.items():
            if hasattr(doc, 'metadata'):
                src_id = (doc.metadata.get("source_id", "") or "").strip().lower()
                if src_id:
                    chunk_sources.add(src_id)

        print(f"Total unique source_ids in chunk index: {len(chunk_sources)}")
        for src in sorted(chunk_sources):
            print(f"  - {src}")

        # DEBUG: What files are in the doc routing index?
        print("\n=== DOC ROUTING INDEX CONTENTS ===")
        doc_sources = set()
        for doc_id, doc in self.doc_db.docstore._dict.items():
            if hasattr(doc, 'metadata'):
                src_id = (doc.metadata.get("source_id", "") or "").strip().lower()
                if src_id:
                    doc_sources.add(src_id)

        print(f"Total unique source_ids in doc routing index: {len(doc_sources)}")
        for src in sorted(doc_sources):
            print(f"  - {src}")

        print("\n=== MISMATCH CHECK ===")
        only_in_routing = doc_sources - chunk_sources
        only_in_chunks = chunk_sources - doc_sources

        if only_in_routing:
            print(f"Files ONLY in routing index ({len(only_in_routing)}):")
            for src in only_in_routing:
                print(f"  - {src}")

        if only_in_chunks:
            print(f"Files ONLY in chunk index ({len(only_in_chunks)}):")
            for src in only_in_chunks:
                print(f"  - {src}")

    def ask(self, question: str, k: int | None = None) -> dict:
        def _extract_page(d) -> int | str:
            meta_page = d.metadata.get("page") if (d.metadata or {}) else None
            if isinstance(meta_page, int):
                return meta_page
            if isinstance(meta_page, str) and meta_page.isdigit():
                return int(meta_page)

            text = (d.page_content or "")[:400]
            patterns = [r"\[page\s*(\d+)\]", r"\bpage[:\s#-]+(\d+)\b", r"\bp\.\s*(\d+)\b"]
            for p in patterns:
                m = re.search(p, text, flags=re.IGNORECASE)
                if m:
                    try:
                        return int(m.group(1))
                    except Exception:
                        pass
            return "?"

        top_k = k or self.settings.rag_top_k
        sys_prompt = SYSTEM_PROMPT

        # ---- Stage A: route to top-N documents (use source_id for matching) ----
        routed = self.doc_db.similarity_search(question, k=self.settings.doc_route_top_n)

        routed_docs_display: list[str] = []  # filenames for output/debug
        routed_ids: list[str] = []  # canonical ids for matching
        seen_ids = set()

        for d in routed:
            sid = (d.metadata.get("source_id", "") or "").strip().lower()
            src = (d.metadata.get("source", "") or "").strip()
            if not sid:
                # If doc index doesn't have source_id, routing is mis-built
                continue

            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            routed_ids.append(sid)
            routed_docs_display.append(src or sid)

        routed_id_set = set(routed_ids)

        # ---- Stage B: retrieve chunks broadly then filter to routed docs ----
        fetch_k = 500 #max(self.settings.chunk_fetch_k, top_k * 20)
        candidates = self.db.similarity_search(question, k=fetch_k)

        # DEBUG: confirm source_id exists in chunk candidates and overlaps routing ids
        cand_ids = sorted({(c.metadata.get("source_id", "") or "").strip().lower() for c in candidates})
        cand_ids = [x for x in cand_ids if x]
        print("ROUTED_IDS:", routed_ids)
        print("CAND_SOURCE_ID_SAMPLE:", cand_ids[:10])
        print("CAND_HAS_ROUTED_ID?:", any(cid in routed_id_set for cid in cand_ids))

        filtered = []
        for d in candidates:
            cid = (d.metadata.get("source_id", "") or "").strip().lower()
            if cid and cid in routed_id_set:
                filtered.append(d)
        print(f"DEBUG: Routed to {len(routed_id_set)} docs: {routed_ids}")
        print(f"DEBUG: Found {len(filtered)} chunks from routed docs out of {len(candidates)} candidates")
        # if len(filtered) == 0:
        #     docs = candidates[:top_k]
        #     routing_used = False
        # else:
        #     docs = filtered[:top_k]
        #     routing_used = True
        if len(filtered) == 0:
            print("WARNING: No chunks found from routed docs. Doing direct search...")
            # Get ALL chunks from routed docs (no similarity filter)
            all_chunks = []
            for doc_meta in self.db.docstore._dict.values():
                if hasattr(doc_meta, 'metadata'):
                    cid = (doc_meta.metadata.get("source_id", "") or "").strip().lower()
                    if cid in routed_id_set:
                        all_chunks.append(doc_meta)

            # If we found chunks this way, use them
            if all_chunks:
                filtered = all_chunks[:top_k * 3]
                routing_used = True
            else:
                # Last resort: use top candidates
                docs = candidates[:top_k]
                routing_used = False
        else:
            docs = filtered[:top_k]
            routing_used = True

        # Deduplicate by (source_id, page)
        seen = set()
        deduped = []
        for d in docs:
            page_val = _extract_page(d)
            sid = (d.metadata.get("source_id", "") or "").strip().lower()
            key = (sid, page_val)
            if key in seen:
                continue
            seen.add(key)
            d.metadata = {**(d.metadata or {}), "page": page_val}
            deduped.append(d)

        # Build bounded context + sources/snippets
        max_chars = 8000
        context_blocks = []
        sources = []
        snippets = []
        used_chars = 0

        for d in deduped:
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", "?")
            chunk_id = d.metadata.get("chunk_id", None)
            text = (d.page_content or "").strip()

            sources.append({"source": src, "page": page, "chunk_id": chunk_id})
            snippet = re.sub(r"\s+", " ", text)[:240]
            snippets.append({"source": src, "page": page, "chunk_id": chunk_id, "text": snippet})

            block = f"Source: {src} | Page: {page}\n{text}\n"
            if used_chars + len(block) > max_chars:
                break
            used_chars += len(block)
            context_blocks.append(block)

        context = "\n---\n".join(context_blocks)

        # IMPORTANT: no indentation in the prompt
        user = f"""CONTEXT:
    {context}

    QUESTION:
    {question}

    Answer now."""

        answer = self.llm.invoke(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]
        ).content.strip()

        return {
            "answer": answer,
            "routed_docs": routed_docs_display,
            "routing_used": routing_used,
            "sources": sources,
            "snippets": snippets,
            "retrieved_count": len(candidates),
            "used_count": len(sources),
        }
