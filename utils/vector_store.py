# utils/vector_store.py

import chromadb
from sentence_transformers import SentenceTransformer
import hashlib


class ChromaVectorStore:
    def __init__(self):
        """
        Initialize an in-memory Chroma client (session-scoped).
        Each document is stored in a single shared collection,
        tagged with a doc_id for filtering.
        """
        self.client = chromadb.Client()  # ephemeral / in-memory
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.collection = self.client.get_or_create_collection(
            name="multi_document_store",
            metadata={"hnsw:space": "cosine"}
        )

    def _make_doc_id(self, doc_name: str) -> str:
        """Stable short ID derived from filename."""
        return hashlib.md5(doc_name.encode()).hexdigest()[:8]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embedding_model.encode(texts, show_progress_bar=False).tolist()

    def index_document(self, chunks: list[dict], doc_name: str) -> str:
        """
        Add chunks from a single document into the shared collection.
        Each chunk is tagged with doc_id and doc_name for filtering & citation.
        Returns the doc_id.
        """
        doc_id = self._make_doc_id(doc_name)

        self.delete_document(doc_id)

        documents = [c["content"] for c in chunks]
        metadatas = [
            {
                "doc_id":   doc_id,
                "doc_name": doc_name,
                "page":     c["page"],
            }
            for c in chunks
        ]
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        embeddings = self.embed_texts(documents)

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        return doc_id

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: list[str] | None = None,
    ) -> list[dict]:
        """
        Retrieve the top-k most relevant chunks.
        Optionally filter to specific doc_ids.
        Returns a list of dicts: {content, doc_name, page, distance}.
        """
        query_embedding = self.embed_texts([query])

        where_filter = (
            {"doc_id": {"$in": doc_ids}} if doc_ids and len(doc_ids) > 0 else None
        )

        kwargs = dict(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        if where_filter:
            kwargs["where"] = where_filter

        results = self.collection.query(**kwargs)

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "content":  doc,
                "doc_name": meta.get("doc_name", "Unknown"),
                "doc_id":   meta.get("doc_id", ""),
                "page":     meta.get("page", "?"),
                "distance": dist,
            })

        hits.sort(key=lambda x: x["distance"])
        return hits

    def delete_document(self, doc_id: str) -> None:
        """Remove all chunks belonging to a document."""
        try:
            existing = self.collection.get(where={"doc_id": {"$eq": doc_id}})
            if existing["ids"]:
                self.collection.delete(ids=existing["ids"])
        except Exception:
            pass

    def list_documents(self) -> list[dict]:
        """Return unique documents currently indexed: [{doc_id, doc_name}]."""
        try:
            all_meta = self.collection.get(include=["metadatas"])["metadatas"]
            seen = {}
            for m in all_meta:
                did = m.get("doc_id", "")
                if did not in seen:
                    seen[did] = m.get("doc_name", "Unknown")
            return [{"doc_id": k, "doc_name": v} for k, v in seen.items()]
        except Exception:
            return []

    def total_chunks(self) -> int:
        return self.collection.count()