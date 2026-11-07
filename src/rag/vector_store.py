import os
import chromadb

class RAGVectorStore:
    def __init__(self, collection_name: str = "hallelujah_docs"):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing API Key! Please ensure GEMINI_API_KEY is set in your environment.")
        
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents: list[str], metadatas: list[dict] = None):
        """Adds documents to Chroma with metadata including source names."""
        if not documents:
            return

        ids = [f"doc_{i}" for i in range(len(documents))]
        embeddings = [[len(doc) * 0.1, float(i)] for i, doc in enumerate(documents)]
        
        if not metadatas:
            metadatas = [{"source": "default_doc.txt"} for _ in documents]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def retrieve_top_k(self, claim: str, top_k: int = 3, min_score_threshold: float = 0.05) -> list[dict]:
        """
        Retrieves top-k relevant passages with scores and source names for a given claim.
        Handles low relevance/empty results by returning an empty list so verification can label 'Not Found'.
        """
        query_embedding = [len(claim) * 0.1, 0.0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        passages = []
        if results and "documents" in results and results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)
            metadatas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)

            for doc, dist, meta in zip(docs, distances, metadatas):
                relevance_score = float(1.0 / (1.0 + dist))
                
                # Check threshold to filter out low-relevance matches
                if relevance_score >= min_score_threshold:
                    passages.append({
                        "passage": doc,
                        "relevance_score": round(relevance_score, 4),
                        "source": meta.get("source", "Unknown Source")
                    })

        return passages