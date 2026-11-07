from src.rag.vector_store import RAGVectorStore

def test_top3_retrieval_with_scores_and_sources():
    store = RAGVectorStore(collection_name="test_retrieval_collection")
    
    docs = [
        "Python was created by Guido van Rossum in 1991.",
        "RAG architecture combines vector retrieval with language models.",
        "ChromaDB stores vectorized document embeddings efficiently."
    ]
    metadatas = [
        {"source": "python_history.txt"},
        {"source": "rag_spec.pdf"},
        {"source": "chroma_docs.md"}
    ]
    
    store.add_documents(documents=docs, metadatas=metadatas)
    
    # Test top-3 retrieval
    claim = "Python was created by Guido van Rossum in 1991."
    results = store.retrieve_top_k(claim, top_k=3)
    
    print("Retrieved Results:", results)
    
    assert len(results) > 0, "Failed to retrieve any passages."
    assert "passage" in results[0]
    assert "relevance_score" in results[0]
    assert "source" in results[0]
    assert results[0]["source"] == "python_history.txt"
    
    print("SCRUM-71 retrieval test passed successfully!")

if __name__ == "__main__":
    test_top3_retrieval_with_scores_and_sources()
