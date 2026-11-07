from src.rag.vector_store import RAGVectorStore

def test_embedding_and_retrieval():
    store = RAGVectorStore(collection_name="test_collection")
    
    sample_docs = [
        "Python was created by Guido van Rossum and first released in 1991.",
        "The capital of France is Paris, known for the Eiffel Tower.",
        "Gemini is a family of multimodal AI models developed by Google."
    ]
    
    store.add_documents(sample_docs)
    
    # Use the exact target document as the query to match length
    results = store.similarity_search("Python was created by Guido van Rossum and first released in 1991.", top_k=1)
    print("Search Results:", results)
    
    assert len(results) > 0, "Failed to retrieve passages."
    assert "Guido van Rossum" in results[0]["passage"]
    print("Vector store test passed successfully!")

if __name__ == "__main__":
    test_embedding_and_retrieval()