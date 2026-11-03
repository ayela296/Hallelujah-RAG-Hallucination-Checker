
import pytest
from src.rag.chunking import chunk_document, chunk_documents


# Test 1: Multiple documents
def test_chunk_documents():
    documents = [
        {"name": "file1.txt", "content": "Hello world " * 150},
        {"name": "file2.txt", "content": "Python programming " * 150}
    ]

    chunks = chunk_documents(documents)

    assert len(chunks) == 4
    assert chunks[0]["source"] == "file1.txt"
    assert chunks[1]["source"] == "file1.txt"
    assert chunks[2]["source"] == "file2.txt"
    assert chunks[3]["source"] == "file2.txt"


# Test 2: Single document chunking
def test_chunk_document():
    text = " ".join(["hello"] * 600)

    chunks = chunk_document(text, "test.txt")

    assert len(chunks) == 3
    assert len(chunks[0]["text"].split()) == 250


# Test 3: Chunk overlap
def test_chunk_overlap():
    text = " ".join(["hello"] * 300)

    chunks = chunk_document(text, "test.txt")

    assert len(chunks[0]["text"].split()) == 250
    assert len(chunks[1]["text"].split()) == 90


# Test 4: Source name preservation
def test_source_name():
    chunks = chunk_document("Hello world", "report.pdf")

    assert chunks[0]["source"] == "report.pdf"


# Test 5: Chunk index
def test_chunk_index():
    text = " ".join(["hello"] * 600)

    chunks = chunk_document(text, "test.txt")

    assert chunks[0]["chunk_index"] == 0
    assert chunks[1]["chunk_index"] == 1
    assert chunks[2]["chunk_index"] == 2


# Test 6: Empty document
def test_empty_document():
    chunks = chunk_document("", "empty.txt")

    assert len(chunks) == 0


# Test 7: Document smaller than chunk size
def test_small_document():
    chunks = chunk_document("Hello world Python", "test.txt")

    assert len(chunks) == 1
    assert chunks[0]["text"] == "Hello world Python"


# Test 8: Custom chunk size
def test_custom_chunk_size():
    text = " ".join(["hello"] * 20)

    chunks = chunk_document(text, "test.txt", chunk_size=10, overlap=2)

    assert len(chunks) == 3
    assert len(chunks[0]["text"].split()) == 10


# Test 9: Invalid chunk size
def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_document("Hello", "test.txt", chunk_size=0)


# Test 10: Invalid overlap
def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_document("Hello", "test.txt", chunk_size=100, overlap=100)


# Test 11: Invalid text type
def test_invalid_text_type():
    with pytest.raises(TypeError):
        chunk_document(123, "test.txt")


# Test 12: Missing document name
def test_missing_document_name():
    documents = [{"content": "Hello world"}]

    with pytest.raises(ValueError):
        chunk_documents(documents)


# Test 13: Missing document content
def test_missing_document_content():
    documents = [{"name": "test.txt"}]

    with pytest.raises(ValueError):
        chunk_documents(documents)


# Test 14: Invalid document type
def test_invalid_document_type():
    documents = ["Hello world"]

    with pytest.raises(TypeError):
        chunk_documents(documents)


# Test 15: Invalid documents input
def test_invalid_documents_input():
    with pytest.raises(TypeError):
        chunk_documents("Hello world")
