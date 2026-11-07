
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
    assert [chunk["source"] for chunk in chunks] == [
        "file1.txt", "file1.txt", "file2.txt", "file2.txt"
    ]


# Test 2: Single document chunking
def test_chunk_document():
    text = " ".join(["hello"] * 600)

    chunks = chunk_document(text, "test.txt")

    assert len(chunks) == 3
    assert len(chunks[0]["text"].split()) == 250


# Test 3: Chunk overlap
def test_chunk_overlap():
    words = [f"word{i}" for i in range(300)]
    text = " ".join(words)

    chunks = chunk_document(text, "test.txt")

    first_words = chunks[0]["text"].split()
    second_words = chunks[1]["text"].split()

    assert first_words[-40:] == second_words[:40]


# Test 4: Source name preservation
def test_source_name():
    chunks = chunk_document("Hello world", "report.pdf")

    assert chunks[0]["source"] == "report.pdf"


def test_document_metadata_and_chunk_id_are_preserved():
    document = {
        "name": "report.pdf",
        "content": "Hello world",
        "metadata": {"department": "research"},
    }

    chunks = chunk_documents([document])

    assert chunks[0]["metadata"] == {"department": "research"}
    assert chunks[0]["chunk_id"]
    assert chunks[0]["chunk_id"] == chunk_documents([document])[0]["chunk_id"]


# Test 5: Chunk index
def test_chunk_index():
    text = " ".join(["hello"] * 600)

    chunks = chunk_document(text, "test.txt")

    assert [chunk["chunk_index"] for chunk in chunks] == [0, 1, 2]


# Test 6: Empty document
def test_empty_document():
    chunks = chunk_document("", "empty.txt")

    assert chunks == []


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
    with pytest.raises(ValueError):
        chunk_documents([{"content": "Hello world"}])


# Test 13: Missing document content
def test_missing_document_content():
    with pytest.raises(ValueError):
        chunk_documents([{"name": "test.txt"}])


# Test 14: Invalid document type
def test_invalid_document_type():
    with pytest.raises(TypeError):
        chunk_documents(["Hello world"])


# Test 15: Invalid documents input
def test_invalid_documents_input():
    with pytest.raises(TypeError):
        chunk_documents("Hello world")
