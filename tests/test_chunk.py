import pytest
from pydantic import ValidationError
from src.rag.chunk import Document, Chunk, RecursiveLinguisticSplitter, split_document

def test_document_contract_creation():
    text = "Enterprise RAG Contract Testing."
    source = "/path/to/doc.txt"
    file_type = "txt"
    
    doc = Document.create(text=text, source_path=source, file_type=file_type)
    
    # Assert deterministic SHA-256 hash
    import hashlib
    expected_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert doc.id == expected_hash
    assert doc.content_hash == expected_hash
    assert doc.metadata.source_path == source
    assert doc.metadata.file_type == file_type
    assert doc.metadata.created_at is not None

def test_document_contract_validation_error():
    # Attempting to create Document metadata without source_path should fail at instantiation
    with pytest.raises(ValidationError):
        # source_path is a required field of DocumentMetadata
        Document(
            id="test",
            text="text",
            content_hash="test",
            metadata={}  # Invalid dictionary, lacks fields
        )

def test_chunk_deterministic_and_sequential():
    doc = Document.create(text="Test chunk sequential ids.", source_path="dummy.txt", file_type="txt")
    chunks = split_document(doc, chunk_size=10, chunk_overlap=2)
    
    assert len(chunks) > 0
    for idx, chunk in enumerate(chunks):
        assert chunk.parent_doc_id == doc.id
        assert chunk.id == f"{doc.id}#chunk_{idx}"
        assert chunk.index == idx
        assert chunk.metadata.source_path == doc.metadata.source_path

def test_recursive_linguistic_split_hierarchy():
    text = "# Section 1\n\nParagraph one.\n\nParagraph two.\n# Section 2\n\nParagraph three."
    doc = Document.create(text=text, source_path="doc.md", file_type="md")
    
    # Set chunk_size small enough to split at headers and paragraphs
    chunks = split_document(doc, chunk_size=40, chunk_overlap=5)
    
    assert len(chunks) >= 3
    # Check that splitting honored boundaries and didn't mangle text
    assert "# Section 1" in chunks[0].text
    assert "Section 2" in chunks[-1].text

def test_strict_fallback_splitting():
    # Text with no spaces or headers (long sequence of 'a')
    text = "a" * 100
    splitter = RecursiveLinguisticSplitter(chunk_size=30, chunk_overlap=10)
    chunks = splitter.split_text(text)
    
    # Should fall back to character splitting
    for chunk in chunks:
        assert len(chunk) <= 30
