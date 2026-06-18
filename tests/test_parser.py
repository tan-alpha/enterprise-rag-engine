import os
import pytest
from unittest.mock import MagicMock, patch
from src.rag.parser import parse_file

def test_parse_txt_file(tmp_path):
    text_content = "Hello, this is a plain text file."
    txt_file = tmp_path / "test.txt"
    txt_file.write_text(text_content, encoding="utf-8")
    
    doc = parse_file(str(txt_file))
    
    assert doc.text == text_content
    assert doc.metadata.file_type == "txt"
    assert doc.metadata.source_path == os.path.abspath(txt_file)
    assert doc.content_hash is not None

def test_parse_md_file(tmp_path):
    md_content = "# Header\n\nSome markdown content."
    md_file = tmp_path / "test.md"
    md_file.write_text(md_content, encoding="utf-8")
    
    doc = parse_file(str(md_file))
    
    assert doc.text == md_content
    assert doc.metadata.file_type == "md"
    assert doc.metadata.source_path == os.path.abspath(md_file)

def test_parse_non_existent_file():
    with pytest.raises(FileNotFoundError):
        parse_file("non_existent_file_path.txt")

def test_parse_unsupported_file(tmp_path):
    dummy_file = tmp_path / "test.xyz"
    dummy_file.write_text("xyz content", encoding="utf-8")
    
    with pytest.raises(ValueError):
        parse_file(str(dummy_file))

@patch("src.rag.parser.DocumentConverter")
def test_parse_pdf_file_mocked(mock_converter_class, tmp_path):
    # Setup mock docling output
    mock_converter = MagicMock()
    mock_converter_class.return_value = mock_converter
    
    mock_result = MagicMock()
    mock_result.document.export_to_markdown.return_value = "# Extracted PDF content in Markdown"
    mock_converter.convert.return_value = mock_result
    
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_text("fake pdf bytes", encoding="utf-8") # just needs to exist
    
    doc = parse_file(str(pdf_file))
    
    assert doc.text == "# Extracted PDF content in Markdown"
    assert doc.metadata.file_type == "pdf"
    assert doc.metadata.source_path == os.path.abspath(pdf_file)
    mock_converter.convert.assert_called_once_with(os.path.abspath(pdf_file))
