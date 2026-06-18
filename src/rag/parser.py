import os
from src.rag.chunk import Document

try:
    from docling.document_converter import DocumentConverter
    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False

def parse_file(file_path: str) -> Document:
    """
    Parses an incoming file (TXT, MD, PDF) into a typed and validated Document schema.
    For PDFs, utilizes IBM's Docling layout converter offline to produce high-fidelity Markdown.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found at: {file_path}")

    abs_path = os.path.abspath(file_path)
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".txt":
        with open(abs_path, "r", encoding="utf-8") as f:
            text = f.read()
        return Document.create(text=text, source_path=abs_path, file_type="txt")

    elif ext == ".md":
        with open(abs_path, "r", encoding="utf-8") as f:
            text = f.read()
        return Document.create(text=text, source_path=abs_path, file_type="md")

    elif ext == ".pdf":
        if not HAS_DOCLING:
            raise ImportError(
                "The 'docling' library is not installed. Please install it using "
                "'pip install docling' to process PDF files."
            )
        
        converter = DocumentConverter()
        result = converter.convert(abs_path)
        markdown_text = result.document.export_to_markdown()
        return Document.create(text=markdown_text, source_path=abs_path, file_type="pdf")

    else:
        raise ValueError(f"Unsupported file format: {ext}. Only .txt, .md, and .pdf are supported.")
