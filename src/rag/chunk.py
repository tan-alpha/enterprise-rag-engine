import hashlib
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class DocumentMetadata(BaseModel):
    """
    Typed, strict structure enforcing metadata parameters for incoming text blocks.
    """
    source_path: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    file_type: str
    custom: Dict[str, Any] = Field(default_factory=dict)

class Document(BaseModel):
    """
    Represents an immutable, validated raw document.
    """
    id: str  # Deterministic SHA-256 hash of the text content
    text: str
    content_hash: str
    metadata: DocumentMetadata

    @classmethod
    def create(
        cls,
        text: str,
        source_path: str,
        file_type: str,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> "Document":
        """
        Creates a Document model with an auto-calculated SHA-256 hash to prevent duplicates.
        """
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        metadata = DocumentMetadata(
            source_path=source_path,
            file_type=file_type,
            custom=custom_metadata or {}
        )
        return cls(
            id=content_hash,
            text=text,
            content_hash=content_hash,
            metadata=metadata
        )

class Chunk(BaseModel):
    """
    Represents a verified, chunked segment of a parent Document.
    """
    id: str  # Deterministic sequential ID: {parent_doc_id}#chunk_{index}
    parent_doc_id: str
    text: str
    metadata: DocumentMetadata
    index: int
    content_hash: str

    @classmethod
    def create(
        cls,
        parent_doc_id: str,
        text: str,
        index: int,
        metadata: DocumentMetadata
    ) -> "Chunk":
        """
        Creates a Chunk model with a sequential ID and its own cryptographic content hash.
        """
        chunk_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        chunk_id = f"{parent_doc_id}#chunk_{index}"
        return cls(
            id=chunk_id,
            parent_doc_id=parent_doc_id,
            text=text,
            metadata=metadata,
            index=index,
            content_hash=chunk_hash
        )

class RecursiveLinguisticSplitter:
    """
    Splits text recursively using structural delimiters (headers, paragraphs, lines, spaces)
    to keep related concepts together within chunk_size limit.
    """
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Delimiter hierarchy prioritizing Markdown headers, paragraphs, lists, words, then chars.
        self.separators = separators or ["\n# ", "\n## ", "\n### ", "\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # Fallback to strict character splitting if we run out of delimiters
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), max(1, self.chunk_size - self.chunk_overlap))
            ]

        separator = separators[0]
        next_separators = separators[1:]

        # Split text by current separator
        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        final_chunks = []
        current_doc = []
        current_len = 0

        for split in splits:
            # If a single split is larger than the chunk size, split it recursively
            if len(split) > self.chunk_size:
                # Flush the current buffer first
                if current_doc:
                    merged = separator.join(current_doc)
                    final_chunks.extend(self._merge_splits(merged.split(separator) if separator != "" else list(merged), separator))
                    current_doc = []
                    current_len = 0

                # Recursively split the large block
                recursive_splits = self._split_text(split, next_separators)
                final_chunks.extend(recursive_splits)
            else:
                # Check if adding this split exceeds chunk_size
                separator_len = len(separator) if current_doc else 0
                if current_len + separator_len + len(split) > self.chunk_size:
                    if current_doc:
                        merged = separator.join(current_doc)
                        final_chunks.append(merged)

                        # Keep elements that fit within overlap limits
                        overlap_doc = []
                        overlap_len = 0
                        for d in reversed(current_doc):
                            d_sep_len = len(separator) if overlap_doc else 0
                            if overlap_len + d_sep_len + len(d) <= self.chunk_overlap:
                                overlap_doc.insert(0, d)
                                overlap_len += d_sep_len + len(d)
                            else:
                                break
                        current_doc = overlap_doc
                        current_len = overlap_len

                current_doc.append(split)
                separator_len = len(separator) if len(current_doc) > 1 else 0
                current_len += separator_len + len(split)

        if current_doc:
            final_chunks.append(separator.join(current_doc))

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs = []
        current_doc = []
        total = 0
        for s in splits:
            sep_len = len(separator) if current_doc else 0
            if total + sep_len + len(s) > self.chunk_size:
                if current_doc:
                    docs.append(separator.join(current_doc))
                    # Keep overlap
                    overlap_doc = []
                    overlap_len = 0
                    for d in reversed(current_doc):
                        d_sep_len = len(separator) if overlap_doc else 0
                        if overlap_len + d_sep_len + len(d) <= self.chunk_overlap:
                            overlap_doc.insert(0, d)
                            overlap_len += d_sep_len + len(d)
                        else:
                            break
                    current_doc = overlap_doc
                    total = overlap_len
            current_doc.append(s)
            sep_len = len(separator) if len(current_doc) > 1 else 0
            total += sep_len + len(s)
        if current_doc:
            docs.append(separator.join(current_doc))
        return docs

def split_document(
    doc: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: Optional[List[str]] = None
) -> List[Chunk]:
    """
    Splits a Document into a list of Chunk objects, preserving and expanding metadata.
    """
    splitter = RecursiveLinguisticSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    text_chunks = splitter.split_text(doc.text)
    
    chunks = []
    for idx, text in enumerate(text_chunks):
        chunks.append(
            Chunk.create(
                parent_doc_id=doc.id,
                text=text,
                index=idx,
                metadata=doc.metadata
            )
        )
    return chunks
