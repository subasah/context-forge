"""
ContextForge Multi-Format Document Ingestion Engine.
Parses Markdown, PDF, Plaintext, JSON, and Source Code into canonical Document models.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from context_forge.core.document import Document


class DocumentParser:
    """Production parser supporting multiple file types and raw strings."""

    @classmethod
    def from_text(cls, text: str, title: str = "Raw Text", metadata: Optional[Dict[str, Any]] = None) -> Document:
        meta = metadata or {}
        meta.setdefault("title", title)
        meta.setdefault("source", "raw_text")
        meta.setdefault("format", "txt")
        return Document(text=text, metadata=meta)

    @classmethod
    def from_file(cls, file_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> Document:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        meta = metadata or {}
        meta.setdefault("source", str(path))
        meta.setdefault("title", path.stem.replace("_", " ").title())
        meta.setdefault("filename", path.name)
        meta.setdefault("extension", path.suffix.lower())
        meta.setdefault("file_size_bytes", path.stat().st_size)

        ext = path.suffix.lower()

        if ext in [".txt", ".md", ".markdown"]:
            text = path.read_text(encoding="utf-8", errors="replace")
            meta.setdefault("format", "markdown" if ext in [".md", ".markdown"] else "txt")
            return Document(text=text, metadata=meta)

        elif ext == ".pdf":
            return cls._parse_pdf(path, meta)

        elif ext == ".json":
            return cls._parse_json(path, meta)

        elif ext in [".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".yaml", ".yml", ".sql"]:
            text = path.read_text(encoding="utf-8", errors="replace")
            meta.setdefault("format", "code")
            meta.setdefault("language", ext.lstrip("."))
            return Document(text=text, metadata=meta)

        else:
            # Fallback to UTF-8 text read
            text = path.read_text(encoding="utf-8", errors="replace")
            meta.setdefault("format", "raw")
            return Document(text=text, metadata=meta)

    @classmethod
    def _parse_pdf(cls, path: Path, metadata: Dict[str, Any]) -> Document:
        """Parse PDF with page-level boundary markers."""
        pages_text: List[str] = []
        total_pages = 0

        # Try fitz (PyMuPDF) first
        try:
            import fitz
            doc = fitz.open(str(path))
            total_pages = len(doc)
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                page_text = page.get_text("text").strip()
                if page_text:
                    pages_text.append(f"--- [Page {page_num + 1}] ---\n{page_text}")
            doc.close()
        except ImportError:
            # Fallback to pypdf
            try:
                import pypdf
                reader = pypdf.PdfReader(str(path))
                total_pages = len(reader.pages)
                for idx, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        pages_text.append(f"--- [Page {idx + 1}] ---\n{page_text.strip()}")
            except Exception as e:
                pages_text.append(f"[Error reading PDF with pypdf: {e}]")

        full_text = "\n\n".join(pages_text) if pages_text else "[Empty PDF Document]"
        metadata.setdefault("format", "pdf")
        metadata.setdefault("page_count", total_pages)
        return Document(text=full_text, metadata=metadata)

    @classmethod
    def _parse_json(cls, path: Path, metadata: Dict[str, Any]) -> Document:
        """Parse JSON documents into clean readable representations."""
        raw_content = path.read_text(encoding="utf-8", errors="replace")
        try:
            data = json.loads(raw_content)
            if isinstance(data, list):
                # Array of records
                formatted = "\n\n".join(
                    f"Record {i + 1}:\n" + json.dumps(item, indent=2)
                    for i, item in enumerate(data)
                )
            else:
                formatted = json.dumps(data, indent=2)
        except Exception:
            formatted = raw_content

        metadata.setdefault("format", "json")
        return Document(text=formatted, metadata=metadata)
