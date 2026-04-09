"""
PDF ingestion and semantic chunking for Utah FORGE daily drilling reports.

Loads PDFs → extracts raw text → splits into semantically coherent chunks →
attaches metadata (well, date, section) to each chunk for downstream retrieval.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


# ── Section header patterns found in Utah FORGE daily drilling report PDFs ──
_SECTION_PATTERNS = [
    (r"(?i)drilling\s+operations?", "Drilling Operations"),
    (r"(?i)mud\s+report|mud\s+properties", "Mud Report"),
    (r"(?i)bit\s+record|bit\s+information", "Bit Record"),
    (r"(?i)directional\s+survey|deviation\s+survey", "Directional Survey"),
    (r"(?i)daily\s+costs?|cost\s+summary", "Daily Cost"),
    (r"(?i)formation\s+evaluation|lithology", "Formation Evaluation"),
    (r"(?i)casing\s+(record|information|summary)", "Casing Record"),
    (r"(?i)wellbore\s+schematic|well\s+schematic", "Wellbore Schematic"),
    (r"(?i)morning\s+report|toolpusher\s+report", "Morning Report"),
    (r"(?i)pump\s+information|hydraulics", "Hydraulics"),
    (r"(?i)time\s+breakdown|time\s+distribution", "Time Breakdown"),
]


def _detect_section(text: str) -> Optional[str]:
    """Return the first matching section name for a block of text."""
    for pattern, name in _SECTION_PATTERNS:
        if re.search(pattern, text):
            return name
    return None


def _extract_date_from_text(text: str) -> Optional[str]:
    """Try to parse report date from page text."""
    patterns = [
        r"(?:Date|Report Date)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"(?:Date|Report Date)[:\s]+(\w+ \d{1,2},?\s*\d{4})",
        r"\b(\d{4}-\d{2}-\d{2})\b",
        r"\b(\d{1,2}/\d{1,2}/\d{4})\b",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()
    return None


def _extract_report_number(text: str) -> Optional[int]:
    """Try to parse sequential report number."""
    m = re.search(r"(?:Daily|Report)\s*#?\s*(\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _extract_well_name(text: str) -> Optional[str]:
    """Try to parse well name from page text."""
    m = re.search(r"Well(?:\s+Name)?[:\s]+([A-Z0-9\-/()]+)", text)
    if m:
        return m.group(1).strip()
    # Utah FORGE specific
    m2 = re.search(r"\b(78B-32|16A\(78\)-32|58-32)\b", text)
    if m2:
        return m2.group(1)
    return None


class DrillReportLoader:
    """
    Loads a folder of daily drilling report PDFs and returns LangChain Documents
    with semantic chunking and rich metadata.
    """

    def __init__(
        self,
        pdf_dir: str | Path,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.pdf_dir = Path(pdf_dir)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def load_all(self) -> list[Document]:
        """Load every PDF in pdf_dir and return a flat list of chunks."""
        docs: list[Document] = []
        pdf_files = sorted(self.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(
                f"No PDFs found in {self.pdf_dir}. "
                "Run `python scripts/download_data.py` first."
            )
        for path in pdf_files:
            docs.extend(self.load_pdf(path))
        return docs

    def load_pdf(self, pdf_path: str | Path) -> list[Document]:
        """Load a single PDF → extract text per page → chunk → attach metadata."""
        pdf_path = Path(pdf_path)
        pages = self._extract_pages(pdf_path)
        full_text = "\n\n".join(p["text"] for p in pages)

        # Global doc-level metadata (parsed from first two pages)
        header_text = "\n".join(p["text"] for p in pages[:2])
        doc_meta = {
            "source_file": pdf_path.name,
            "well_name": _extract_well_name(header_text),
            "report_date": _extract_date_from_text(header_text),
            "report_number": _extract_report_number(header_text),
            "total_pages": len(pages),
        }

        # Split full text into chunks
        raw_chunks = self.splitter.split_text(full_text)
        documents: list[Document] = []
        for i, chunk_text in enumerate(raw_chunks):
            meta = {
                **doc_meta,
                "chunk_index": i,
                "section": _detect_section(chunk_text),
            }
            documents.append(Document(page_content=chunk_text, metadata=meta))

        return documents

    # ── Internal ──────────────────────────────────────────────────────────────

    def _extract_pages(self, pdf_path: Path) -> list[dict]:
        """Use pdfplumber to extract text from each page, preserving layout."""
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text(layout=True) or ""
                # Also try extracting tables as structured text
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        for row in table:
                            row_text = " | ".join(
                                str(cell) if cell is not None else "" for cell in row
                            )
                            text += "\n" + row_text
                pages.append({"page_num": page_num, "text": text.strip()})
        return pages
