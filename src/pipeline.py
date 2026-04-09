"""
Main pipeline orchestrator.

Run order:
  1. Load all PDFs from data/raw_pdfs/
  2. Chunk + tag with metadata
  3. Store chunks in ChromaDB vector store
  4. Extract structured parameters via LLM
  5. Persist structured records to SQLite

Usage:
  from src.pipeline import DrillPipeline
  pipeline = DrillPipeline()
  pipeline.run()                    # process all new PDFs
  pipeline.run(force=True)          # re-process everything
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from src.config import PDF_DIR
from src.database import DrillDatabase
from src.extractor import ParameterExtractor
from src.pdf_loader import DrillReportLoader
from src.vectorstore import DrillVectorStore

console = Console()


class DrillPipeline:
    """
    End-to-end pipeline: PDF ingestion → chunking → vector indexing
    → LLM extraction → structured database storage.
    """

    def __init__(
        self,
        pdf_dir: str | Path = PDF_DIR,
        skip_extraction: bool = False,
    ):
        self.pdf_dir = Path(pdf_dir)
        self.skip_extraction = skip_extraction

        self._loader = DrillReportLoader(self.pdf_dir)
        self._vectorstore = DrillVectorStore()
        self._db = DrillDatabase()
        self._extractor: Optional[ParameterExtractor] = None

        if not skip_extraction:
            self._extractor = ParameterExtractor()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, force: bool = False) -> dict:
        """
        Process all PDFs in pdf_dir.

        Args:
            force: If True, re-process PDFs already in the database.

        Returns:
            Summary dict with counts of processed/skipped/failed reports.
        """
        pdf_files = sorted(self.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            console.print(
                f"[red]No PDFs found in {self.pdf_dir}.[/red]\n"
                "Run:  python scripts/download_data.py"
            )
            return {"processed": 0, "skipped": 0, "failed": 0}

        existing = self._get_existing_files()

        to_process = pdf_files if force else [
            f for f in pdf_files if f.name not in existing
        ]

        console.print(
            f"\n[bold cyan]Utah FORGE Drilling Report Pipeline[/bold cyan]\n"
            f"  PDFs found:    {len(pdf_files)}\n"
            f"  Already done:  {len(pdf_files) - len(to_process)}\n"
            f"  To process:    {len(to_process)}\n"
        )

        results = {"processed": 0, "skipped": len(pdf_files) - len(to_process), "failed": 0}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing reports...", total=len(to_process))

            for pdf_path in to_process:
                progress.update(task, description=f"[cyan]{pdf_path.name}[/cyan]")
                try:
                    self._process_one(pdf_path)
                    results["processed"] += 1
                except Exception as e:
                    console.print(f"  [red]FAILED[/red] {pdf_path.name}: {e}")
                    results["failed"] += 1
                finally:
                    progress.advance(task)

        self._print_summary(results)
        return results

    def run_one(self, pdf_path: str | Path) -> dict:
        """Process a single PDF. Returns the extracted parameters as a dict."""
        pdf_path = Path(pdf_path)
        params = self._process_one(pdf_path)
        return params.model_dump()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _process_one(self, pdf_path: Path):
        """Load → chunk → vectorise → extract → store."""
        # 1. Load & chunk
        docs = self._loader.load_pdf(pdf_path)

        # 2. Vector store
        self._vectorstore.add_documents(docs)

        # 3. LLM extraction (optional — skip if no API key configured)
        if self._extractor:
            params = self._extractor.extract(docs)
        else:
            from src.models import DrillingParameters
            meta = docs[0].metadata if docs else {}
            params = DrillingParameters(
                well_name=meta.get("well_name"),
                report_date=meta.get("report_date"),
                report_number=meta.get("report_number"),
            )

        # 4. Persist to SQLite
        self._db.upsert(pdf_path.name, params)
        return params

    def _get_existing_files(self) -> set[str]:
        try:
            df = self._db.get_all()
            return set(df["source_file"].tolist()) if not df.empty else set()
        except Exception:
            return set()

    def _print_summary(self, results: dict):
        table = Table(title="Pipeline Summary", show_header=True)
        table.add_column("Status", style="bold")
        table.add_column("Count", justify="right")
        table.add_row("[green]Processed[/green]", str(results["processed"]))
        table.add_row("[yellow]Skipped (already done)[/yellow]", str(results["skipped"]))
        table.add_row("[red]Failed[/red]", str(results["failed"]))
        table.add_row(
            "Total in DB",
            str(self._db.count()),
        )
        table.add_row(
            "Total vectors",
            str(self._vectorstore.count()),
        )
        console.print(table)
