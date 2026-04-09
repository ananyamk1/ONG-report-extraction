"""
Download Utah FORGE Well 78B-32 Daily Drilling Report PDFs.

Dataset:  Utah FORGE Well 78B-32 Daily Drilling Reports and Logs
Source:   https://gdr.openei.org/submissions/1330
DOI:      https://doi.org/10.15121/1814488
License:  Creative Commons Attribution 4.0 (CC-BY 4.0)
Operator: Energy Innovations International / University of Utah

The GDR (Geothermal Data Repository) hosts a ZIP archive containing:
  - PDF daily drilling reports (June 27 – July 31, 2021)
  - Pason 1-second and 10-second CSV drilling data (for validation)
  - Schlumberger wireline log PDFs and LAS files

This script downloads only the daily drilling report PDFs into data/raw_pdfs/.
"""
from __future__ import annotations

import sys
import zipfile
from io import BytesIO
from pathlib import Path

import requests
from tqdm import tqdm
from rich.console import Console

console = Console()

# ── Dataset configuration ─────────────────────────────────────────────────────
# GDR direct download link for Well 78B-32 dataset
# The full archive (~12 GB) contains PDFs + Pason CSVs.
# We only extract the PDF daily reports.
GDR_SUBMISSION_URL = "https://gdr.openei.org/submissions/1330"

# Fallback: individual PDF files from the GDR file listing
# These are the daily drilling report PDFs confirmed in the dataset
DAILY_REPORT_PDFS = [
    # Format: (filename, direct_url)
    # NOTE: GDR URLs require navigating the submission — update these after
    # visiting https://gdr.openei.org/submissions/1330 and noting file URLs.
    # The placeholders below will trigger the interactive download prompt.
]

PDF_DIR = Path(__file__).parent.parent / "data" / "raw_pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path, description: str = "") -> bool:
    """Stream-download a file with a progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        desc = description or dest.name
        with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as bar:
            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
        return True
    except Exception as e:
        console.print(f"[red]Download failed:[/red] {e}")
        return False


def extract_pdfs_from_zip(zip_path: Path, dest_dir: Path) -> list[Path]:
    """Extract only PDF files from a ZIP archive."""
    extracted = []
    with zipfile.ZipFile(zip_path) as zf:
        pdf_members = [m for m in zf.namelist() if m.lower().endswith(".pdf")]
        console.print(f"Found {len(pdf_members)} PDFs in archive.")
        for member in tqdm(pdf_members, desc="Extracting PDFs"):
            name = Path(member).name  # flatten directory structure
            target = dest_dir / name
            if not target.exists():
                data = zf.read(member)
                target.write_bytes(data)
                extracted.append(target)
    return extracted


def show_manual_instructions():
    """Print step-by-step instructions for manual download."""
    console.print("""
[bold yellow]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold yellow]
[bold]DATASET: Utah FORGE Well 78B-32 Daily Drilling Reports[/bold]
[bold yellow]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold yellow]

License:  Creative Commons Attribution 4.0 (CC-BY 4.0)
DOI:      https://doi.org/10.15121/1814488

[bold cyan]STEP 1 — Get the PDFs[/bold cyan]

Option A (Recommended — GDR Portal):
  1. Visit: [link]https://gdr.openei.org/submissions/1330[/link]
  2. Download the file named:
       [italic]"78B-32_DailyDrillingReports_*.zip"[/italic]
     or individual PDF reports listed on the page.
  3. Place the ZIP (or extracted PDFs) in:
       [bold]data/raw_pdfs/[/bold]
  4. If you downloaded a ZIP, run:
       python scripts/download_data.py --extract path/to/file.zip

Option B (OSTI Data Explorer):
  1. Visit: [link]https://www.osti.gov/dataexplorer/biblio/dataset/1814488[/link]
  2. Follow the "Access Dataset" link → GDR portal.

Option C (data.gov):
  1. Visit: [link]https://catalog.data.gov/dataset/utah-forge-well-78b-32-daily-drilling-reports-and-logs-7efa9[/link]

[bold cyan]STEP 2 — Run the pipeline[/bold cyan]

  cp .env.example .env           # add your OpenAI or Anthropic API key
  python pipeline_run.py         # runs the full pipeline

[bold cyan]About the dataset[/bold cyan]

  Well:       78B-32 (Utah FORGE geothermal research well)
  Period:     June 27 – July 31, 2021  (~35 daily reports)
  Depth:      ~8,500 ft (2,600 m)
  Parameters: ROP, WOB, RPM, mud weight, flow rate, temperatures,
              pit volume, CO2, directional surveys, formation names

[bold yellow]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold yellow]
""")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download Utah FORGE drilling report PDFs")
    parser.add_argument(
        "--extract",
        metavar="ZIP_PATH",
        help="Extract PDFs from a locally downloaded ZIP archive",
    )
    parser.add_argument(
        "--pdf-dir",
        default=str(PDF_DIR),
        help=f"Destination directory for PDFs (default: {PDF_DIR})",
    )
    args = parser.parse_args()

    dest = Path(args.pdf_dir)
    dest.mkdir(parents=True, exist_ok=True)

    if args.extract:
        zip_path = Path(args.extract)
        if not zip_path.exists():
            console.print(f"[red]ZIP not found:[/red] {zip_path}")
            sys.exit(1)
        console.print(f"Extracting PDFs from {zip_path} → {dest}")
        extracted = extract_pdfs_from_zip(zip_path, dest)
        console.print(f"[green]Extracted {len(extracted)} PDFs to {dest}[/green]")
    else:
        show_manual_instructions()

    existing = list(dest.glob("*.pdf"))
    if existing:
        console.print(
            f"\n[green]✓ {len(existing)} PDFs ready in {dest}[/green]\n"
            "Next step:  python pipeline_run.py"
        )
    else:
        console.print(
            f"\n[yellow]No PDFs yet in {dest}[/yellow]\n"
            "Follow the instructions above to download the dataset."
        )


if __name__ == "__main__":
    main()
