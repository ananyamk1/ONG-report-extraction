"""
Entry point — runs the full drilling report pipeline.

  python pipeline_run.py                     # process all new PDFs
  python pipeline_run.py --force             # re-process everything
  python pipeline_run.py --skip-extraction   # chunk + vectorise only (no LLM)
  python pipeline_run.py --stats             # print DB stats only
"""
from __future__ import annotations

import argparse

from rich.console import Console

from src.pipeline import DrillPipeline
from src.database import DrillDatabase

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Utah FORGE Drilling Report Extraction Pipeline"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process PDFs already in the database.",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Only chunk + vectorise; skip LLM parameter extraction.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print database statistics and exit.",
    )
    parser.add_argument(
        "--pdf-dir",
        default=None,
        help="Override path to PDF directory.",
    )
    args = parser.parse_args()

    if args.stats:
        db = DrillDatabase()
        console.print(f"\n[bold]Total reports in DB:[/bold] {db.count()}")
        df = db.get_all()
        if not df.empty:
            console.print("\n[bold]Summary statistics (numeric parameters):[/bold]")
            console.print(db.summary_stats().to_string())
            console.print("\n[bold]Most recent reports:[/bold]")
            cols = ["source_file", "report_date", "hole_depth_ft", "rop_ft_per_hr",
                    "wob_klbs", "mud_weight_in_ppg", "temp_out_f"]
            available = [c for c in cols if c in df.columns]
            console.print(df.sort_values("report_date", ascending=False).head(5)[available].to_string(index=False))
        return

    kwargs = {"skip_extraction": args.skip_extraction}
    if args.pdf_dir:
        kwargs["pdf_dir"] = args.pdf_dir

    pipeline = DrillPipeline(**kwargs)
    pipeline.run(force=args.force)


if __name__ == "__main__":
    main()
