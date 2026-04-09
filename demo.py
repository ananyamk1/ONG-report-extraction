"""
Interactive demo — ask questions about ingested drilling reports.

  python demo.py                     # interactive Q&A loop
  python demo.py --question "What was the mud weight on July 15?"
  python demo.py --show-table        # print the structured database as a table
"""
from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown

from src.qa_chain import DrillingQA
from src.vectorstore import DrillVectorStore
from src.database import DrillDatabase

console = Console()

EXAMPLE_QUESTIONS = [
    "What was the average ROP across all reports?",
    "Which day had the highest mud temperature at outlet?",
    "Were there any significant mud losses? If so, when?",
    "What formation was being drilled in the most recent reports?",
    "Summarise the trend in hole depth and daily footage drilled.",
    "Were there any CO2 spikes above 1%? What was happening that day?",
    "What was the typical WOB and RPM range used on this well?",
    "Describe the casing program — what depths were casing strings set?",
]


def print_banner():
    console.print("""
[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]
[bold]  Utah FORGE Well 78B-32 — Drilling Report RAG Assistant[/bold]
[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]
  Dataset:  DOE/OSTI  |  License: CC-BY 4.0  |  ~35 daily reports
  Ask any question about the drilling data. Type [bold]quit[/bold] to exit.
  Type [bold]examples[/bold] to see sample questions.
[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]
""")


def show_examples():
    console.print("\n[bold]Example questions:[/bold]")
    for i, q in enumerate(EXAMPLE_QUESTIONS, 1):
        console.print(f"  {i}. {q}")
    console.print()


def show_table():
    db = DrillDatabase()
    df = db.get_all()
    if df.empty:
        console.print("[yellow]No records in database yet. Run pipeline_run.py first.[/yellow]")
        return

    display_cols = [
        "report_date", "well_name", "hole_depth_ft", "daily_footage_ft",
        "rop_ft_per_hr", "wob_klbs", "rpm", "mud_weight_in_ppg",
        "temp_out_f", "formation_name",
    ]
    available = [c for c in display_cols if c in df.columns]
    df_display = df.sort_values("report_date")[available].head(40)

    table = Table(title="Extracted Drilling Parameters — Utah FORGE 78B-32", show_lines=True)
    for col in available:
        table.add_column(col.replace("_", " ").title(), no_wrap=True)

    for _, row in df_display.iterrows():
        table.add_row(*[str(row[c]) if row[c] is not None else "—" for c in available])

    console.print(table)


def run_interactive(qa: DrillingQA):
    while True:
        try:
            question = console.input("\n[bold green]Question:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break
        if question.lower() == "examples":
            show_examples()
            continue

        with console.status("[dim]Thinking...[/dim]"):
            result = qa.ask(question)

        console.print("\n[bold yellow]Answer:[/bold yellow]")
        console.print(Markdown(result["answer"]))

        if result.get("source_chunks"):
            console.print(f"\n[dim]Sources: {', '.join(set(c.metadata.get('source_file', '?') for c in result['source_chunks']))}[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Drilling Report RAG Demo")
    parser.add_argument("--question", "-q", help="Single question (non-interactive)")
    parser.add_argument("--show-table", action="store_true", help="Print structured DB table")
    parser.add_argument("--well", help="Filter to a specific well name")
    args = parser.parse_args()

    if args.show_table:
        show_table()
        return

    vs = DrillVectorStore()
    try:
        vs.load()
        count = vs.count()
        if count == 0:
            console.print("[yellow]Vector store is empty. Run pipeline_run.py first.[/yellow]")
            return
        console.print(f"[dim]Loaded vector store: {count} chunks[/dim]")
    except Exception:
        console.print("[yellow]Vector store not found. Run pipeline_run.py first.[/yellow]")
        return

    qa = DrillingQA(vs)

    if args.question:
        result = qa.ask(args.question, well_filter=args.well)
        console.print(f"\n[bold yellow]Answer:[/bold yellow]\n{result['answer']}")
        return

    print_banner()
    run_interactive(qa)


if __name__ == "__main__":
    main()
