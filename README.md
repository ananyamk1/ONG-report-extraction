# GenAI PDF Automation Pipeline — Oil & Gas Drilling Reports

A LangChain pipeline that ingests daily drilling report PDFs and automatically extracts **28 operational parameters** into a structured SQLite database, with a semantic RAG layer so engineers can query the entire report corpus in natural language — no manual reading required.

---

## Dataset

**Utah FORGE Well 78B-32 Daily Drilling Reports**
- **Source:** [DOE Geothermal Data Repository](https://gdr.openei.org/submissions/1330)
- **DOI:** https://doi.org/10.15121/1814488
- **License:** Creative Commons Attribution 4.0 (CC-BY 4.0)
- **Contents:** ~35 PDF daily drilling reports (June 27 – July 31, 2021) from a geothermal research well in Milford, Utah, drilled to ~8,500 ft
- **Why this dataset:** These reports are representative of the real-world tedious daily task O&G engineers face — each PDF must be manually read to track trends in ROP, mud weight, temperatures, and formation changes across weeks of drilling. This pipeline eliminates that manual work.

### Research Gap Addressed

Daily drilling reports are produced for every active well, every single day — across thousands of wells globally. Yet they remain locked in PDFs and require engineers to manually search, read, and tabulate data. Key pain points:

- **No structured data:** Reports are narrative + table PDFs, not machine-readable
- **Trend tracking is manual:** Spotting ROP decline, mud weight drift, or CO2 spikes requires reading every page
- **Cross-report queries are impossible:** "What was our average WOB in granite vs. granite transitions?" takes hours
- **Audit and compliance:** Regulators require accurate parameter logs — manually compiled from PDFs

This pipeline solves all four.

---

## Extracted Parameters (28)

| Category | Parameters |
|---|---|
| **Identification** | well_name, report_date, report_number, rig_name, operator |
| **Depth** | hole_depth_ft, bit_depth_ft, daily_footage_ft, casing_depth_ft |
| **Drilling Mechanics** | rop_ft_per_hr, wob_klbs, rpm, torque_ftlbs, hook_load_klbs |
| **Hydraulics** | flow_rate_gpm, standpipe_pressure_psi, pump_pressure_psi, pump_strokes_per_min |
| **Mud Properties** | mud_weight_in_ppg, mud_weight_out_ppg, mud_type, mud_viscosity_cp, pit_volume_bbls, mud_loss_bbls |
| **Temperature** | temp_in_f, temp_out_f |
| **Gas / Formation** | co2_pct, formation_name |
| **Directional** | inclination_deg, azimuth_deg |
| **Bit** | bit_size_in, bit_type |
| **Operations** | days_since_spud, operations_summary |

---

## Architecture

```
PDF Daily Reports
      │
      ▼
┌─────────────────────┐
│   DrillReportLoader │  pdfplumber — extracts text + tables per page
│   (pdf_loader.py)   │  RecursiveCharacterTextSplitter — semantic chunks
└────────┬────────────┘  Metadata tagging: well, date, section, chunk_index
         │
         ├──────────────────────────────────────┐
         ▼                                      ▼
┌──────────────────────┐            ┌───────────────────────┐
│   ParameterExtractor │            │    DrillVectorStore   │
│   (extractor.py)     │            │    (vectorstore.py)   │
│                      │            │                       │
│  LLM (GPT-4o-mini or │            │  ChromaDB + local     │
│  Claude Haiku) with  │            │  HuggingFace embeds   │
│  Pydantic JSON schema│            │  (all-MiniLM-L6-v2)   │
└────────┬─────────────┘            └──────────┬────────────┘
         │                                     │
         ▼                                     ▼
┌──────────────────────┐            ┌───────────────────────┐
│    DrillDatabase     │            │      DrillingQA       │
│    (database.py)     │            │    (qa_chain.py)      │
│                      │            │                       │
│  SQLite via          │            │  RAG chain: retrieve  │
│  SQLAlchemy ORM      │            │  top-k chunks → LLM   │
│  28-column table     │            │  → natural language   │
└──────────────────────┘            │  answer with source   │
                                    │  citations            │
                                    └───────────────────────┘
```

---

## Setup

```bash
# 1. Clone
git clone https://github.com/ananyamk1/ong-report-extraction
cd ong-report-extraction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY or ANTHROPIC_API_KEY

# 4. Download dataset
python scripts/download_data.py
# Follow the printed instructions to download PDFs from:
# https://gdr.openei.org/submissions/1330
# Place PDFs in data/raw_pdfs/

# Or if you downloaded the ZIP:
python scripts/download_data.py --extract path/to/78B-32_DailyReports.zip
```

---

## Running the Pipeline

```bash
# Process all PDFs (chunk + vectorise + extract parameters)
python pipeline_run.py

# Re-process everything (if you update the extraction schema)
python pipeline_run.py --force

# Chunk + vectorise only (no LLM calls — no API key needed)
python pipeline_run.py --skip-extraction

# View extraction statistics
python pipeline_run.py --stats
```

---

## Querying the Reports

```bash
# Interactive Q&A mode
python demo.py

# Single question
python demo.py --question "What was the average ROP in July 2021?"
python demo.py --question "Were there any mud losses above 50 bbls?"

# View structured database as table
python demo.py --show-table
```

### Example questions engineers can ask:
- *"What was the mud weight trend over the drilling campaign?"*
- *"Which days had the highest CO2 readings and what was the depth?"*
- *"Summarise all casing operations — depths and dates."*
- *"What formation was being drilled when ROP dropped below 5 ft/hr?"*
- *"Compare inlet vs outlet mud temperature — is there a geothermal gradient?"*

---

## Programmatic API

```python
from src.pipeline import DrillPipeline
from src.qa_chain import DrillingQA
from src.vectorstore import DrillVectorStore
from src.database import DrillDatabase

# Run the pipeline
pipeline = DrillPipeline()
pipeline.run()

# Query with RAG
vs = DrillVectorStore().load()
qa = DrillingQA(vs)
result = qa.ask("What was the ROP when drilling through granite?")
print(result["answer"])

# Access structured database
db = DrillDatabase()
df = db.get_all()          # all 28 parameters as pandas DataFrame
print(db.summary_stats())  # descriptive statistics
```

---

## File Structure

```
ong-report-extraction/
├── src/
│   ├── config.py          # env vars, paths
│   ├── models.py          # Pydantic schema (28 parameters)
│   ├── pdf_loader.py      # PDF ingestion + semantic chunking
│   ├── extractor.py       # LLM structured extraction
│   ├── database.py        # SQLAlchemy SQLite ORM
│   ├── vectorstore.py     # ChromaDB + local embeddings
│   ├── pipeline.py        # Orchestrator
│   └── qa_chain.py        # RAG Q&A chain
├── scripts/
│   └── download_data.py   # Dataset download helper
├── data/
│   ├── raw_pdfs/          # PDF daily reports (gitignored)
│   └── db/                # SQLite database (gitignored)
├── vectorstore/           # ChromaDB persistent store (gitignored)
├── pipeline_run.py        # CLI entry point
├── demo.py                # Interactive RAG demo
├── requirements.txt
├── .env.example
└── README.md
```

---

## Tech Stack

| Component | Library |
|---|---|
| PDF parsing | pdfplumber, pypdf |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local, free) |
| Vector store | ChromaDB (persisted to disk) |
| LLM extraction | OpenAI GPT-4o-mini or Anthropic Claude Haiku |
| Structured output | Pydantic v2 |
| Relational DB | SQLite via SQLAlchemy ORM |
| RAG chain | LangChain RetrievalQA |
| CLI | Rich + Typer |

---

## Citation

```
Energy Innovations International / University of Utah (2021).
Utah FORGE Well 78B-32 Daily Drilling Reports and Logs.
Geothermal Data Repository. https://doi.org/10.15121/1814488
License: CC-BY 4.0
```
