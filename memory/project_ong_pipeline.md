---
name: ong-report-extraction project
description: GenAI PDF automation pipeline for oil & gas drilling reports — LangChain + ChromaDB + SQLite
type: project
---

Pipeline built for github.com/ananyamk1/ong-report-extraction.

**Why:** O&G engineers must manually read PDF daily drilling reports every day to track operational parameters — a tedious but critical task. This pipeline automates extraction of 28 parameters into structured DB + RAG Q&A layer.

**Dataset:** Utah FORGE Well 78B-32 Daily Drilling Reports (DOE/OSTI)
- URL: https://gdr.openei.org/submissions/1330
- DOI: https://doi.org/10.15121/1814488
- License: CC-BY 4.0 (free, public)
- ~35 PDF daily drilling reports, June 27 – July 31, 2021
- Well depth ~8,500 ft, Milford Utah geothermal research well

**How to apply:** When asked about the project, reference Utah FORGE dataset. Parameters extracted include ROP, WOB, RPM, mud weight in/out, temperature in/out, pit volume, CO2%, formation name, inclination, azimuth, bit size, and more (28 total).
