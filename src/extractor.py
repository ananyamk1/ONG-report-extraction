"""
LLM-based structured parameter extraction from drilling report text.

For each daily report, feeds the full document text to an LLM with a
structured output schema (Pydantic) and returns a DrillingParameters record.
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from src.config import LLM_PROVIDER, LLM_MODEL, OPENAI_API_KEY, ANTHROPIC_API_KEY
from src.models import DrillingParameters


_SYSTEM_PROMPT = """\
You are an expert petroleum engineer and data extraction specialist.
You will be given the full text of a daily drilling report from the Utah FORGE \
geothermal/oil-and-gas well project (Well 78B-32 or similar).

Your job is to extract every operational parameter you can find and return \
a single JSON object matching the schema provided.

Rules:
- Return ONLY valid JSON, no markdown fences, no explanation.
- Use null for any parameter not mentioned in the report.
- Units: depths in feet, pressures in psi, weights in klbs, temperature in °F, \
  volumes in barrels (bbls), flow rate in gpm, mud weight in ppg, RPM as integer.
- If a range is given (e.g. "WOB: 8-10 klbs") use the midpoint.
- For report_date, normalise to YYYY-MM-DD where possible.
- operations_summary: write 1-2 sentences summarising the main activities.
"""

_HUMAN_PROMPT = """\
Extract drilling parameters from the following daily drilling report text.

REPORT TEXT:
{report_text}

Return JSON matching this schema (all fields optional / nullable):
{schema}
"""


def _get_llm():
    """Instantiate LLM based on configured provider."""
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=LLM_MODEL or "claude-3-5-haiku-20241022",
            api_key=ANTHROPIC_API_KEY,
            temperature=0,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=LLM_MODEL or "gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            temperature=0,
        )


class ParameterExtractor:
    """
    Extracts 28 structured drilling parameters from a single PDF's text
    using an LLM with JSON-mode output.
    """

    def __init__(self):
        self._llm = _get_llm()
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", _HUMAN_PROMPT),
        ])
        self._schema_str = json.dumps(
            DrillingParameters.model_json_schema(), indent=2
        )
        self._chain = self._prompt | self._llm

    def extract(self, documents: list[Document]) -> DrillingParameters:
        """
        Given chunks from a single PDF, concatenate their text and extract
        structured parameters.

        Args:
            documents: List of Document chunks all from the same PDF.

        Returns:
            DrillingParameters instance (fields may be None if not found).
        """
        # Concatenate all chunks — truncate to ~12k chars to stay within context
        report_text = "\n\n".join(doc.page_content for doc in documents)
        report_text = report_text[:12000]

        response = self._chain.invoke({
            "report_text": report_text,
            "schema": self._schema_str,
        })

        raw_json = response.content.strip()
        # Strip markdown fences if the model added them anyway
        if raw_json.startswith("```"):
            raw_json = raw_json.split("```")[1]
            if raw_json.startswith("json"):
                raw_json = raw_json[4:]

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}\nRaw: {raw_json[:500]}")

        # Merge source metadata from the first chunk's metadata
        if documents:
            meta = documents[0].metadata
            data.setdefault("well_name", meta.get("well_name"))
            data.setdefault("report_date", meta.get("report_date"))
            data.setdefault("report_number", meta.get("report_number"))

        try:
            return DrillingParameters(**data)
        except ValidationError as e:
            # Best-effort: filter only known fields
            known = DrillingParameters.model_fields.keys()
            cleaned = {k: v for k, v in data.items() if k in known}
            return DrillingParameters(**cleaned)

    def extract_from_file(self, pdf_path: str | Path, loader=None) -> DrillingParameters:
        """Convenience: load a single PDF and extract."""
        from src.pdf_loader import DrillReportLoader
        if loader is None:
            loader = DrillReportLoader(Path(pdf_path).parent)
        docs = loader.load_pdf(pdf_path)
        return self.extract(docs)
