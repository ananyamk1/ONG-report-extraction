"""
SQLAlchemy SQLite storage for structured drilling parameters.

Each PDF → one row in the `drilling_reports` table.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Session

from src.config import DB_PATH
from src.models import DrillingParameters


class Base(DeclarativeBase):
    pass


class DrillingReport(Base):
    """ORM model — one row per daily drilling report."""
    __tablename__ = "drilling_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    source_file = Column(String(255), unique=True, nullable=False)

    # ── Identification ────────────────────────────────────────────────────────
    well_name = Column(String(50))
    report_date = Column(String(20))
    report_number = Column(Integer)
    rig_name = Column(String(100))
    operator = Column(String(150))

    # ── Depth ─────────────────────────────────────────────────────────────────
    hole_depth_ft = Column(Float)
    bit_depth_ft = Column(Float)
    daily_footage_ft = Column(Float)
    casing_depth_ft = Column(Float)

    # ── Drilling mechanics ────────────────────────────────────────────────────
    rop_ft_per_hr = Column(Float)
    wob_klbs = Column(Float)
    rpm = Column(Float)
    torque_ftlbs = Column(Float)
    hook_load_klbs = Column(Float)

    # ── Hydraulics ────────────────────────────────────────────────────────────
    flow_rate_gpm = Column(Float)
    standpipe_pressure_psi = Column(Float)
    pump_pressure_psi = Column(Float)
    pump_strokes_per_min = Column(Float)

    # ── Mud ───────────────────────────────────────────────────────────────────
    mud_weight_in_ppg = Column(Float)
    mud_weight_out_ppg = Column(Float)
    mud_type = Column(String(200))
    mud_viscosity_cp = Column(Float)
    pit_volume_bbls = Column(Float)
    mud_loss_bbls = Column(Float)

    # ── Temperature ───────────────────────────────────────────────────────────
    temp_in_f = Column(Float)
    temp_out_f = Column(Float)

    # ── Gas / formation ───────────────────────────────────────────────────────
    co2_pct = Column(Float)
    formation_name = Column(String(200))

    # ── Directional ───────────────────────────────────────────────────────────
    inclination_deg = Column(Float)
    azimuth_deg = Column(Float)

    # ── Bit ───────────────────────────────────────────────────────────────────
    bit_size_in = Column(Float)
    bit_type = Column(String(100))

    # ── Operations ────────────────────────────────────────────────────────────
    days_since_spud = Column(Integer)
    operations_summary = Column(Text)


class DrillDatabase:
    """Thin wrapper around SQLite for storing and querying drilling records."""

    def __init__(self, db_path: str | Path = DB_PATH):
        self._engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self._engine)

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(self, source_file: str, params: DrillingParameters) -> int:
        """
        Insert or update a record for the given source PDF.
        Returns the row id.
        """
        with Session(self._engine) as session:
            # Check if already exists
            stmt = select(DrillingReport).where(
                DrillingReport.source_file == source_file
            )
            record = session.scalars(stmt).first()

            data = params.model_dump()
            if record is None:
                record = DrillingReport(source_file=source_file, **data)
                session.add(record)
            else:
                for k, v in data.items():
                    if hasattr(record, k):
                        setattr(record, k, v)

            session.commit()
            session.refresh(record)
            return record.id

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_all(self) -> pd.DataFrame:
        """Return all records as a pandas DataFrame."""
        with Session(self._engine) as session:
            records = session.scalars(select(DrillingReport)).all()
            return pd.DataFrame(
                [
                    {c.name: getattr(r, c.name) for c in DrillingReport.__table__.columns}
                    for r in records
                ]
            )

    def get_by_date(self, date: str) -> Optional[DrillingReport]:
        with Session(self._engine) as session:
            return session.scalars(
                select(DrillingReport).where(DrillingReport.report_date == date)
            ).first()

    def get_by_well(self, well_name: str) -> pd.DataFrame:
        with Session(self._engine) as session:
            records = session.scalars(
                select(DrillingReport).where(DrillingReport.well_name == well_name)
            ).all()
            return pd.DataFrame(
                [
                    {c.name: getattr(r, c.name) for c in DrillingReport.__table__.columns}
                    for r in records
                ]
            )

    def count(self) -> int:
        with Session(self._engine) as session:
            return session.scalars(select(DrillingReport)).all().__len__()

    def summary_stats(self) -> pd.DataFrame:
        """Descriptive statistics across all ingested reports."""
        df = self.get_all()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        # Drop id/report_number from stats
        for c in ["id", "report_number", "days_since_spud"]:
            if c in numeric_cols:
                numeric_cols.remove(c)
        return df[numeric_cols].describe()
