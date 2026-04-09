"""
Pydantic models for structured extraction of drilling report parameters.

Source dataset: Utah FORGE Well 78B-32 Daily Drilling Reports
DOI: https://doi.org/10.15121/1814488  |  License: CC-BY 4.0
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class DrillingParameters(BaseModel):
    """
    28 operational parameters extracted from a single daily drilling report.
    All units are as reported in Utah FORGE Well 78B-32 PDFs.
    """

    # ── Identification ────────────────────────────────────────────────────────
    well_name: Optional[str] = Field(None, description="Well identifier (e.g. 78B-32)")
    report_date: Optional[str] = Field(None, description="Date of the daily drilling report (YYYY-MM-DD or as found)")
    report_number: Optional[int] = Field(None, description="Sequential daily report number")
    rig_name: Optional[str] = Field(None, description="Drilling rig name or number")
    operator: Optional[str] = Field(None, description="Operating company (e.g. Energy Innovations International)")

    # ── Depth ─────────────────────────────────────────────────────────────────
    hole_depth_ft: Optional[float] = Field(None, description="Current total hole depth in feet")
    bit_depth_ft: Optional[float] = Field(None, description="Current bit depth in feet")
    daily_footage_ft: Optional[float] = Field(None, description="Footage drilled during this 24-hour period in feet")
    casing_depth_ft: Optional[float] = Field(None, description="Depth of most recent casing string set, in feet")

    # ── Drilling mechanics ────────────────────────────────────────────────────
    rop_ft_per_hr: Optional[float] = Field(None, description="Rate of penetration in ft/hr (average for day)")
    wob_klbs: Optional[float] = Field(None, description="Weight on bit in thousands of pounds (klbs)")
    rpm: Optional[float] = Field(None, description="Rotary speed at surface in RPM")
    torque_ftlbs: Optional[float] = Field(None, description="Surface torque in ft-lbs")
    hook_load_klbs: Optional[float] = Field(None, description="Hook load (string weight) in klbs")

    # ── Hydraulics ────────────────────────────────────────────────────────────
    flow_rate_gpm: Optional[float] = Field(None, description="Mud circulation flow rate in gallons per minute")
    standpipe_pressure_psi: Optional[float] = Field(None, description="Standpipe pressure in psi")
    pump_pressure_psi: Optional[float] = Field(None, description="Mud pump pressure in psi")
    pump_strokes_per_min: Optional[float] = Field(None, description="Pump strokes per minute (SPM)")

    # ── Mud properties ────────────────────────────────────────────────────────
    mud_weight_in_ppg: Optional[float] = Field(None, description="Mud weight at inlet (flow in) in pounds per gallon")
    mud_weight_out_ppg: Optional[float] = Field(None, description="Mud weight at outlet (flow out/return) in pounds per gallon")
    mud_type: Optional[str] = Field(None, description="Mud system description (e.g. freshwater gel, KCl polymer)")
    mud_viscosity_cp: Optional[float] = Field(None, description="Mud viscosity (funnel) in centipoise or seconds/qt")
    pit_volume_bbls: Optional[float] = Field(None, description="Active pit volume in barrels")
    mud_loss_bbls: Optional[float] = Field(None, description="Cumulative or daily mud lost to formation in barrels")

    # ── Temperature ───────────────────────────────────────────────────────────
    temp_in_f: Optional[float] = Field(None, description="Mud temperature at inlet (°F)")
    temp_out_f: Optional[float] = Field(None, description="Mud temperature at outlet/return (°F)")

    # ── Gas / formation ───────────────────────────────────────────────────────
    co2_pct: Optional[float] = Field(None, description="CO2 concentration in return gas (%)")
    formation_name: Optional[str] = Field(None, description="Formation or lithology being drilled (e.g. Mineral Mountains Granite)")

    # ── Directional ───────────────────────────────────────────────────────────
    inclination_deg: Optional[float] = Field(None, description="Borehole inclination in degrees from vertical")
    azimuth_deg: Optional[float] = Field(None, description="Borehole azimuth in degrees from north")

    # ── Bit ───────────────────────────────────────────────────────────────────
    bit_size_in: Optional[float] = Field(None, description="Drill bit diameter in inches")
    bit_type: Optional[str] = Field(None, description="Bit type (e.g. PDC, tri-cone, diamond)")

    # ── Operations ────────────────────────────────────────────────────────────
    days_since_spud: Optional[int] = Field(None, description="Number of days since well was spudded")
    operations_summary: Optional[str] = Field(None, description="Brief narrative of key operations performed during this reporting period")

    class Config:
        json_schema_extra = {
            "example": {
                "well_name": "78B-32",
                "report_date": "2021-07-15",
                "report_number": 19,
                "hole_depth_ft": 7842.0,
                "bit_depth_ft": 7842.0,
                "daily_footage_ft": 312.0,
                "rop_ft_per_hr": 13.0,
                "wob_klbs": 8.5,
                "rpm": 60.0,
                "flow_rate_gpm": 85.0,
                "standpipe_pressure_psi": 1450.0,
                "mud_weight_in_ppg": 8.8,
                "mud_weight_out_ppg": 8.9,
                "temp_in_f": 72.0,
                "temp_out_f": 148.0,
                "pit_volume_bbls": 520.0,
                "co2_pct": 0.3,
                "formation_name": "Mineral Mountains Granite",
                "inclination_deg": 65.2,
                "azimuth_deg": 225.0,
                "bit_size_in": 8.5,
            }
        }


class ReportChunkMetadata(BaseModel):
    """Metadata attached to every vector-store chunk for filtering."""
    source_file: str = Field(..., description="Original PDF filename")
    well_name: Optional[str] = None
    report_date: Optional[str] = None
    report_number: Optional[int] = None
    chunk_index: int = Field(..., description="Zero-based chunk index within the document")
    section: Optional[str] = Field(None, description="Section of report this chunk belongs to (e.g. 'Drilling Operations', 'Mud Report')")
