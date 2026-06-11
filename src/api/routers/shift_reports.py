"""Shift report listing endpoint."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.routers.common import as_utc
from src.api.schemas import ShiftReportItem
from src.database.models import ShiftReport

router = APIRouter(prefix="/shift-reports", tags=["shift-reports"])


@router.get("", response_model=list[ShiftReportItem])
async def list_shift_reports(
    limit: int = Query(default=4, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Most recent shift reports, newest first."""
    rows = (
        db.query(ShiftReport)
        .order_by(ShiftReport.shift_start.desc())
        .limit(limit)
        .all()
    )
    return [
        ShiftReportItem(
            id=row.id,
            shift_type=row.shift_type,
            shift_start=as_utc(row.shift_start),
            shift_end=as_utc(row.shift_end),
            generated_at=as_utc(row.generated_at),
            report_data=row.report_data or {},
        )
        for row in rows
    ]
