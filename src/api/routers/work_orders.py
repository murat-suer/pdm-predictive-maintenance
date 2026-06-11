"""Work order listing endpoint."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.routers.common import as_utc
from src.api.schemas import WorkOrderItem
from src.database.models import WorkOrder

router = APIRouter(prefix="/work-orders", tags=["work-orders"])


@router.get("", response_model=list[WorkOrderItem])
async def list_work_orders(
    machine_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """List work orders, newest first, optionally filtered."""
    query = db.query(WorkOrder).order_by(WorkOrder.created_at.desc())
    if machine_id:
        query = query.filter(WorkOrder.machine_id == machine_id)
    if status:
        query = query.filter(WorkOrder.status == status.upper())

    return [
        WorkOrderItem(
            id=row.id,
            work_order_number=row.work_order_number,
            machine_id=row.machine_id,
            fault_type=row.fault_type,
            recommended_action=row.recommended_action,
            priority=row.priority,
            status=row.status,
            estimated_cost_eur=row.estimated_cost_eur,
            created_at=as_utc(row.created_at),
        )
        for row in query.limit(limit).all()
    ]
