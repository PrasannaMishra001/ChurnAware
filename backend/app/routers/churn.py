# backend/app/routers/churn.py
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Query

from backend.app.db import engine
from backend.app.schemas import CustomerSummary
from backend.app.services import get_churn_metrics

router = APIRouter(prefix="/api/churn", tags=["churn"])


@router.get("/top-risk", response_model=list[CustomerSummary])
def top_risk(
    limit: int = Query(100, ge=1, le=500),
    segment: Optional[str] = None,
):
    where = "WHERE frequency > 0"
    params = {"limit": limit}
    if segment:
        where += " AND segment_name = :segment"
        params["segment"] = segment

    df = pd.read_sql(
        f"""SELECT * FROM customers {where}
            ORDER BY churn_proba DESC LIMIT :limit""",
        engine, params=params,
    )
    return df.to_dict("records")


@router.get("/metrics")
def churn_metrics():
    return get_churn_metrics()
