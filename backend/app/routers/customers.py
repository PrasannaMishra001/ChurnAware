# backend/app/routers/customers.py
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from backend.app.db import engine
from backend.app.schemas import CustomerDetail, CustomerPage

router = APIRouter(prefix="/api/customers", tags=["customers"])


@router.get("", response_model=CustomerPage)
def list_customers(
    segment: Optional[str] = None,
    min_frequency: float = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    where = "WHERE frequency >= :min_freq"
    params = {"min_freq": min_frequency, "limit": limit, "offset": offset}
    if segment:
        where += " AND segment_name = :segment"
        params["segment"] = segment

    total = pd.read_sql(
        f"SELECT COUNT(*) AS n FROM customers {where}", engine, params=params
    )["n"].iloc[0]

    items = pd.read_sql(
        f"""SELECT * FROM customers {where}
            ORDER BY monetary DESC LIMIT :limit OFFSET :offset""",
        engine, params=params,
    )
    return {
        "total": int(total),
        "limit": limit,
        "offset": offset,
        "items": items.to_dict("records"),
    }


@router.get("/{customer_id}", response_model=CustomerDetail)
def get_customer(customer_id: int):
    df = pd.read_sql(
        "SELECT * FROM customers WHERE customer_id = :cid",
        engine, params={"cid": customer_id},
    )
    if df.empty:
        raise HTTPException(status_code=404, detail="Customer not found")
    return df.iloc[0].to_dict()
