# backend/app/routers/segments.py
import pandas as pd
from fastapi import APIRouter

from backend.app.db import engine
from backend.app.schemas import SegmentProfile

router = APIRouter(prefix="/api/segments", tags=["segments"])


@router.get("", response_model=list[SegmentProfile])
def list_segments():
    df = pd.read_sql(
        "SELECT * FROM segment_profiles ORDER BY composite_score DESC", engine
    )
    return df.to_dict("records")
