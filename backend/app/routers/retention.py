# backend/app/routers/retention.py
import json
import os

import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.app.db import engine
from backend.app.schemas import ActionRecommendation
from backend.app.services import recommend_actions, action_catalog

router = APIRouter(prefix="/api/retention", tags=["retention"])

REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "..", "reports"
)


@router.get("/actions")
def list_actions():
    return action_catalog()


@router.get("/evaluation")
def policy_evaluation():
    path = os.path.normpath(os.path.join(REPORTS_DIR, "rl_evaluation.json"))
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Evaluation report not found")
    with open(path) as f:
        return json.load(f)


@router.get("/recommend/{customer_id}", response_model=ActionRecommendation)
def recommend(customer_id: int):
    df = pd.read_sql(
        "SELECT * FROM customers WHERE customer_id = :cid",
        engine, params={"cid": customer_id},
    )
    if df.empty:
        raise HTTPException(status_code=404, detail="Customer not found")

    action_ids, action_names, q_values = recommend_actions(df)
    catalog = action_catalog()
    return {
        "customer_id": customer_id,
        "churn_proba": float(df['churn_proba'].iloc[0]),
        "recommended_action_id": int(action_ids[0]),
        "recommended_action": action_names[0],
        "q_values": {
            catalog[str(i)]['name']: float(q_values[0][i])
            for i in range(q_values.shape[1])
        },
    }
