# backend/app/main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.db import db_exists
from backend.app.routers import customers, segments, churn, retention, kpi


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not db_exists():
        from backend.app.ingest import run_ingest
        run_ingest()
    yield


app = FastAPI(
    title="ChurnAware API",
    description="Customer segmentation, churn prediction and retention action recommendation for the Blinkit e-grocery dataset.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(kpi.router)
app.include_router(customers.router)
app.include_router(segments.router)
app.include_router(churn.router)
app.include_router(retention.router)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "churnaware-api"}
