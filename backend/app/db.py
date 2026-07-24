# backend/app/db.py
import os
from sqlalchemy import create_engine

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(ROOT, "models", "churnaware.db")

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
)


def db_exists():
    return os.path.exists(DB_PATH)
