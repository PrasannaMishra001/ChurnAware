# backend/app/services.py
import json
import os
import sys
from functools import lru_cache

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import joblib
import numpy as np
import pandas as pd
import torch

from src.utils import MODELS_DIR
from src.rl.simulator import CHURN_WEEKS
from src.rl.dqn import QNetwork

STATE_DIM = 8
N_ACTIONS = 4


@lru_cache(maxsize=1)
def get_churn_model():
    model = joblib.load(os.path.join(MODELS_DIR, "churn_prediction_model.pkl"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "churn_feature_cols.joblib"))
    return model, feature_cols


@lru_cache(maxsize=1)
def get_policy():
    q_net = QNetwork(STATE_DIM, N_ACTIONS)
    q_net.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, "retention_dqn.pt"), weights_only=True)
    )
    q_net.eval()
    with open(os.path.join(MODELS_DIR, "retention_policy_meta.json")) as f:
        meta = json.load(f)
    return q_net, meta


@lru_cache(maxsize=1)
def get_churn_metrics():
    with open(os.path.join(MODELS_DIR, "churn_model_metrics.json")) as f:
        return json.load(f)


def score_churn(df):
    model, feature_cols = get_churn_model()
    X = df.reindex(columns=feature_cols, fill_value=0).fillna(0)
    return model.predict_proba(X)[:, 1]


def build_policy_states(df):
    states = np.column_stack([
        np.clip(df['recency_days'].values / 7.0, 0, CHURN_WEEKS - 1) / CHURN_WEEKS,
        np.log1p(df['frequency'].values) / 3.0,
        np.log1p(df['monetary'].values) / 10.0,
        np.clip(df['avg_text_sentiment'].values, -1, 1),
        df['on_time_ratio'].values,
        np.zeros(len(df)),
        np.zeros(len(df)),
        np.zeros(len(df)),
    ]).astype(np.float32)
    return states


def recommend_actions(df):
    q_net, meta = get_policy()
    states = build_policy_states(df)
    with torch.no_grad():
        q_values = q_net(torch.from_numpy(states)).numpy()
    action_ids = q_values.argmax(axis=1)
    action_names = [meta['actions'][str(a)]['name'] for a in action_ids]
    return action_ids, action_names, q_values


def action_catalog():
    _, meta = get_policy()
    return meta['actions']
