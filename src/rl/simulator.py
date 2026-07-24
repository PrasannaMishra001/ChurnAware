# src/rl/simulator.py
import json
import os
import numpy as np
import pandas as pd

ACTIONS = {
    0: {'name': 'no_action', 'hazard_mult': 1.00, 'flat_cost': 0.0, 'discount_rate': 0.00, 'fulfil_cost': 0.0},
    1: {'name': 'push_notification', 'hazard_mult': 1.15, 'flat_cost': 1.0, 'discount_rate': 0.00, 'fulfil_cost': 0.0},
    2: {'name': 'free_delivery', 'hazard_mult': 1.40, 'flat_cost': 0.0, 'discount_rate': 0.00, 'fulfil_cost': 30.0},
    3: {'name': 'discount_10pct', 'hazard_mult': 2.00, 'flat_cost': 0.0, 'discount_rate': 0.10, 'fulfil_cost': 0.0},
}

EPISODE_WEEKS = 39
CHURN_WEEKS = 13
MAX_START_RECENCY_WEEKS = 8.0
RECENCY_DECAY = 0.10
SENTIMENT_EFFECT = 0.25
FATIGUE_GAIN = 0.30
FATIGUE_DECAY = 0.90
DELAY_SENTIMENT_HIT = 0.12
NEG_FEEDBACK_SENTIMENT_HIT = 0.25
GOOD_ORDER_SENTIMENT_GAIN = 0.04


def calibrate_from_data(features_csv, products_csv):
    df = pd.read_csv(features_csv)
    products = pd.read_csv(products_csv)

    df = df[df['frequency'] > 0].copy()
    df['registration_date'] = pd.to_datetime(df['registration_date'], errors='coerce')
    end_date = df['registration_date'].max() + pd.Timedelta(days=1)
    tenure_weeks = ((end_date - df['registration_date']).dt.days / 7.0).clip(lower=4)
    df['weekly_hazard'] = (df['frequency'] / tenure_weeks).clip(0.01, 0.8)

    margin_rate = float(products['margin_percentage'].mean()) / 100.0

    segments = {}
    for name, g in df.groupby('segment_name'):
        aov = g.loc[g['avg_order_value'] > 0, 'avg_order_value']
        log_aov = np.log(aov.clip(lower=1))
        segments[name] = {
            'weekly_hazard': float(g['weekly_hazard'].mean()),
            'aov_log_mu': float(log_aov.mean()),
            'aov_log_sigma': float(max(log_aov.std(), 0.05)),
            'delay_prob': float((1 - g['on_time_ratio']).mean()),
            'neg_feedback_prob': float((g['negative_feedback_count'] / g['frequency']).clip(0, 1).mean()),
            'size': int(len(g)),
        }

    customer_pool = df[[
        'customer_id', 'segment_name', 'recency_days', 'frequency', 'monetary',
        'avg_sentiment', 'avg_text_sentiment', 'on_time_ratio', 'avg_order_value'
    ]].reset_index(drop=True)

    return {'segments': segments, 'margin_rate': margin_rate}, customer_pool


class CustomerRetentionEnv:
    STATE_DIM = 8
    N_ACTIONS = len(ACTIONS)

    def __init__(self, calibration, customer_pool, seed=42):
        self.calib = calibration
        self.pool = customer_pool
        self.rng = np.random.default_rng(seed)
        self.margin_rate = calibration['margin_rate']

    def _get_state(self):
        return np.array([
            self.recency_weeks / CHURN_WEEKS,
            np.log1p(self.frequency) / 3.0,
            np.log1p(self.monetary) / 10.0,
            self.sentiment,
            self.on_time_ratio,
            self.week / EPISODE_WEEKS,
            self.fatigue,
            self.last_action / (self.N_ACTIONS - 1),
        ], dtype=np.float32)

    def reset(self, customer_idx=None):
        if customer_idx is None:
            customer_idx = int(self.rng.integers(0, len(self.pool)))
        row = self.pool.iloc[customer_idx]

        self.segment = row['segment_name']
        self.seg = self.calib['segments'][self.segment]
        self.recency_weeks = min(float(row['recency_days']) / 7.0, MAX_START_RECENCY_WEEKS)
        self.frequency = float(row['frequency'])
        self.monetary = float(row['monetary'])
        self.sentiment = float(np.clip(row['avg_text_sentiment'], -1, 1))
        self.on_time_ratio = float(row['on_time_ratio'])
        self.week = 0
        self.fatigue = 0.0
        self.last_action = 0
        self.done = False
        self.churned = False
        return self._get_state()

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode finished; call reset().")

        spec = ACTIONS[int(action)]
        reward = -spec['flat_cost']
        incentive_spend = spec['flat_cost']

        boost = 1.0 + (spec['hazard_mult'] - 1.0) * (1.0 - self.fatigue)
        if action != 0:
            self.fatigue = min(1.0, self.fatigue + FATIGUE_GAIN)

        base = self.seg['weekly_hazard']
        recency_factor = np.exp(-RECENCY_DECAY * self.recency_weeks)
        sentiment_factor = 1.0 + SENTIMENT_EFFECT * self.sentiment
        p_order = float(np.clip(base * recency_factor * sentiment_factor * boost, 0.0, 0.95))

        ordered = self.rng.random() < p_order
        order_value = 0.0

        if ordered:
            order_value = float(np.exp(self.rng.normal(self.seg['aov_log_mu'], self.seg['aov_log_sigma'])))
            paid = order_value * (1.0 - spec['discount_rate'])
            cogs = order_value * (1.0 - self.margin_rate)
            profit = paid - cogs - spec['fulfil_cost']
            reward += profit
            incentive_spend += order_value * spec['discount_rate'] + spec['fulfil_cost']

            self.frequency += 1
            self.monetary += order_value
            self.recency_weeks = 0.0

            delayed = self.rng.random() < self.seg['delay_prob']
            self.on_time_ratio = 0.9 * self.on_time_ratio + 0.1 * (0.0 if delayed else 1.0)
            if delayed:
                self.sentiment -= DELAY_SENTIMENT_HIT
            else:
                self.sentiment += GOOD_ORDER_SENTIMENT_GAIN
            if self.rng.random() < self.seg['neg_feedback_prob']:
                self.sentiment -= NEG_FEEDBACK_SENTIMENT_HIT
            self.sentiment = float(np.clip(self.sentiment, -1, 1))
        else:
            self.recency_weeks += 1.0

        self.fatigue *= FATIGUE_DECAY
        self.last_action = int(action)
        self.week += 1

        if self.recency_weeks >= CHURN_WEEKS:
            self.done = True
            self.churned = True
        elif self.week >= EPISODE_WEEKS:
            self.done = True

        info = {
            'ordered': ordered,
            'order_value': order_value,
            'incentive_spend': incentive_spend,
            'churned': self.churned,
            'p_order': p_order,
        }
        return self._get_state(), float(reward), self.done, info


def save_calibration(calibration, path):
    with open(path, 'w') as f:
        json.dump(calibration, f, indent=2)


def load_calibration(path):
    with open(path) as f:
        return json.load(f)
