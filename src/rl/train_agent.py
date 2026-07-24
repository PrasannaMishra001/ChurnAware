# src/rl/train_agent.py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch

from src.utils import MODELS_DIR, DATA_DIR
from src.rl.simulator import (
    CustomerRetentionEnv, ACTIONS, EPISODE_WEEKS, CHURN_WEEKS,
    calibrate_from_data, save_calibration
)
from src.rl.dqn import train_dqn


def main():
    print("RETENTION POLICY TRAINING (DQN)")

    features_csv = os.path.join(MODELS_DIR, "customer_feature_engineered_with_segments.csv")
    products_csv = os.path.join(DATA_DIR, "blinkit_products.csv")

    print("\n[1/3] Calibrating simulator from Blinkit data...")
    calibration, pool = calibrate_from_data(features_csv, products_csv)
    print(f"Margin rate: {calibration['margin_rate']:.1%}")
    for name, seg in calibration['segments'].items():
        print(f"  {name}: hazard={seg['weekly_hazard']:.3f}/wk, "
              f"delay_prob={seg['delay_prob']:.2f}, neg_fb_prob={seg['neg_feedback_prob']:.2f}, "
              f"n={seg['size']}")

    meta = {
        'calibration': calibration,
        'actions': {str(k): v for k, v in ACTIONS.items()},
        'episode_weeks': EPISODE_WEEKS,
        'churn_weeks': CHURN_WEEKS,
        'state_features': [
            'recency_weeks_norm', 'log_frequency_norm', 'log_monetary_norm',
            'sentiment', 'on_time_ratio', 'week_frac', 'fatigue', 'last_action_norm'
        ],
    }
    save_calibration(meta, os.path.join(MODELS_DIR, "retention_policy_meta.json"))
    print("Saved calibration to: models/retention_policy_meta.json")

    print("\n[2/3] Training DQN agent...")
    env = CustomerRetentionEnv(calibration, pool, seed=42)
    q_net, history = train_dqn(env, episodes=12000, seed=42)

    print("\n[3/3] Saving policy...")
    torch.save(q_net.state_dict(), os.path.join(MODELS_DIR, "retention_dqn.pt"))
    print("Saved policy to: models/retention_dqn.pt")
    print("\nRun 'python -m src.rl.evaluate' to compare against baseline policies.")


if __name__ == "__main__":
    main()
