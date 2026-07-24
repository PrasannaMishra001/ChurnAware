# src/rl/evaluate.py
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import torch

from src.utils import MODELS_DIR, DATA_DIR, REPORTS_DIR
from src.rl.simulator import CustomerRetentionEnv, EPISODE_WEEKS, calibrate_from_data
from src.rl.dqn import QNetwork, DQNPolicy
from src.rl.policies import NeverActPolicy, AlwaysDiscountPolicy, RuleBasedPolicy

N_EPISODES = 5000


def run_policy(policy, calibration, pool, n_episodes=N_EPISODES, seed=123):
    env = CustomerRetentionEnv(calibration, pool, seed=seed)
    idx_rng = np.random.default_rng(seed)
    customer_ids = idx_rng.integers(0, len(pool), size=n_episodes)

    rewards, retained, orders, spend = [], [], [], []
    for i in range(n_episodes):
        state = env.reset(customer_idx=int(customer_ids[i]))
        total_reward = 0.0
        total_orders = 0
        total_spend = 0.0
        done = False
        while not done:
            action = policy.act(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            total_orders += int(info['ordered'])
            total_spend += info['incentive_spend']
        rewards.append(total_reward)
        retained.append(0 if env.churned else 1)
        orders.append(total_orders)
        spend.append(total_spend)

    return {
        'mean_net_profit': float(np.mean(rewards)),
        'median_net_profit': float(np.median(rewards)),
        'retention_rate': float(np.mean(retained)),
        'mean_orders': float(np.mean(orders)),
        'mean_incentive_spend': float(np.mean(spend)),
        'episodes': n_episodes,
    }


def main():
    print("RETENTION POLICY EVALUATION")

    features_csv = os.path.join(MODELS_DIR, "customer_feature_engineered_with_segments.csv")
    products_csv = os.path.join(DATA_DIR, "blinkit_products.csv")
    calibration, pool = calibrate_from_data(features_csv, products_csv)

    q_net = QNetwork(CustomerRetentionEnv.STATE_DIM, CustomerRetentionEnv.N_ACTIONS)
    q_net.load_state_dict(torch.load(os.path.join(MODELS_DIR, "retention_dqn.pt"), weights_only=True))

    policies = [
        NeverActPolicy(),
        AlwaysDiscountPolicy(),
        RuleBasedPolicy(),
        DQNPolicy(q_net),
    ]
    names = ['never_act', 'always_discount', 'rule_based', 'dqn']

    results = {}
    for name, policy in zip(names, policies):
        print(f"\nSimulating policy: {name} ({N_EPISODES} customers, {EPISODE_WEEKS} weeks)...")
        results[name] = run_policy(policy, calibration, pool)
        r = results[name]
        print(f"  Net profit/customer: Rs. {r['mean_net_profit']:,.2f}")
        print(f"  Retention rate:      {r['retention_rate']:.1%}")
        print(f"  Orders/customer:     {r['mean_orders']:.2f}")
        print(f"  Incentive spend:     Rs. {r['mean_incentive_spend']:,.2f}")

    baseline = results['never_act']['mean_net_profit']
    for name in names:
        results[name]['uplift_vs_never_act'] = float(
            results[name]['mean_net_profit'] - baseline
        )

    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, "rl_evaluation.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSUMMARY (net profit per customer over {EPISODE_WEEKS} weeks)")
    for name in names:
        r = results[name]
        print(f"  {name:16s} Rs. {r['mean_net_profit']:>10,.2f}   "
              f"retention {r['retention_rate']:.1%}   uplift {r['uplift_vs_never_act']:>+10,.2f}")
    print(f"\nSaved evaluation to: reports/rl_evaluation.json")


if __name__ == "__main__":
    main()
