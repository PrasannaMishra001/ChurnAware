# src/rl/policies.py
from .simulator import CHURN_WEEKS


class NeverActPolicy:
    name = 'never_act'

    def act(self, state):
        return 0


class AlwaysDiscountPolicy:
    name = 'always_discount'

    def act(self, state):
        return 3


class RuleBasedPolicy:
    name = 'rule_based'

    def __init__(self, notify_weeks=3, discount_weeks=6):
        self.notify_frac = notify_weeks / CHURN_WEEKS
        self.discount_frac = discount_weeks / CHURN_WEEKS

    def act(self, state):
        recency_frac = state[0]
        if recency_frac >= self.discount_frac:
            return 3
        if recency_frac >= self.notify_frac:
            return 1
        return 0
