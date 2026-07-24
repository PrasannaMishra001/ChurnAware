# src/rl/dqn.py
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
                np.array(s2), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


def train_dqn(env, episodes=12000, gamma=0.97, lr=5e-4, batch_size=128,
              buffer_capacity=100000, warmup=1000, target_sync=1000,
              eps_start=1.0, eps_end=0.02, eps_decay_steps=120000,
              reward_scale=0.01, seed=42, log_every=1000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    q_net = QNetwork(env.STATE_DIM, env.N_ACTIONS)
    target_net = QNetwork(env.STATE_DIM, env.N_ACTIONS)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)

    step_count = 0
    history = []

    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            eps = max(eps_end, eps_start - (eps_start - eps_end) * step_count / eps_decay_steps)
            if random.random() < eps:
                action = random.randrange(env.N_ACTIONS)
            else:
                with torch.no_grad():
                    qvals = q_net(torch.from_numpy(state).unsqueeze(0))
                    action = int(qvals.argmax(dim=1).item())

            next_state, reward, done, info = env.step(action)
            buffer.push(state, action, reward * reward_scale, next_state, float(done))
            state = next_state
            ep_reward += reward
            step_count += 1

            if len(buffer) >= max(warmup, batch_size):
                s, a, r, s2, d = buffer.sample(batch_size)
                s_t = torch.from_numpy(s)
                a_t = torch.from_numpy(a).long().unsqueeze(1)
                r_t = torch.from_numpy(r)
                s2_t = torch.from_numpy(s2)
                d_t = torch.from_numpy(d)

                q_pred = q_net(s_t).gather(1, a_t).squeeze(1)
                with torch.no_grad():
                    next_actions = q_net(s2_t).argmax(dim=1, keepdim=True)
                    q_next = target_net(s2_t).gather(1, next_actions).squeeze(1)
                    q_target = r_t + gamma * (1 - d_t) * q_next

                loss = nn.functional.smooth_l1_loss(q_pred, q_target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 5.0)
                optimizer.step()

            if step_count % target_sync == 0:
                target_net.load_state_dict(q_net.state_dict())

        history.append(ep_reward)
        if (ep + 1) % log_every == 0:
            avg = np.mean(history[-log_every:])
            print(f"Episode {ep + 1}/{episodes} | avg reward (last {log_every}): {avg:,.1f} | eps: {eps:.3f}")

    return q_net, history


class DQNPolicy:
    def __init__(self, q_net):
        self.q_net = q_net
        self.q_net.eval()

    def act(self, state):
        with torch.no_grad():
            qvals = self.q_net(torch.from_numpy(state).unsqueeze(0))
            return int(qvals.argmax(dim=1).item())
