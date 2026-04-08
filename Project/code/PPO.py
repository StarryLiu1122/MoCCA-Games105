import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        mu = self.net(x)
        std = torch.exp(self.log_std)
        return mu, std

class Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)

class PPOPolicy:
    def __init__(self, state_dim, action_dim, gamma=0.99, clip_eps=0.2, lr=3e-4):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.buffer = []

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        mu, std = self.actor(state)
        dist = torch.distributions.Normal(mu, std)
        action = mu if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.squeeze(0).detach().numpy(), log_prob.item()

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.buffer.append((state, action, reward, next_state, done, log_prob))

    def update(self):
        states, actions, rewards, next_states, dones, log_probs_old = zip(*self.buffer)
        self.buffer.clear()

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)
        old_log_probs = torch.tensor(np.array(log_probs_old), dtype=torch.float32)

        values = self.critic(states)
        with torch.no_grad():
            next_values = self.critic(torch.tensor(next_states, dtype=torch.float32))
            targets = rewards + self.gamma * next_values * (1 - dones)

        advantages = targets - values

        mu, std = self.actor(states)
        dist = torch.distributions.Normal(mu, std)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        ratio = (new_log_probs - old_log_probs).exp()
        clipped_adv = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        actor_loss = -torch.min(ratio * advantages, clipped_adv).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optim.step()

        value_loss = nn.functional.mse_loss(values, targets)
        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()
