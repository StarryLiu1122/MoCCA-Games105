from rl_env import CharacterWalkEnv
from PPO import PPOPolicy
from viewer.viewer_new import SimpleViewer

viewer = SimpleViewer(substep = 32, simu_flag=1)
env = CharacterWalkEnv(viewer)
obs_dim = env.get_obs().shape[0]     # state dim = 83
action_dim = env.action_dim          # action dim = 60 每个关节的力矩
policy = PPOPolicy(state_dim=obs_dim, action_dim=action_dim)

for epoch in range(1000):
    state = env.reset()
    ep_reward = 0
    while True:
        action, log_prob = policy.select_action(state)
        next_state, reward, done, _ = env.step(action)
        policy.store_transition(state, action, reward, next_state, done, log_prob)
        state = next_state
        ep_reward += reward
        if done:
            break
    policy.update()
    print(f"[Epoch {epoch}] reward: {ep_reward:.5f}")