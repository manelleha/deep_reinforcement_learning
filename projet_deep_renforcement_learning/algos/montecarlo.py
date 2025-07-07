import random
from collections import defaultdict

def on_policy_mc_control(env, episodes=1000, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: defaultdict(float))        # Q[s][a] = valeur estimée
    returns = defaultdict(lambda: defaultdict(list))   # returns[s][a] = liste des G
    policy = {}                                        # policy[s] = meilleure action

    for _ in range(episodes):
        state = env.reset()
        episode = []
        done = False

        while not done:
            actions = env.get_actions(state)
            if random.random() < epsilon or state not in policy:
                action = random.choice(actions)
            else:
                q_vals = Q[state]
                action = max(q_vals, key=q_vals.get)
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        visited = set()
        G = 0
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r
            if (s, a) not in visited:
                visited.add((s, a))
                returns[s][a].append(G)
                Q[s][a] = sum(returns[s][a]) / len(returns[s][a])
                policy[s] = max(Q[s], key=Q[s].get)

    return policy, Q
