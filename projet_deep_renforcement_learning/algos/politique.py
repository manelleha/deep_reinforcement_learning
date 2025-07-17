
import random
import numpy as np

# #========================================================================
# # Politique ε-greedy 
# #========================================================================
def politique_epsilon_greedy(Q, état, actions, epsilon, verbose=False):
    if random.uniform(0, 1) < epsilon:
        action = random.choice(actions)
        if verbose:
            print(f"🔀 Exploration : action aléatoire → {action}")
        return action
    else:
        valeurs_q = [Q[(état, a)] for a in actions]
        index_max = np.argmax(valeurs_q)
        action = actions[index_max]
        if verbose:
            print(f"✅ Exploitation : meilleure action selon Q → {action}")
        return action