import random
from collections import defaultdict
import pandas as pd

from algos.politique import politique_epsilon_greedy  # ε-greedy(S, Q)


############## DYNA-Q===============
def dyna_q(
    reinitialiser,
    faire_un_pas,
    obtenir_actions,
    episodes=100,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.1,
    planning_steps=10,
    etats_suivis=None,
    verbose=False
):
    Q = defaultdict(float)    # Q(s, a)
    model = dict()            # Model(s, a) = (R, S')
    suivi_q = []

    for episode in range(episodes):

        S = reinitialiser()   # (a) S ← état actuel
        terminé = False

        while not terminé:
            # (b) A ← ε-greedy(S, Q)
            actions = obtenir_actions(S)
            A = politique_epsilon_greedy(Q, S, actions, epsilon, verbose)

            # (c) Take action A; observe reward and next state
            S_prime, R, terminé = faire_un_pas(A)

            # (d) Q-learning update
            if terminé:
                Q[(S, A)] += alpha * (R - Q[(S, A)])
            else:
                Q[(S, A)] += alpha * (
                    R + gamma * max(Q[(S_prime, a)] for a in obtenir_actions(S_prime)) - Q[(S, A)]
                )

            # (e) Model update
            model[(S, A)] = (R, S_prime)

            # (f) Planning step
            for _ in range(planning_steps):
                S_sim, A_sim = random.choice(list(model.keys()))
                R_sim, S_prime_sim = model[(S_sim, A_sim)]

                if obtenir_actions(S_prime_sim):
                    Q[(S_sim, A_sim)] += alpha * (
                        R_sim + gamma * max(Q[(S_prime_sim, a)] for a in obtenir_actions(S_prime_sim)) - Q[(S_sim, A_sim)]
                    )
                else:
                    Q[(S_sim, A_sim)] += alpha * (R_sim - Q[(S_sim, A_sim)])

            S = S_prime

        # Enregistrement du suivi
        if etats_suivis:
            snapshot = {
                f"{e}-{a}": Q[(e, a)]
                for e in etats_suivis
                for a in obtenir_actions(e)
            }
            snapshot["épisode"] = episode
            suivi_q.append(snapshot)

    df_q = pd.DataFrame(suivi_q)
    return Q, df_q
