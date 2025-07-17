#========================= Algo SARSA + Q-Learning ========================================================

# SARSA/  mise a jour des etat par: Q(S,A)<-Q(S,A)+alpha[R'+gammaQ(S',A')-Q(S,A)]
# Q(S,A): valeur de la paire etat action actuelle 
# Alpha : taux d'apprentissage compris entre (0,1)
# R' : recomprense recue apres avoir fait l'action A et etre ds l'etat S
# gamma : facteur qui determine l'importance des recompence future
# Q(S',A'): etat action suivante 
# etat terminal : Q(S',A')=0 -> Q(s,a) ← Q(s,a) + α[r + γ*0 - Q(s,a)]
# Q-LEARNING/  mise a jour des etat par: Q(S,A)<-Q(S,A)+alpha[R'+gamma max Q(S',A')-Q(S,A)]

import numpy as np
import random
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

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
#==============================================================================
# Alog SARSA + Q-learning 
#=============================================================================

def sarsa_q_learning(
    reinitialiser,
    faire_un_pas,
    obtenir_actions,
    episodes=100,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.1,
    etats_suivis=None,
    mode="sarsa",  # ← soit "sarsa", soit "q_learning"
    verbose=False
):
    Q = defaultdict(float)
    suivi_q = []

    for episode in range(episodes):
        if verbose:
            print(f"\n🎯 Épisode {episode+1}/{episodes}")

        etat = reinitialiser()
        action = None
        if mode == "sarsa":
            action = politique_epsilon_greedy(Q, etat, obtenir_actions(etat), epsilon, verbose)

        termine = False
        while not termine:
            if mode == "q_learning":
                action = politique_epsilon_greedy(Q, etat, obtenir_actions(etat), epsilon, verbose)

            etat_suiv, recompense, termine = faire_un_pas(action)

            if termine:
                Q[(etat, action)] += alpha * (recompense - Q[(etat, action)])
                if verbose:
                    print(f"🛑 État terminal atteint. Récompense = {recompense}")
            else:
                if mode == "sarsa":
                    action_suiv = politique_epsilon_greedy(Q, etat_suiv, obtenir_actions(etat_suiv), epsilon, verbose)
                    Q[(etat, action)] += alpha * (
                        recompense + gamma * Q[(etat_suiv, action_suiv)] - Q[(etat, action)]
                    )
                    etat, action = etat_suiv, action_suiv
                elif mode == "q_learning":
                    actions_suiv = obtenir_actions(etat_suiv)
                    max_q = max([Q[(etat_suiv, a)] for a in actions_suiv])
                    Q[(etat, action)] += alpha * (
                        recompense + gamma * max_q - Q[(etat, action)]
                    )
                    etat = etat_suiv

#=================== suivi =================
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


def tracer_q(df_q, etat, action):
    colonne = f"{etat}-{action}"
    if colonne not in df_q.columns:
        print(f" La Q-valeur {colonne} n'a pas été suivie.")
        return
    plt.plot(df_q["épisode"], df_q[colonne])
    plt.title(f"Évolution de Q({etat}, {action})")
    plt.xlabel("Épisode")
    plt.ylabel("Valeur Q")
    plt.grid()
    plt.show()