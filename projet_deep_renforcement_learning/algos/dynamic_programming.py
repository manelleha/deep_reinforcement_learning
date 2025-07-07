import nbformat
from IPython import get_ipython
import numpy as np
import time
from typing import List
##======================================== Iterative policy evaluation =========================================

#  formule a utiliser : V(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V(s')]
# V(s) : valeur de l’état s
# γ : facteur de discount permet de dire si les gain future son important ou moins important que les gain imediat (ex. 0.9)
# P(s'|s,a) : probabilité d’aller de s à s' avec l’action a
# R(s,a,s') : récompense obtenue

def iterative_policy_evaluation(
    pi: np.ndarray,
    S: List[int],
    A: List[int],
    R: List[float],
    T: List[int], # terminal states
    p: np.ndarray,
    theta: float = 0.00001,
    gamma: float = 0.9999999,
):
    V = np.random.random(len(S))
    V[T] = 0.0
    iteration = 0

    while True:
        delta = 0.0
        print(f"\n Itération {iteration}")
        for s in S:
            if s in T:
                print(f"V({s}) = 0 (état terminal)")
                continue

            print(f"\n État {s} :")
            v = V[s]
            new_v = 0.0

            for a in A:
                action_prob = pi[s, a]
                if action_prob == 0:
                    continue

                print(f"   Action {a} (proba {action_prob}) :")
                subtotal = 0.0
                for s_p in S:
                    for r_index in range(len(R)):
                        prob = p[s, a, s_p, r_index]
                        if prob > 0:
                            r = R[r_index]
                            print(f" s’ = {s_p}, r = {r}, V(s’) = {V[s_p]:.4f}, gamma = {gamma}")
                            subtotal += prob * (r + gamma * V[s_p])

                weighted = action_prob * subtotal
                print(f" -> Contribution action {a} = {weighted:.6f}")
                new_v += weighted

            print(f" Mise à jour : V({s}) = {new_v:.6f}")
            V[s] = new_v
            delta = max(delta, abs(v - V[s]))

        print(f"\n max(delta) = {delta:.8f}")
        if delta < theta:
            print("\n Convergence atteinte.")
            break
        iteration += 1

    return V

##====================================== Polyci =============================================

