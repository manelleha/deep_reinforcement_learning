"""
Temporal‑Difference — SARSA & Q‑learning  avec  ε dégressif optionnel
---------------------------------------------------------------------

• Si `epsilon_decay` est None  →  ε reste constant.
• Sinon  →  trois plannings possibles : "exponential", "linear", "step".


"""

import numpy as np, random
from collections import defaultdict
import pandas as pd
from tqdm import trange, tqdm

# ─────────────────────────────────────────────────────────────
# ε‑greedy
# ─────────────────────────────────────────────────────────────
def politique_epsilon_greedy(Q, état, actions, epsilon, verbose=False):
    if random.random() < epsilon:
        action = random.choice(actions)
        if verbose:
            print(f"🔀 Exploration : {action}")
        return action
    valeurs = [Q[(état, a)] for a in actions]
    action = actions[int(np.argmax(valeurs))]
    if verbose:
        print(f"✅ Exploitation : {action}")
    return action


# ─────────────────────────────────────────────────────────────
# SARSA / Q‑learning
# ─────────────────────────────────────────────────────────────
def sarsa_q_learning(
    reinitialiser,
    faire_un_pas,
    obtenir_actions,
    episodes         = 100_000,
    alpha            = 0.1,
    gamma            = 0.99,
    epsilon          = 0.1,
    etats_suivis     = None,
    mode             = "q_learning",        # ou "sarsa"
    verbose          = False,
    Q_init           = None,
    # ---- nouveaux paramètres pour ε‑decay -------------------
    epsilon_decay    = None,                # ex. 0.995  (None = ε fixe)
    epsilon_min      = 0.01,
    epsilon_schedule = "exponential"        # "exponential" | "linear" | "step"
):
    Q = defaultdict(float) if Q_init is None else Q_init
    historique_loss, historique_eps = [], []
    suivi_q = []

    eps = float(epsilon)        # ε courant
    eps0 = eps                  # ε initial (pour le schedule linéaire)

    for episode in trange(episodes, desc="TD‑learning", leave=True):
        state = reinitialiser()
        action = politique_epsilon_greedy(Q, state,
                                          obtenir_actions(state), eps)

        loss_ep = []

        while True:
            s_next, reward, done = faire_un_pas(action)

            if done:
                cible = reward                       # Q(s',a') = 0
                td_error = cible - Q[(state, action)]
                loss = 0.5 * td_error**2
                Q[(state, action)] += alpha * td_error
                loss_ep.append(loss)
                break
            else:
                if mode == "sarsa":
                    a_next = politique_epsilon_greedy(Q, s_next,
                                                      obtenir_actions(s_next), eps)
                    cible = reward + gamma * Q[(s_next, a_next)]
                    td_error = cible - Q[(state, action)]
                    loss = 0.5 * td_error**2
                    Q[(state, action)] += alpha * td_error
                    loss_ep.append(loss)                 # ← ajouté
                    state, action = s_next, a_next

                elif mode == "q_learning":
                    max_q = max(Q[(s_next, a)] for a in obtenir_actions(s_next))
                    cible = reward + gamma * max_q
                    td_error = cible - Q[(state, action)]
                    loss = 0.5 * td_error**2
                    Q[(state, action)] += alpha * td_error
                    loss_ep.append(loss)
                    state = s_next
                    action = politique_epsilon_greedy(Q, state,
                                                      obtenir_actions(state), eps)

        if loss_ep:
            historique_loss.append(np.mean(loss_ep))
        historique_eps.append(eps)

        # ─ mise à jour ε ─────────────────────────────────────
        if epsilon_decay is not None:
            if epsilon_schedule == "exponential":
                eps = max(epsilon_min, eps * epsilon_decay)
            elif epsilon_schedule == "linear":
                step = (eps0 - epsilon_min) / episodes
                eps = max(epsilon_min, eps0 - step * (episode + 1))
            elif epsilon_schedule == "step":
                if (episode + 1) % (episodes // 4) == 0:
                    eps = max(epsilon_min, eps * 0.5)
        # ─────────────────────────────────────────────────────

        # suivi facultatif des Q
        if etats_suivis:
            snap = {f"{e}-{a}": Q[(e, a)]
                    for e in etats_suivis
                    for a in obtenir_actions(e)}
            snap["episode"] = episode
            snap["loss"] = historique_loss[-1]
            snap["epsilon"] = eps
            suivi_q.append(snap)

    df_q = pd.DataFrame(suivi_q)
    return Q, df_q, historique_loss, historique_eps


# ─────────────────────────────────────────────────────────────
# Helpers d’affichage
# ─────────────────────────────────────────────────────────────
def afficher_politique(Q, etats, actions, etats_terminaux):
    print("\n🧠 Politique greedy :")
    for s in etats:
        if s in etats_terminaux:
            print(f"État {s} : ⛔ terminal")
        else:
            a_opt = max(actions, key=lambda a: Q[(s, a)])
            nom = {0: "stay", 1: "switch", 2: "door C?"}.get(a_opt, a_opt)
            print(f"État {s} → action {a_opt} ({nom})  Q={Q[(s,a_opt)]:.3f}")


def tracer_loss(loss):
    import matplotlib.pyplot as plt
    plt.plot(loss); plt.title("Loss moyenne"); plt.xlabel("Episode"); plt.grid(); plt.show()
