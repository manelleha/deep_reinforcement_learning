import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
import os

# ============================================================================
# Politique ε-greedy (on-policy)
# ============================================================================
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

# ============================================================================
# Monte Carlo On-Policy (First-Visit)
# ============================================================================
def monte_carlo_on_policy(
    reinitialiser,
    faire_un_pas,
    etats,
    actions,
    etats_terminaux,
    gamma=1.0,
    episodes=5000,
    epsilon=0.1,
    verbose=False
):
    Q = defaultdict(float)
    returns = defaultdict(list)
    historique_loss = []

    for ep in range(episodes):
        reinitialiser()
        etat = etats[len(etats) // 2]
        episode = []
        action = politique_epsilon_greedy(Q, etat, actions, epsilon)

        while True:
            etat_suiv, r, termine = faire_un_pas(action)
            episode.append((etat, action, r))
            if termine:
                break
            etat = etat_suiv
            action = politique_epsilon_greedy(Q, etat, actions, epsilon)

        # First-visit Monte Carlo
        G = 0
        deja_vu = set()
        loss_ep = []

        for t in reversed(range(len(episode))):
            etat, action, r = episode[t]
            G = gamma * G + r
            if (etat, action) not in deja_vu:
                returns[(etat, action)].append(G)
                ancien = Q[(etat, action)]
                Q[(etat, action)] = np.mean(returns[(etat, action)])
                loss_ep.append((G - ancien) ** 2)
                deja_vu.add((etat, action))

        if loss_ep:
            historique_loss.append(np.mean(loss_ep))

        if verbose and (ep + 1) % 100 == 0:
            print(f"✅ Épisode {ep + 1} terminé – longueur : {len(episode)}")

    return Q, historique_loss

# ============================================================================
# Affichage de la politique greedy
# ============================================================================
def afficher_politique(Q, etats, actions, etats_terminaux):
    print("\n🧭 Politique apprise (greedy) :")
    for etat in etats:
        if etat in etats_terminaux:
            print(f"État {etat} (terminal) : ⛔")
        else:
            meilleure_action = max(actions, key=lambda a: Q[(etat, a)])
            symbole = "←" if meilleure_action == 0 else "→"
            print(f"État {etat} : action = {meilleure_action} {symbole} (Q = {Q[(etat, meilleure_action)]:.2f})")

# ============================================================================
# Tracer la loss
# ============================================================================
def tracer_loss(historique_loss):
    plt.plot(historique_loss)
    plt.title("Loss moyenne par épisode")
    plt.xlabel("Épisode")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

# ============================================================================
# Sauvegarder les résultats Monte Carlo
# ============================================================================
def sauvegarder_resultats_mc(Q, etats, actions, dossier="sauvegarde/monte_carlo_on_policy"):
    if not os.path.exists(dossier):
        os.makedirs(dossier)

    with open(os.path.join(dossier, "Q_values.pkl"), "wb") as f:
        pickle.dump(dict(Q), f)

    policy = {}
    for s in etats:
        if all((s, a) not in Q for a in actions):
            continue
        best_a = max(actions, key=lambda a: Q.get((s, a), 0.0))
        policy[s] = best_a

    with open(os.path.join(dossier, "policy.pkl"), "wb") as f:
        pickle.dump(policy, f)

    V = {}
    for s in etats:
        if all((s, a) not in Q for a in actions):
            continue
        V[s] = max(Q.get((s, a), 0.0) for a in actions)

    with open(os.path.join(dossier, "value_function.pkl"), "wb") as f:
        pickle.dump(V, f)

    print(f"✅ Résultats Monte Carlo sauvegardés dans : {dossier}")
