import random
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import trange, tqdm
import matplotlib.pyplot as plt


def politique_epsilon_greedy(Q, état, actions, epsilon, verbose=False):
    """Politique ε-greedy pour la sélection d'actions"""
    if random.uniform(0, 1) < epsilon:
        action = random.choice(actions)
        if verbose:
            tqdm.write(f"🔀 Exploration : action aléatoire → {action}")
        return action
    else:
        valeurs_q = [Q[(état, a)] for a in actions]
        index_max = np.argmax(valeurs_q)
        action = actions[index_max]
        if verbose:
            tqdm.write(f"✅ Exploitation : meilleure action selon Q → {action}")
        return action

def dyna_q(
    reinitialiser,
    faire_un_pas,
    obtenir_actions,
    episodes=10000,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.1,
    planning_steps=1000,
    etats_suivis=None,
    verbose=False
):
    Q = defaultdict(float)    # Q(s, a)
    model = dict()            # Model(s, a) = (R, S')
    suivi_q = []
    loss_episode = []

    for episode in trange(episodes, desc="Entraînement Dyna-Q", leave=True):
        S = reinitialiser()
        episode_losses = []
        pas = 0

        while S is not None:
            actions = obtenir_actions(S)
            if not actions:
                break

            A = politique_epsilon_greedy(Q, S, actions, epsilon, verbose)

            S_prime, R, terminé = faire_un_pas(A)

            # Mise à jour Q-learning
            if terminé or S_prime is None:
                td_target = R
            else:
                actions_suivantes = obtenir_actions(S_prime)
                td_target = R + gamma * max(Q[(S_prime, a)] for a in actions_suivantes) if actions_suivantes else R

            td_error = td_target - Q[(S, A)]
            Q[(S, A)] += alpha * td_error

            loss = 0.5 * td_error**2
            episode_losses.append(loss)

            # Mise à jour du modèle
            model[(S, A)] = (R, S_prime)

            if verbose:
                tqdm.write(f"\n🎯 Épisode {episode + 1} - Pas {pas}")
                tqdm.write(f"🧠 État     : {S}")
                tqdm.write(f"🎮 Action   : {A}")
                tqdm.write(f"💰 Récompense : {R}")
                tqdm.write(f"📊 Q({S}, {A}) = {Q[(S, A)]:.4f}")
                tqdm.write(f"📉 TD Error  : {td_error:.4f}")
                tqdm.write(f"📉 Loss      : {loss:.4f}")
                tqdm.write("📦 Modèle actuel :")
                for (s_mod, a_mod), (r_mod, s_p_mod) in model.items():
                    tqdm.write(f"  ({s_mod}, {a_mod}) → (R={r_mod}, S'={s_p_mod})")
                tqdm.write("-" * 60)

            # Phase de planification
            for _ in range(planning_steps):
                S_sim, A_sim = random.choice(list(model.keys()))
                R_sim, S_prime_sim = model[(S_sim, A_sim)]
                actions_sim = obtenir_actions(S_prime_sim) if S_prime_sim is not None else []

                target_sim = R_sim + gamma * max(Q[(S_prime_sim, a)] for a in actions_sim) if actions_sim else R_sim
                Q[(S_sim, A_sim)] += alpha * (target_sim - Q[(S_sim, A_sim)])

            if terminé:
                break

            S = S_prime
            pas += 1

        moyenne_loss = np.mean(episode_losses) if episode_losses else 0
        loss_episode.append(moyenne_loss)

        if etats_suivis:
            snapshot = {
                f"{e}-{a}": Q[(e, a)]
                for e in etats_suivis
                for a in obtenir_actions(e)
                if obtenir_actions(e)
            }
            snapshot["épisode"] = episode
            snapshot["loss_moyenne"] = moyenne_loss
            suivi_q.append(snapshot)

    df_q = pd.DataFrame(suivi_q)
    return Q, df_q, loss_episode

def afficher_politique(Q, etats, actions, etats_terminaux):
    print("\n🧠 Politique apprise (greedy) :")
    for etat in etats:
        if etat in etats_terminaux:
            print(f"État {etat} (terminal) : ⛔")
        else:
            meilleure_action = max(actions, key=lambda a: Q[(etat, a)])
            symbole = "←" if meilleure_action == 0 else "→"
            print(f"État {etat} : action optimale = {meilleure_action} {symbole} (Q = {Q[(etat, meilleure_action)]:.2f})")




#========================================================================
# Évaluer la politique figée
#========================================================================
def tester_politique_figee(Q, reinitialiser, faire_un_pas, obtenir_actions, episodes=100, verbose=False):
    total_gain = 0

    for ep in range(episodes):
        etat = reinitialiser()
        gain = 0
        terminé = False
        pas = 0

        if verbose:
            print(f"\n🧪 Évaluation - Épisode {ep+1}")

        while not terminé and etat is not None:
            actions = obtenir_actions(etat)
            if not actions:
                break

            action = max(actions, key=lambda a: Q[(etat, a)])  # politique figée : greedy
            etat_suiv, recompense, terminé = faire_un_pas(action)
            gain += recompense

            if verbose:
                print(f"  Pas {pas} : état={etat}, action={action}, récompense={recompense}, état_suiv={etat_suiv}")
            
            etat = etat_suiv
            pas += 1

        total_gain += gain
        if verbose:
            print(f"🎯 Gain total épisode {ep+1} : {gain}")

    gain_moyen = total_gain / episodes
    print("\n📈 Évaluation de la stratégie figée :")
    print(f"🎯 Gain total sur {episodes} épisodes : {total_gain}")
    print(f"📊 Gain moyen par épisode           : {gain_moyen:.2f}")

    return gain_moyen
