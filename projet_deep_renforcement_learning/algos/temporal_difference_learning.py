#========================= Algo SARSA + Q-Learning ========================================================

# SARSA/  mise a jour des etat par: Q(S,A)<-Q(S,A)+alpha[R'+gammaQ(S',A')-Q(S,A)]
# Q(S,A): valeur de la paire etat action actuelle 
# Alpha : taux d'apprentissage compris entre (0,1)
# R' : recompense recue apres avoir fait l'action A et etre dans l'etat S
# gamma : facteur qui determine l'importance des recompenses futures
# Q(S',A'): etat action suivante 
# etat terminal : Q(S',A')=0 -> Q(s,a) ← Q(s,a) + α[r + γ*0 - Q(s,a)]
import numpy as np
import random
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import pickle
import os


#========================================================================
# Politique ε-greedy 
#========================================================================
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

#========================================================================
# Algo SARSA + Q-learning avec suivi, loss et stratégie
#========================================================================
def sarsa_q_learning(
    reinitialiser,
    faire_un_pas,
    obtenir_actions,
    episodes=100,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.1,
    etats_suivis=None,
    mode="sarsa",
    verbose=False
):
    Q = defaultdict(float)
    suivi_q = []
    historique_loss = []

    for episode in trange(episodes, desc="Entraînement", leave=True):
        etat = reinitialiser()
        action = politique_epsilon_greedy(Q, etat, obtenir_actions(etat), epsilon)
        
        loss_episode = []
        
        while True:
            etat_suiv, recompense, termine = faire_un_pas(action)
            
            if termine:
                # État terminal : Q(s',a') = 0
                cible = recompense
                td_error = cible - Q[(etat, action)]
                loss = 0.5 * td_error**2
                Q[(etat, action)] += alpha * td_error
                
                if verbose:
                    tqdm.write(f"🎯 Épisode {episode + 1} (Terminal)")
                    tqdm.write(f"🧠 État     : {etat}")
                    tqdm.write(f"🎮 Action   : {action}")
                    tqdm.write(f"💰 Récompense : {recompense}")
                    tqdm.write(f"📊 Q({etat},{action}) = {Q[(etat, action)]:.4f}")
                    tqdm.write(f"📉 TD Error  : {td_error:.4f}")
                    tqdm.write(f"📉 Loss      : {loss:.4f}")
                    tqdm.write("-" * 30)
                
                loss_episode.append(loss)
                break
            
            else:
                if mode == "sarsa":
                    # SARSA : utilise la prochaine action selon la politique
                    action_suiv = politique_epsilon_greedy(Q, etat_suiv, obtenir_actions(etat_suiv), epsilon)
                    cible = recompense + gamma * Q[(etat_suiv, action_suiv)]
                    td_error = cible - Q[(etat, action)]
                    loss = 0.5 * td_error**2
                    Q[(etat, action)] += alpha * td_error
                    
                    # Transition pour le prochain pas
                    etat, action = etat_suiv, action_suiv
                    
                elif mode == "q_learning":
                    # Q-Learning : utilise le maximum des Q-valeurs
                    actions_suiv = obtenir_actions(etat_suiv)
                    max_q = max(Q[(etat_suiv, a)] for a in actions_suiv)
                    cible = recompense + gamma * max_q
                    td_error = cible - Q[(etat, action)]
                    loss = 0.5 * td_error**2
                    Q[(etat, action)] += alpha * td_error
                    
                    # Transition pour le prochain pas
                    etat = etat_suiv
                    action = politique_epsilon_greedy(Q, etat, obtenir_actions(etat), epsilon)
                
                if verbose:
                    tqdm.write(f"🎯 Épisode {episode + 1}")
                    tqdm.write(f"🧠 État     : {etat}")
                    tqdm.write(f"🎮 Action   : {action}")
                    tqdm.write(f"💰 Récompense : {recompense}")
                    tqdm.write(f"📊 Q({etat},{action}) = {Q[(etat, action)]:.4f}")
                    tqdm.write(f"📉 TD Error  : {td_error:.4f}")
                    tqdm.write(f"📉 Loss      : {loss:.4f}")
                    tqdm.write("-" * 30)
                
                loss_episode.append(loss)

        # Sauvegarder la loss moyenne de l'épisode
        if loss_episode:
            historique_loss.append(np.mean(loss_episode))

        # suivi facultatif des Q-valeurs
        if etats_suivis:
            snapshot = {
                f"{e}-{a}": Q[(e, a)]
                for e in etats_suivis
                for a in obtenir_actions(e)
            }
            snapshot["épisode"] = episode
            snapshot["loss_moyenne"] = historique_loss[-1] if historique_loss else 0
            suivi_q.append(snapshot)

    df_q = pd.DataFrame(suivi_q)
    return Q, df_q, historique_loss

#========================================================================
# Affichage de la stratégie (politique greedy)
#========================================================================
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
# Tracer l'évolution d'une Q-valeur et de la loss
#========================================================================
def tracer_q(df_q, etat, action):
    colonne = f"{etat}-{action}"
    if colonne not in df_q.columns:
        print(f" La Q-valeur {colonne} n'a pas été suivie.")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Évolution de la Q-valeur
    ax1.plot(df_q["épisode"], df_q[colonne])
    ax1.set_title(f"Évolution de Q({etat}, {action})")
    ax1.set_xlabel("Épisode")
    ax1.set_ylabel("Valeur Q")
    ax1.grid()
    
    # Évolution de la loss
    if "loss_moyenne" in df_q.columns:
        ax2.plot(df_q["épisode"], df_q["loss_moyenne"])
        ax2.set_title("Évolution de la Loss moyenne")
        ax2.set_xlabel("Épisode")
        ax2.set_ylabel("Loss")
        ax2.grid()
    
    plt.tight_layout()
    plt.show()

def tracer_loss(historique_loss):
    """Tracer l'évolution de la loss au fil des épisodes"""
    plt.figure(figsize=(10, 6))
    plt.plot(historique_loss)
    plt.title("Évolution de la Loss moyenne par épisode")
    plt.xlabel("Épisode")
    plt.ylabel("Loss moyenne")
    plt.grid()
    plt.show()

#========================================================================
# Évaluer la politique figée
#========================================================================
def tester_politique_figee(Q, reinitialiser, faire_un_pas, obtenir_actions, episodes=100):
    total = 0
    for _ in range(episodes):
        etat = reinitialiser()
        termine = False
        gain = 0
        while not termine:
            actions = obtenir_actions(etat)
            action = max(actions, key=lambda a: Q[(etat, a)])
            etat, recompense, termine = faire_un_pas(action)
            gain += recompense
        total += gain
    moyenne = total / episodes
    print(f"\n📈 Performance de la politique figée sur {episodes} épisodes :")
    print(f"→ Gain moyen : {moyenne:.2f}")
    
    return moyenne