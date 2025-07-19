import random

# ─────────────────────────────────────────────
# Définir les actions possibles
# ─────────────────────────────────────────────
actions = ["rock", "paper", "scissors"]
action_to_index = {a: i for i, a in enumerate(actions)}

# ─────────────────────────────────────────────
# Variable globale pour maintenir l'état
# ─────────────────────────────────────────────
current_state = None

# ─────────────────────────────────────────────
# Initialiser l'état : (round, action_1 de l'agent)
# ─────────────────────────────────────────────
def reinitialiser():
    global current_state
    current_state = (0, None)
    return current_state

# ─────────────────────────────────────────────
# Fonction utilitaire pour calculer le score
# ─────────────────────────────────────────────
def resultat(agent_action, adversaire_action):
    if agent_action == adversaire_action:
        return 0
    elif (agent_action == "rock"     and adversaire_action == "scissors") or \
         (agent_action == "paper"    and adversaire_action == "rock") or \
         (agent_action == "scissors" and adversaire_action == "paper"):
        return 1
    else:
        return -1

# ─────────────────────────────────────────────
# Fonction de transition adaptée pour SARSA
# ─────────────────────────────────────────────
def faire_un_pas(action_index):
    global current_state
    
    if current_state is None:
        raise ValueError("L'environnement n'a pas été initialisé. Appelez reinitialiser() d'abord.")
    
    round_actuel, action_1 = current_state
    action_agent = actions[action_index]
    
    if round_actuel == 0:
        # Premier round : adversaire joue aléatoirement
        adversaire_action = random.choice(actions)
        reward = resultat(action_agent, adversaire_action)
        prochain_etat = (1, action_agent)
        est_terminal = False
        
        # Mettre à jour l'état global
        current_state = prochain_etat
        
    elif round_actuel == 1:
        # Deuxième round : adversaire copie l'action précédente de l'agent
        adversaire_action = action_1
        reward = resultat(action_agent, adversaire_action)
        prochain_etat = None  # État terminal
        est_terminal = True
        
        # Mettre à jour l'état global
        current_state = prochain_etat
        
    else:
        raise ValueError(f"Round invalide : {round_actuel}")
    
    return prochain_etat, reward, est_terminal

# ─────────────────────────────────────────────
# Fonction pour obtenir les actions possibles
# ─────────────────────────────────────────────
def obtenir_actions(etat):
    """Retourne les actions possibles pour un état donné"""
    if etat is None:  # État terminal
        return []
    
    round_actuel, _ = etat
    if round_actuel in [0, 1]:
        return [0, 1, 2]  # Indices pour rock, paper, scissors
    else:
        return []

# ─────────────────────────────────────────────
# Fonction utilitaire pour afficher l'état
# ─────────────────────────────────────────────
def afficher_etat(etat):
    if etat is None:
        return "État terminal"
    
    round_actuel, action_1 = etat
    if round_actuel == 0:
        return f"Round {round_actuel} (début)"
    elif round_actuel == 1:
        return f"Round {round_actuel} (action précédente: {action_1})"
    else:
        return f"Round {round_actuel} (inconnu)"


########################### Pour le dynamique programing =================================

import numpy as np

# ----- constantes -----
S, A, R = 5, 3, 3
p = np.zeros((S, A, S, R))

def r_to_idx(r):          # -1→0, 0→1, +1→2
    return { -1:0, 0:1, 1:2 }[r]

# état 0  (round 0)
for a_agent in range(3):          # tes 3 coups
    for adv in range(3):          # rock/paper/scissors 1/3
        prob = 1/3
        next_state = 1 + a_agent   # 1,2,3
        reward = resultat(actions[a_agent], actions[adv])  # -1/0/1
        p[0, a_agent, next_state, r_to_idx(reward)] += prob

# états 1–3  (round 1)
for state in [1,2,3]:
    prev_action = actions[state-1]      # rock/paper/scissors
    for a_agent in range(3):
        adv = prev_action               # copie
        reward = resultat(actions[a_agent], adv)
        p[state, a_agent, 4, r_to_idx(reward)] = 1.0

# état terminal 4 : déjà 0 partout
etats           = list(range(5))
actions_idx     = [0,1,2]
recompenses     = [-1.0, 0.0, 1.0]
etats_terminaux = [4]
