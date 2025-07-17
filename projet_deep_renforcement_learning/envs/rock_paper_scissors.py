import random

# Définir les actions possibles
actions = ["rock", "paper", "scissors"]

# Initialiser l’état (round actuel, action agent au round 1)
def reinitialiser():
    return (0, None)  # round 0, pas encore d'action passée

# Déterminer le résultat d'un round
def resultat(agent_action, adversaire_action):
    if agent_action == adversaire_action:
        return 0
    elif (agent_action == "rock" and adversaire_action == "scissors") or \
         (agent_action == "paper" and adversaire_action == "rock") or \
         (agent_action == "scissors" and adversaire_action == "paper"):
        return 1
    else:
        return -1

# Fonction de transition : renvoie (état_suivant, récompense, terminal)
def faire_un_pas(etat, action_index):
    round_actuel, action_1 = etat
    action_agent = actions[action_index]

    if round_actuel == 0:
        adversaire_action = random.choice(actions)
        reward = resultat(action_agent, adversaire_action)
        prochain_etat = (1, action_agent)
        est_terminal = False

    elif round_actuel == 1:
        adversaire_action = action_1  # Il copie l’action du round précédent
        reward = resultat(action_agent, adversaire_action)
        prochain_etat = None
        est_terminal = True

    return prochain_etat, reward, est_terminal
