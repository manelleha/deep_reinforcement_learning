import numpy as np

# États, actions, récompenses
états = [0, 1, 2, 3, 4]
actions = [0, 1]  # 0 = gauche, 1 = droite
récompenses = [-1.0, 0.0, 1.0]
états_terminaux = [0, 4]

# Transitions : état, action, état suivant, index de récompense
p = np.zeros((5, 2, 5, 3))
p[3, 0, 2, 1] = 1.0
p[2, 0, 1, 1] = 1.0
p[1, 0, 0, 0] = 1.0
p[3, 1, 4, 2] = 1.0 # etat qui donne +1 
p[2, 1, 3, 1] = 1.0
p[1, 1, 2, 1] = 1.0

nb_etats = len(états)
nb_actions = len(actions)

# État global actuel de l'agent
état_actuel = 3

def réinitialiser():
    global état_actuel
    état_actuel = 3
    return état_actuel

def faire_un_pas(action):
    global état_actuel
    for s_suiv in range(nb_etats):
        for i_r in range(3):
            if p[état_actuel, action, s_suiv, i_r] > 0:
                récompense = récompenses[i_r]
                état_suiv = s_suiv
                état_actuel = état_suiv
                terminé = état_suiv in états_terminaux
                return état_suiv, récompense, terminé
    return état_actuel, 0.0, False  # fallback

def action_aléatoire():
    return np.random.choice(actions)
