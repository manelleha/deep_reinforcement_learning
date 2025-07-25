import numpy as np

# États, actions, récompenses
etats = [0, 1, 2, 3, 4]
actions = [0, 1]  # 0 = gauche, 1 = droite
recompenses = [-1.0, 0.0, 1.0]
etats_terminaux = [0, 4]


p = np.zeros((5, 2, 5, 3))
p[3, 0, 2, 1] = 1.0
p[2, 0, 1, 1] = 1.0
p[1, 0, 0, 0] = 1.0
p[3, 1, 4, 2] = 1.0
p[2, 1, 3, 1] = 1.0
p[1, 1, 2, 1] = 1.0

nb_etats = len(etats)
nb_actions = len(actions)


etat_actuel = len(etats) // 2

def reinitialiser():
    global etat_actuel
    etat_actuel = len(etats) // 2
    return etat_actuel

def faire_un_pas(action):
    global etat_actuel
    for s_suiv in range(nb_etats):
        for i_r in range(3):
            if p[etat_actuel, action, s_suiv, i_r] > 0:
                recompense = recompenses[i_r]
                etat_suiv = s_suiv
                etat_actuel = etat_suiv
                termine = etat_suiv in etats_terminaux
                return etat_suiv, recompense, termine
    return etat_actuel, 0.0, False  # fallback

### pour monte carlo 

def faire_un_pas_depuis(etat_depart, action):
    for s_suiv in range(nb_etats):
        for i_r in range(3):
            if p[etat_depart, action, s_suiv, i_r] > 0:
                recompense = recompenses[i_r]
                termine = s_suiv in etats_terminaux
                return s_suiv, recompense, termine
    return etat_depart, 0.0, False

def action_aléatoire():
    return np.random.choice(actions)


def obtenir_actions(s):
    return [a for a in actions if np.sum(p[s, a]) > 0]