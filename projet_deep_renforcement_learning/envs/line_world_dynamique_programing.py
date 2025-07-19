import numpy as np
from collections import defaultdict

# États, actions, récompenses
etats = [0, 1, 2, 3, 4]
actions = [0, 1]  # 0 = gauche, 1 = droite
recompenses = [-1.0, 0.0, 1.0]
etats_terminaux = [0, 4]

# Transitions sous forme de tenseur
p = np.zeros((5, 2, 5, 3))
p[3, 0, 2, 1] = 1.0
p[2, 0, 1, 1] = 1.0
p[1, 0, 0, 0] = 1.0
p[3, 1, 4, 2] = 1.0
p[2, 1, 3, 1] = 1.0
p[1, 1, 2, 1] = 1.0

# Conversion en dictionnaire P[s][a] = [(prob, s', r, done), ...]
P = defaultdict(lambda: defaultdict(list))
actions_dict = {}

for s in etats:
    actions_disponibles = []
    for a in actions:
        transitions = []
        for s_suiv in etats:
            for i_r, r in enumerate(recompenses):
                prob = p[s, a, s_suiv, i_r]
                if prob > 0:
                    done = s_suiv in etats_terminaux
                    transitions.append((prob, s_suiv, r, done))
        if transitions:
            P[s][a] = transitions
            actions_disponibles.append(a)
    actions_dict[s] = actions_disponibles
