
##======================================== DYNA-Q =============================================================

import numpy as np
import random
from collections import defaultdict

def dyna_q(
    nb_etats,
    nb_actions,
    réinitialiser,
    faire_un_pas,
    action_aléatoire,
    episodes=100,
    alpha=0.1,
    gamma=0.95,
    epsilon=0.1,
    etapes_planification=10,
    debug=False
):
    Q = np.zeros((nb_etats, nb_actions))
    modele = defaultdict(lambda: (0, 0))
    historiques = set()

    for ep in range(episodes):
        état = réinitialiser()
        terminé = False
        if debug:
            print(f"\n--- Épisode {ep+1} ---")

        while not terminé:
            # Choix action
            if random.random() < epsilon:
                action = action_aléatoire()
                source = "exploration"
            else:
                action = np.argmax(Q[état])
                source = "exploitation"

            état_suiv, récompense, terminé = faire_un_pas(action)

            # Mise à jour réelle
            ancien_q = Q[état, action]
            Q[état, action] += alpha * (récompense + gamma * np.max(Q[état_suiv]) - Q[état, action])

            modele[(état, action)] = (état_suiv, récompense)
            historiques.add((état, action))

            if debug:
                print(f"État: {état}, Action: {action} ({source}), État suivant: {état_suiv}, Récompense: {récompense}")
                print(f"  Ancien Q[{état},{action}] = {ancien_q:.3f}, Nouveau = {Q[état, action]:.3f}")

            # Hallucinations (planification)
            for _ in range(etapes_planification):
                e, a = random.choice(list(historiques))
                e_suiv, r = modele[(e, a)]
                Q[e, a] += alpha * (r + gamma * np.max(Q[e_suiv]) - Q[e, a])

            état = état_suiv

    return Q


