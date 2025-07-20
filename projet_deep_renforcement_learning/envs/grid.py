"""
Grille 5 × 5  —  R / D / T

Légende des cases
-----------------
R  (bleu clair)  : case neutre, récompense 0
D  (rouge)       : case dangereuse, récompense –3 et fin d’épisode
T  (bleu foncé)  : case objectif,  récompense +1 et fin d’épisode

• Le départ se fait en haut à gauche (ligne 0, colonne 0).
• Actions possibles partout : 0=HAUT, 1=DROITE, 2=BAS, 3=GAUCHE
• Si l’agent tente de sortir de la grille, il reste en place (récompense 0).
• L’API est identique à celle de Monty Hall : reinitialiser / faire_un_pas / obtenir_actions
"""

import numpy as np
from collections import defaultdict

# ───────────────── constantes de la grille ───────────────────
NB_LIGNES, NB_COLONNES = 5, 5
DEPART = (0, 0)                             # ligne, colonne

# Cases spéciales
CASES_DANGER   = {(0, 3), (1, 4), (2, 4), (3, 4)}   # reward –3
CASES_TERMINAL = {(0, 4), (4, 4)}                   # reward +1

# Actions : 0=Haut, 1=Droite, 2=Bas, 3=Gauche
ACTIONS_IDX = [0, 1, 2, 3]
DEPLACEMENTS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

# ────────── conversion (ligne, col) ↔︎ identifiant d’état ────
def lc_vers_etat(ligne, col):   return ligne * NB_COLONNES + col
def etat_vers_lc(etat):         return divmod(etat, NB_COLONNES)

ETATS = list(range(NB_LIGNES * NB_COLONNES))
ETATS_TERMINAUX = {lc_vers_etat(*xy) for xy in (*CASES_DANGER, *CASES_TERMINAL)}

RECOMPENSES = [0.0, -3.0, +1.0]
IND_RECOMP = {r: i for i, r in enumerate(RECOMPENSES)}

# ────────────────── tenseur p[état, action, état', r_idx] ─────
p = np.zeros((NB_LIGNES * NB_COLONNES, 4,
              NB_LIGNES * NB_COLONNES, len(RECOMPENSES)))

for s in ETATS:
    lig0, col0 = etat_vers_lc(s)

    for a in ACTIONS_IDX:
        # Case terminale : rester sur place prob 1
        if s in ETATS_TERMINAUX:
            p[s, a, s, IND_RECOMP[0.0]] = 1.0
            continue

        dlig, dcol = DEPLACEMENTS[a]
        lig1, col1 = lig0 + dlig, col0 + dcol

        # Sortie de la grille → rebond
        if not (0 <= lig1 < NB_LIGNES and 0 <= col1 < NB_COLONNES):
            s_suiv, r = s, 0.0
        else:
            s_suiv = lc_vers_etat(lig1, col1)
            if (lig1, col1) in CASES_DANGER:
                r = -3.0
            elif (lig1, col1) in CASES_TERMINAL:
                r = +1.0
            else:
                r = 0.0

        p[s, a, s_suiv, IND_RECOMP[r]] = 1.0

# ─────────── interface RL compatible avec tes algos ───────────
_etat_courant = None

def reinitialiser():
    """Réinitialise la partie et renvoie l’état de départ."""
    global _etat_courant
    _etat_courant = lc_vers_etat(*DEPART)
    return _etat_courant

def faire_un_pas(action):
    """Exécute l’action, retourne (état_suivant | None, récompense, terminé)."""
    global _etat_courant
    if _etat_courant in ETATS_TERMINAUX:
        raise RuntimeError("Épisode déjà terminé.")

    # Le tenseur n’a qu’une transition non nulle pour (s,a)
    probs = p[_etat_courant, action]
    s_suiv = int(np.argmax(probs.sum(axis=1)))
    r_idx  = int(np.argmax(probs[s_suiv]))
    recomp = RECOMPENSES[r_idx]
    fini   = s_suiv in ETATS_TERMINAUX

    _etat_courant = None if fini else s_suiv
    return _etat_courant, recomp, fini

def obtenir_actions(etat):
    if etat is None or etat in ETATS_TERMINAUX:
        return []
    return ACTIONS_IDX

# ─────────── alias publics (compatibilité) ────────────────────
etats            = ETATS
actions_idx      = ACTIONS_IDX
recompenses      = RECOMPENSES
etats_terminaux  = list(ETATS_TERMINAUX)
