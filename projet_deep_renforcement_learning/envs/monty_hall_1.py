

  #  P(gagner | rester)  = 1/3
   #  P(gagner | changer) = 2/3



import numpy as np
import random

# ───────────────── constantes générales ──────────────────────
DOORS = [0, 1, 2]                        # indices des portes (A,B,C)
REWARDS = [0.0, 1.0]                     # r_idx 0 → 0, 1 → 1
R_IDX = {r: i for i, r in enumerate(REWARDS)}

# ─────────── encodage des états pour la planification ─────────
# état 0 : début ; états 1‑6 : (chosen, remaining) ; état 7 : terminal
stage1_states = {}
idx = 1
for chosen in DOORS:
    for remaining in DOORS:
        if remaining != chosen:
            stage1_states[(chosen, remaining)] = idx
            idx += 1
TERMINAL_STATE = 7

ETATS           = list(range(8))
ACTIONS_IDX     = [0, 1, 2]
ETATS_TERMINAUX = [TERMINAL_STATE]
RECOMPENSES     = REWARDS

p = np.zeros((8, 3, 8, 2))               


for a in DOORS:                  
    for winning in DOORS:        
        prob_w = 1/3
        others = [d for d in DOORS if d != a]

        if winning == a:
            
            for revealed in others:
                remaining = (set(others) - {revealed}).pop()
                s_next = stage1_states[(a, remaining)]
                p[0, a, s_next, R_IDX[0.0]] += prob_w * 0.5  
        else:
            
            revealed = (set(others) - {winning}).pop()
            remaining = winning
            s_next = stage1_states[(a, remaining)]
            p[0, a, s_next, R_IDX[0.0]] += prob_w            


for (chosen, remaining), s in stage1_states.items():
   

   
    p[s, 0, TERMINAL_STATE, R_IDX[1.0]] = 1/3   # reward 1
    p[s, 0, TERMINAL_STATE, R_IDX[0.0]] = 2/3   # reward 0

   
    p[s, 1, TERMINAL_STATE, R_IDX[1.0]] = 2/3
    p[s, 1, TERMINAL_STATE, R_IDX[0.0]] = 1/3

# ─────────── interface RL (SARSA / Q‑Learning) ───────────────
_current = {}


def reinitialiser():
    """Réinitialise l’environnement et renvoie l’état initial (0)."""
    _current.clear()
    _current["winning"] = random.choice(DOORS)
    _current["stage"] = 0
    _current["chosen"] = None
    _current["remaining"] = None
    return 0


def faire_un_pas(action_idx):
    stage = _current["stage"]

    if stage == 0:
        chosen = action_idx
        winning = _current["winning"]
        others = [d for d in DOORS if d != chosen]

        if winning == chosen:
            revealed = random.choice(others)
        else:
            revealed = (set(others) - {winning}).pop()
        remaining = (set(others) - {revealed}).pop()

        _current.update(stage=1, chosen=chosen, remaining=remaining)
        next_state = stage1_states[(chosen, remaining)]
        return next_state, 0.0, False

    elif stage == 1:
        chosen = _current["chosen"]
        remaining = _current["remaining"]
        winning = _current["winning"]

        final_choice = chosen if action_idx == 0 else remaining
        reward = 1.0 if final_choice == winning else 0.0

        _current["stage"] = 2  # terminal
        return None, reward, True

    else:
        raise RuntimeError("Action après état terminal.")


def obtenir_actions(etat):
    if etat is None or etat in ETATS_TERMINAUX:
        return []
    if etat == 0:
        return [0, 1, 2]
    return [0, 1]  # rester / changer


etats           = ETATS
actions_idx     = ACTIONS_IDX
recompenses     = RECOMPENSES
etats_terminaux = ETATS_TERMINAUX

