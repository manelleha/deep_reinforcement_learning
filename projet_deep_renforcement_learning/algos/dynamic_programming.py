import numpy as np
import pandas as pd
from tqdm import trange
from collections import defaultdict

def policy_iteration_dp(
    etats,
    actions,
    p,  # Tenseur de probabilités [état, action, état_suivant, récompense_index]
    recompenses,
    etats_terminaux,
    gamma=0.99,
    theta=0.00001,
    max_iterations=100,
    max_eval_iterations=100,
    verbose=False
):
    
    
    # Initialisation
    V = {s: 0.0 for s in etats}
    
    # Politique initiale : première action valide pour chaque état
    policy = {}
    for s in etats:
        actions_valides = [a for a in actions if np.sum(p[s, a]) > 0]
        policy[s] = actions_valides[0] if actions_valides else actions[0]
    
    historique_loss = []
    historique_V = []
    
    if verbose:
        print(f"🧠 États : {etats}")
        print(f"🎮 Actions : {actions}")
        print(f"🏁 États terminaux : {etats_terminaux}")
        print(f"🔄 Début de Policy Iteration (Dynamic Programming)...")
    
    for it in trange(max_iterations, desc="Policy Iteration DP"):
        
        # ============== ÉVALUATION DE LA POLITIQUE ====================
        if verbose:
            print(f"\n📈 Évaluation de la politique (itération {it+1})")
            
        for eval_it in range(max_eval_iterations):
            delta = 0
            V_new = V.copy()
            
            for s in etats:
                if s in etats_terminaux:
                    continue  # Les états terminaux gardent V=0
                    
                a = policy[s]
                v_old = V[s]
                
                # Calcul exact avec les probabilités de transition
                v_new = 0.0
                for s_next in etats:
                    for r_idx in range(len(recompenses)):
                        prob = p[s, a, s_next, r_idx]
                        if prob > 0:
                            reward = recompenses[r_idx]
                            if s_next in etats_terminaux:
                                v_new += prob * reward  # Pas de continuation si terminal
                            else:
                                v_new += prob * (reward + gamma * V[s_next])
                
                V_new[s] = v_new
                delta = max(delta, abs(v_old - v_new))
                
                if verbose:
                    print(f"🧠 État {s} | 🎮 Action {a} | 📊 V: {v_old:.4f} → {v_new:.4f}")
            
            V = V_new
            
            if verbose:
                print(f"📉 Delta max = {delta:.6f}")
                
            if delta < theta:
                if verbose:
                    print(f"✅ Convergence atteinte en {eval_it+1} itérations d'évaluation")
                break
        
        historique_loss.append(delta)
        historique_V.append(dict(V))
        
        # ============== AMÉLIORATION DE LA POLITIQUE ====================
        if verbose:
            print("🎯 Amélioration de la politique...")
            
        policy_stable = True
        
        for s in etats:
            if s in etats_terminaux:
                continue  # Pas d'action dans les états terminaux
                
            old_action = policy[s]
            
            # Calculer Q-values pour toutes les actions valides
            q_values = {}
            actions_valides = [a for a in actions if np.sum(p[s, a]) > 0]
            
            for a in actions_valides:
                q_val = 0.0
                for s_next in etats:
                    for r_idx in range(len(recompenses)):
                        prob = p[s, a, s_next, r_idx]
                        if prob > 0:
                            reward = recompenses[r_idx]
                            if s_next in etats_terminaux:
                                q_val += prob * reward
                            else:
                                q_val += prob * (reward + gamma * V[s_next])
                
                q_values[a] = q_val
            
            # Choisir la meilleure action
            if q_values:
                best_action = max(q_values, key=q_values.get)
                policy[s] = best_action
                
                if verbose:
                    print(f"🧠 État {s} | 🔁 {old_action} → 🎮 {best_action}")
                    for act, val in q_values.items():
                        marker = "🏆" if act == best_action else "  "
                        print(f"    {marker} Q({s},{act}) = {val:.4f}")
                
                if best_action != old_action:
                    policy_stable = False
        
        if verbose:
            print(f"✅ Politique stable : {policy_stable}")
            
        if policy_stable:
            if verbose:
                print(f"🎉 Convergence de la politique en {it+1} itérations !")
            break
    
    # Créer DataFrame avec l'historique des valeurs
    df_v = pd.DataFrame(historique_V).fillna(0)
    
    return policy, dict(V), df_v, historique_loss


def value_iteration_dp(
    etats,
    actions,
    p,
    recompenses,
    etats_terminaux,
    gamma=0.99,
    theta=0.00001,
    max_iterations=100,
    verbose=False
):
    # Initialisation des valeurs d'état
    V = {s: 0.0 for s in etats}
    historique_loss = []
    historique_V = []

    if verbose:
        print(f"🧠 États : {etats}")
        print(f"🎮 Actions : {actions}")
        print(f"🏁 États terminaux : {etats_terminaux}")
        print(f"🔄 Début de Value Iteration (Dynamic Programming)...")

    for it in trange(max_iterations, desc="Value Iteration DP"):
        delta = 0
        V_new = V.copy()

        for s in etats:
            if s in etats_terminaux:
                continue  # Les états terminaux restent à 0

            v_old = V[s]
            q_values = []

            for a in actions:
                q_val = 0.0
                for s_next in etats:
                    for r_idx in range(len(recompenses)):
                        prob = p[s, a, s_next, r_idx]
                        if prob > 0:
                            r = recompenses[r_idx]
                            if s_next in etats_terminaux:
                                q_val += prob * r
                            else:
                                q_val += prob * (r + gamma * V[s_next])
                q_values.append(q_val)

            V_new[s] = max(q_values)
            delta = max(delta, abs(v_old - V_new[s]))

            if verbose:
                print(f"🧠 État {s} | 📊 V: {v_old:.4f} → {V_new[s]:.4f}")

        V = V_new
        historique_loss.append(delta)
        historique_V.append(dict(V))

        if verbose:
            print(f"📉 Delta max = {delta:.6f}")

        if delta < theta:
            if verbose:
                print(f"✅ Convergence atteinte à l'itération {it+1}")
            break

    # Politique déterministe
    policy = {}
    for s in etats:
        if s in etats_terminaux:
            policy[s] = None
            continue

        best_a = None
        best_q = -np.inf

        for a in actions:
            q_val = 0.0
            for s_next in etats:
                for r_idx in range(len(recompenses)):
                    prob = p[s, a, s_next, r_idx]
                    if prob > 0:
                        r = recompenses[r_idx]
                        if s_next in etats_terminaux:
                            q_val += prob * r
                        else:
                            q_val += prob * (r + gamma * V[s_next])

            if q_val > best_q:
                best_q = q_val
                best_a = a

        policy[s] = best_a

        if verbose:
            print(f"🎮 Politique[{s}] = {best_a} avec Q = {best_q:.4f}")

    df_v = pd.DataFrame(historique_V).fillna(0)

    return policy, dict(V), df_v, historique_loss 