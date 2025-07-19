import pickle
import os


def sauvegarder_resultats(Q, policy, value_function, dossier):
    """
    Sauvegarde les Q-values, la politique et la fonction de valeur dans le dossier spécifié.
    
    - Q : dictionnaire des Q-values
    - policy : dictionnaire de la politique apprise
    - value_function : dictionnaire des valeurs d’état
    - dossier : chemin complet du dossier où sauvegarder les fichiers
    """
    os.makedirs(dossier, exist_ok=True)  # Crée le dossier s'il n'existe pas

    with open(os.path.join(dossier, "Q_values.pkl"), "wb") as f:
        pickle.dump(Q, f)

    with open(os.path.join(dossier, "policy.pkl"), "wb") as f:
        pickle.dump(policy, f)

    with open(os.path.join(dossier, "value_function.pkl"), "wb") as f:
        pickle.dump(value_function, f)

    print(f"✅ Sauvegarde réussie dans : {dossier}")

def charger_resultats_depuis_chemin(dossier):
    """
    Charge les résultats depuis un chemin absolu donné en entrée.
    """
    if not os.path.exists(dossier):
        print(f"❌ Dossier introuvable : {dossier}")
        return None, None, None

    try:
        with open(os.path.join(dossier, "Q_values.pkl"), "rb") as f:
            Q = pickle.load(f)
        with open(os.path.join(dossier, "policy.pkl"), "rb") as f:
            policy = pickle.load(f)
        with open(os.path.join(dossier, "value_function.pkl"), "rb") as f:
            value_function = pickle.load(f)

        print("✅ Chargement réussi depuis :", dossier)
        return Q, policy, value_function

    except FileNotFoundError as e:
        print(f"❌ Fichier manquant : {e}")
        return None, None, None

# Test avec la politique apprise
def tester_policy_en_console(policy, reinitialiser, faire_un_pas, episodes=10):
    if policy is None:
        print("❌ Impossible de tester: politique non chargée")
        return
    
    total_gain = 0
    for ep in range(1, episodes + 1):
        etat = reinitialiser()
        gain = 0
        print(f"\n🎬 Épisode {ep}")
        while True:
            action = policy.get(etat, 0)  # fallback: action 0
            print(f"🧠 État: {etat}, 🎮 Action: {action}")
            etat_suiv, recompense, termine = faire_un_pas(action)
            gain += recompense
            if termine:
                print(f"🏁 Terminé à l'état {etat_suiv} avec gain {gain}")
                break
            etat = etat_suiv
        total_gain += gain
    print(f"\n📈 Gain moyen sur {episodes} épisodes : {total_gain / episodes:.2f}")



############################ Animation 

def jouer_avec_politique(policy, reinitialiser, faire_un_pas, delay=1.0):
    import time
    etat = reinitialiser()
    termine = False
    print("\n🎮 Déroulement automatique de la politique :\n")

    while not termine:
        action = policy.get(etat, None)
        if action is None:
            print(f"⚠️ Aucune action définie pour l'état {etat}")
            break
        print(f"🧠 État : {etat} → 🎮 Action choisie : {action}")
        etat, recompense, termine = faire_un_pas(action)
        print(f"→ 🌟 Récompense : {recompense} | 🚩 Terminé : {termine}\n")
        time.sleep(delay)
    
    print("✅ Fin du déroulement.")

def jouer_manuellement(reinitialiser, faire_un_pas, obtenir_actions):
    etat = reinitialiser()
    termine = False
    print("\n🕹️ Mode manuel activé : choisis toi-même les actions !\n")

    while not termine:
        actions = obtenir_actions(etat)
        print(f"🧠 État actuel : {etat}")
        print("Actions possibles :")
        for i, a in enumerate(actions):
            print(f" {i}. {a}")
        
        choix = input("👉 Entrez le numéro de l'action choisie : ")
        try:
            index = int(choix)
            if index < 0 or index >= len(actions):
                print("⛔ Numéro invalide.")
                continue
            action = actions[index]
        except:
            print("⛔ Entrée invalide.")
            continue
        
        etat, recompense, termine = faire_un_pas(action)
        print(f"→ 🎮 Action : {action} | 🌟 Récompense : {recompense} | 🚩 Terminé : {termine}\n")

    print("✅ Fin du test manuel.")
