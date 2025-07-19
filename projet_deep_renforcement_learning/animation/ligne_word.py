# demo_soutenance.py
"""
Script principal pour la démonstration lors de la soutenance
Intègre votre système de sauvegarde avec l'interface Pygame
"""

import pygame
import os
import pickle
import sys
from pygame_interface import InterfaceRL, ModeJeu, Couleurs

# Vos fonctions de sauvegarde (importées depuis votre code)
def charger_resultats(algo_name="sarsa"):
    """Votre fonction de chargement existante"""
    chemin = os.path.join("sauvegarde", algo_name)
    
    with open(os.path.join(chemin, "Q_values.pkl"), "rb") as f:
        Q = pickle.load(f)
    
    with open(os.path.join(chemin, "policy.pkl"), "rb") as f:
        policy = pickle.load(f)
    
    with open(os.path.join(chemin, "value_function.pkl"), "rb") as f:
        value_function = pickle.load(f)
    
    print(f"✅ Résultats chargés depuis : {chemin}")
    return Q, policy, value_function

class InterfaceSoutenance(InterfaceRL):
    """Interface spécialisée pour la soutenance"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.algos_disponibles = ["sarsa", "q_learning", "expected_sarsa"]
        self.algo_actuel = "sarsa"
        self.statistiques = {}
        self.mode_comparaison = False
        
    def charger_agent(self, algo_name=None):
        """Version améliorée du chargement"""
        if algo_name is None:
            algo_name = self.algo_actuel
            
        try:
            self.Q, self.policy, self.value_function = charger_resultats(algo_name)
            self.algo_actuel = algo_name
            
            # Calculer des statistiques
            self.calculer_statistiques()
            
            print(f"✅ Agent '{algo_name}' chargé pour la démonstration")
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement de '{algo_name}': {e}")
            return False
    
    def calculer_statistiques(self):
        """Calcule des statistiques pour la présentation"""
        if not self.policy or not self.value_function:
            return
            
        self.statistiques = {
            'nb_etats': len(self.policy),
            'nb_paires_qa': len(self.Q) if self.Q else 0,
            'valeur_max': max(self.value_function.values()) if self.value_function else 0,
            'valeur_min': min(self.value_function.values()) if self.value_function else 0,
            'valeur_moyenne': sum(self.value_function.values()) / len(self.value_function) if self.value_function else 0,
        }
        
        # Distribution des actions
        actions_count = {}
        for action in self.policy.values():
            actions_count[action] = actions_count.get(action, 0) + 1
        self.statistiques['distribution_actions'] = actions_count
    
    def dessiner_statistiques(self):
        """Affiche les statistiques pour la soutenance"""
        if not self.statistiques:
            return
            
        # Zone pour les statistiques
        zone_stats = pygame.Rect(50, 620, 600, 100)
        pygame.draw.rect(self.ecran, Couleurs.GRIS_CLAIR, zone_stats)
        pygame.draw.rect(self.ecran, Couleurs.NOIR, zone_stats, 2)
        
        # Titre
        titre = self.police_normale.render(f"Statistiques - {self.algo_actuel.upper()}", 
                                         True, Couleurs.ROUGE)
        self.ecran.blit(titre, (zone_stats.x + 10, zone_stats.y + 5))
        
        # Statistiques en colonnes
        col1_x = zone_stats.x + 10
        col2_x = zone_stats.x + 200
        col3_x = zone_stats.x + 400
        y_base = zone_stats.y + 30
        
        # Colonne 1
        stats_col1 = [
            f"États: {self.statistiques['nb_etats']}",
            f"Paires Q: {self.statistiques['nb_paires_qa']}",
        ]
        
        # Colonne 2
        stats_col2 = [
            f"V max: {self.statistiques['valeur_max']:.3f}",
            f"V min: {self.statistiques['valeur_min']:.3f}",
            f"V moy: {self.statistiques['valeur_moyenne']:.3f}",
        ]
        
        # Colonne 3 - Distribution des actions
        stats_col3 = []
        for action, count in self.statistiques['distribution_actions'].items():
            pourcentage = (count / self.statistiques['nb_etats']) * 100
            stats_col3.append(f"Act {action}: {count} ({pourcentage:.1f}%)")
        
        # Affichage
        for i, stat in enumerate(stats_col1):
            texte = self.police_petite.render(stat, True, Couleurs.NOIR)
            self.ecran.blit(texte, (col1_x, y_base + i * 20))
            
        for i, stat in enumerate(stats_col2):
            texte = self.police_petite.render(stat, True, Couleurs.NOIR)
            self.ecran.blit(texte, (col2_x, y_base + i * 20))
            
        for i, stat in enumerate(stats_col3[:3]):  # Limité à 3 lignes
            texte = self.police_petite.render(stat, True, Couleurs.NOIR)
            self.ecran.blit(texte, (col3_x, y_base + i * 20))
    
    def dessiner_mode_soutenance(self):
        """Affichage spécial pour la soutenance"""
        # Titre principal
        titre = self.police_titre.render("DÉMONSTRATION AGENT RL - SOUTENANCE", 
                                       True, Couleurs.ROUGE)
        titre_rect = titre.get_rect(center=(self.largeur // 2, 30))
        self.ecran.blit(titre, titre_rect)
        
        # Sous-titre avec l'algorithme
        sous_titre = self.police_normale.render(f"Algorithme: {self.algo_actuel.upper()}", 
                                              True, Couleurs.BLEU)
        sous_titre_rect = sous_titre.get_rect(center=(self.largeur // 2, 60))
        self.ecran.blit(sous_titre, sous_titre_rect)
        
        # Indicateur de mode
        mode_text = f"Mode: {self.mode_actuel.value}"
        if self.pause:
            mode_text += " (PAUSE)"
        
        mode_surface = self.police_normale.render(mode_text, True, Couleurs.VERT)
        self.ecran.blit(mode_surface, (self.largeur - 250, 30))
    
    def gerer_evenements_soutenance(self):
        """Gestion d'événements spécialisée pour la soutenance"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.en_cours = False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.en_cours = False
                elif event.key == pygame.K_SPACE:
                    self.pause = not self.pause
                    print(f"⏸️ {'Pause' if self.pause else 'Reprise'}")
                elif event.key == pygame.K_r:
                    self.reset_episode()
                elif event.key == pygame.K_n:
                    self.faire_etape_suivante()
                elif event.key == pygame.K_1:
                    self.charger_agent("sarsa")
                elif event.key == pygame.K_2:
                    self.charger_agent("q_learning")
                elif event.key == pygame.K_3:
                    self.charger_agent("expected_sarsa")
                elif event.key == pygame.K_m:
                    self.mode_actuel = ModeJeu.MANUEL
                elif event.key == pygame.K_a:
                    self.mode_actuel = ModeJeu.AGENT_AUTO
                elif event.key == pygame.K_s:
                    self.mode_actuel = ModeJeu.AGENT_PAS_A_PAS
                elif event.key == pygame.K_i:
                    self.mode_actuel = ModeJeu.ANALYSE
                    
                # Contrôles manuels
                elif self.mode_actuel == ModeJeu.MANUEL:
                    action = None
                    if event.key == pygame.K_UP:
                        action = 0
                    elif event.key == pygame.K_DOWN:
                        action = 1