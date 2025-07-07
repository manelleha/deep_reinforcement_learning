import pygame
import time
import numpy as np
from envs.line_world import (
    nb_etats, états_terminaux,
    réinitialiser, faire_un_pas
)

# === Paramètres Pygame ===
TAILLE_CASE = 100
WIDTH, HEIGHT = TAILLE_CASE * nb_etats, 100

BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ROUGE = (255, 0, 0)
VERT = (0, 200, 0)

def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Line World")
    return screen

def dessiner_env(screen, pos):
    screen.fill(BLANC)
    for i in range(nb_etats):
        rect = pygame.Rect(i * TAILLE_CASE, 20, TAILLE_CASE, 60)
        couleur = VERT if i in états_terminaux else NOIR
        pygame.draw.rect(screen, couleur, rect, 2)
    rect_agent = pygame.Rect(pos * TAILLE_CASE + 25, 35, 50, 30)
    pygame.draw.rect(screen, ROUGE, rect_agent)
    pygame.display.flip()

def jouer_politique(Q, delay=0.7):
    politique = np.argmax(Q, axis=1)
    screen = init_pygame()
    état = réinitialiser()
    dessiner_env(screen, état)
    time.sleep(delay)

    while état not in états_terminaux:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        action = politique[état]
        état, _, _ = faire_un_pas(action)
        dessiner_env(screen, état)
        time.sleep(delay)

    time.sleep(1)
    pygame.quit()

def jouer_manuellement():
    screen = init_pygame()
    état = réinitialiser()
    dessiner_env(screen, état)

    en_cours = True
    while en_cours:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                en_cours = False
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = 0
                elif event.key == pygame.K_RIGHT:
                    action = 1
                else:
                    continue
                état, r, terminé = faire_un_pas(action)
                dessiner_env(screen, état)
                print(f"Action : {action}, Récompense : {r}")
                if terminé:
                    print("Épisode terminé.")
                    en_cours = False
                    time.sleep(1.5)

    pygame.quit()
