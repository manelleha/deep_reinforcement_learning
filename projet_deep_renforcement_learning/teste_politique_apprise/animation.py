"""
Animations Pygame (AUTO / MANU) pour :
  • Line World  ← →
  • Grid R/D/T  (cases neutres, danger –3, objectif +1)
  • Monty Hall  (2 décisions)
  • Pierre‑Feuille‑Ciseaux 2 manches  ← correctif Tour 2

Chaque fonction reçoit :
    policy   : dict { état → action }
    env      : module avec reinitialiser / faire_un_pas / obtenir_actions
    auto_delay (s) : pause entre 2 actions en mode automatique
"""

import pygame, sys, time

# ───────────────────────────────────────── 1) LINE WORLD ──────
def anime_line_world(policy, reinitialiser, faire_un_pas,
                     longueur=9, auto_delay=0.3):
    pygame.init(); T = 80
    W, H = longueur*T, 120
    scr = pygame.display.set_mode((W, H))
    font = pygame.font.SysFont("Arial", 18)
    clock = pygame.time.Clock()

    etat  = reinitialiser()
    auto  = True
    tic   = time.time()

    while True:
        # ─── events
        for e in pygame.event.get():
            if e.type == pygame.QUIT: pygame.quit(); sys.exit()
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE: auto = not auto
                if not auto:
                    if e.key in (pygame.K_LEFT, pygame.K_q):
                        etat, _, _ = faire_un_pas(0)
                    elif e.key in (pygame.K_RIGHT, pygame.K_d):
                        etat, _, _ = faire_un_pas(1)

        # ─── auto
        if auto and time.time() - tic >= auto_delay:
            tic = time.time()
            if etat is None:
                etat = reinitialiser()
            else:
                a = policy.get(etat, 0)
                etat, _, _ = faire_un_pas(a)

        # ─── draw
        scr.fill((240, 240, 240))
        for i in range(longueur):
            pygame.draw.rect(scr, (190, 190, 190),
                             (i*T + 4, 45, T-8, 30), 2)
        if etat is not None:
            pygame.draw.circle(scr, (0, 0, 0),
                               (etat*T + T//2, 60), 15)
        scr.blit(font.render("Espace AUTO/MANU", True, (30, 30, 30)), (6, 6))
        pygame.display.flip(); clock.tick(60)

# ───────────────────────────────────────── 2) GRID R/D/T ──────
def anime_grid_rdt(policy, env, auto_delay=0.25):
    T = 90; m = 2
    W, H = env.NB_COLONNES*T, env.NB_LIGNES*T + 30
    C = dict(fond=(245,245,245), grd=(170,170,170),
             neutre=(180,210,255), danger=(255,80,80),
             obj=(40,110,255), rob=(0,0,0))
    pygame.init(); scr=pygame.display.set_mode((W,H))
    font=pygame.font.SysFont("Arial",18); clock=pygame.time.Clock()

    etat = env.reinitialiser(); auto=True; tic=time.time()
    key2a = {pygame.K_UP:0, pygame.K_z:0, pygame.K_RIGHT:1, pygame.K_d:1,
             pygame.K_DOWN:2, pygame.K_s:2, pygame.K_LEFT:3, pygame.K_q:3}

    while True:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: pygame.quit(); sys.exit()
            elif e.type==pygame.KEYDOWN:
                if e.key==pygame.K_SPACE: auto=not auto
                elif not auto and e.key in key2a and etat not in env.etats_terminaux:
                    etat,_,_ = env.faire_un_pas(key2a[e.key])

        if auto and time.time()-tic>=auto_delay:
            tic=time.time()
            if etat is None: etat=env.reinitialiser()
            elif etat not in env.etats_terminaux:
                a=policy.get(etat, env.obtenir_actions(etat)[0])
                etat,_,_ = env.faire_un_pas(a)

        # draw
        scr.fill(C['fond'])
        for s in env.etats:
            l,c = env.etat_vers_lc(s)
            rect = pygame.Rect(c*T+m, l*T+m, T-2*m, T-2*m)
            if (l,c) in getattr(env,"CASES_DANGER",set()): col=C['danger']
            elif (l,c) in getattr(env,"CASES_TERMINAL",set()): col=C['obj']
            else: col=C['neutre']
            pygame.draw.rect(scr, col, rect)
            if s==etat: pygame.draw.circle(scr, C['rob'], rect.center, T//3)
        for i in range(env.NB_LIGNES+1):
            pygame.draw.line(scr,C['grd'], (0,i*T),(W,i*T))
        for j in range(env.NB_COLONNES+1):
            pygame.draw.line(scr,C['grd'], (j*T,0),(j*T,H-30))
        scr.blit(font.render("Espace AUTO/MANU",True,(25,25,25)),(6,H-24))
        pygame.display.flip(); clock.tick(60)

# ───────────────────────────────────────── 3) MONTY HALL ──────
def anime_monty_hall(policy, env, auto_delay=1.0):
    rev = {v:k for k,v in env.stage1_states.items()}
    W,H,T = 640,380,120; ESP=35
    pygame.init(); scr=pygame.display.set_mode((W,H))
    font=pygame.font.SysFont("Arial",22); clock=pygame.time.Clock()
    portes=[pygame.Rect((W-(3*T+2*ESP))//2+i*(T+ESP),70,T,210) for i in range(3)]
    etat=env.reinitialiser(); auto=True; tic=time.time()
    choix=ouverte=None
    key={'0':0,'1':1,'2':2,'g':0,'c':1}

    while True:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: pygame.quit(); sys.exit()
            elif e.type==pygame.KEYDOWN:
                if e.key==pygame.K_SPACE: auto=not auto
                elif not auto and pygame.key.name(e.key) in key and etat is not None:
                    a=key[pygame.key.name(e.key)]
                    if etat==0 and a in (0,1,2):
                        choix=a; etat,_,_=env.faire_un_pas(a)
                        ch,rem=rev[etat]; ouverte=({0,1,2}-{ch,rem}).pop()
                    elif etat in env.stage1_states.values():
                        etat,_,fin=env.faire_un_pas(a)
                        if fin: time.sleep(1); etat=env.reinitialiser(); choix=ouverte=None

        if auto and time.time()-tic>=auto_delay:
            tic=time.time()
            if etat is None: etat=env.reinitialiser(); choix=ouverte=None
            elif etat==0:
                a=policy.get(etat,0); choix=a
                etat,_,_=env.faire_un_pas(a)
                ch,rem=rev[etat]; ouverte=({0,1,2}-{ch,rem}).pop()
            else:
                a=policy.get(etat,1)
                etat,_,fin=env.faire_un_pas(a)
                if fin: time.sleep(1); etat=env.reinitialiser(); choix=ouverte=None

        scr.fill((245,245,245))
        scr.blit(font.render("Espace AUTO/MANU",True,(30,30,30)),(10,10))
        for i,r in enumerate(portes):
            pygame.draw.rect(scr,(150,100,40),r,border_radius=6)
            if i==choix: pygame.draw.rect(scr,(255,215,0),r,3,border_radius=6)
            if i==ouverte:
                inner=r.inflate(-18,-18)
                pygame.draw.rect(scr,(220,220,220),inner,border_radius=6)
        pygame.display.flip(); clock.tick(60)

# ─────────────────── 4) PIERRE‑FEUILLE‑CISEAUX (2 manches) ────
def anime_pfc(policy, env, auto_delay=0.8):
    icn={0:"R", 1:"P", 2:"S"}   # rock / paper / scissors
    pygame.init(); W,H=520,260
    scr=pygame.display.set_mode((W,H))
    font=pygame.font.SysFont("Arial",22); clock=pygame.time.Clock()
    etat=env.reinitialiser(); auto=True; tic=time.time()
    act_agent=None
    key={'0':0,'1':1,'2':2,'g':0,'c':1}

    def draw():
        scr.fill((250,250,250))
        scr.blit(font.render(f"[Espace] mode {'AUTO' if auto else 'MANU'}",True,(20,20,20)),(10,10))
        scr.blit(font.render("Tour 1" if etat and etat[0]==0 else "Tour 2",True,(20,20,20)),(10,50))
        if act_agent is not None:
            scr.blit(font.render(f"Agent : {icn[act_agent]}",True,(0,0,0)),(210,120))
        pygame.display.flip()

    while True:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: pygame.quit(); sys.exit()
            elif e.type==pygame.KEYDOWN:
                if e.key==pygame.K_SPACE: auto=not auto
                elif not auto and pygame.key.name(e.key) in key and etat is not None:
                    a=key[pygame.key.name(e.key)]
                    if etat[0]==0: act_agent=a
                    etat,_,fin=env.faire_un_pas(a)
                    if fin:
                        scr.fill((250,250,250))
                        scr.blit(font.render("🏁 Manche terminée !",True,(0,120,0)),(140,100))
                        pygame.display.flip(); time.sleep(1)
                        etat=env.reinitialiser(); act_agent=None

        if auto and time.time()-tic>=auto_delay:
            tic=time.time()
            if etat is None:
                etat=env.reinitialiser(); act_agent=None
            else:
                a=policy.get(etat, 0 if etat[0]==0 else 1)
                act_agent=a                               # ← Correctif Tour 2
                etat,_,fin=env.faire_un_pas(a)
                if fin:
                    scr.fill((250,250,250))
                    scr.blit(font.render("🏁 Manche terminée !",True,(0,120,0)),(140,100))
                    pygame.display.flip(); time.sleep(1)
                    etat=env.reinitialiser(); act_agent=None

        draw(); clock.tick(60)
