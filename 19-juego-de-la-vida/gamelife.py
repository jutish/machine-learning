import pygame
import numpy as np
import time

pygame.init()
height, width = 500, 500 #  Alto y ancho de la pantalla
screen = pygame.display.set_mode((height, width))
bg = 25, 25, 25 # Background
screen.fill(bg)

# Nro de celdas en el eje X y en el eje Y
nxC, nyC = 50, 50

# Alto y Ancho de cada celda viene dado por el alto y ancho de la pantalla
dimCH = height / nyC
dimCW = width / nxC

# Estado del juego. Celda viva = 1, celda muerta = 0
gameState = np.zeros((nxC, nyC))

# Automata Palo
gameState[5, 3] = 1
gameState[5, 4] = 1
gameState[5, 5] = 1

#Automata movil
gameState[21, 21] = 1
gameState[22, 22] = 1
gameState[22, 23] = 1
gameState[21, 23] = 1
gameState[20, 23] = 1

# Recorro la pantalla
while True:
    newGameState = np.copy(gameState)
    screen.fill(bg)
    time.sleep(0.1)

    for y in range(nyC):
        for x in range(nxC):
            n_neigh = gameState[(x-1) % nxC, (y-1) % nyC] + \
                      gameState[(x  ) % nxC, (y-1) % nyC] + \
                      gameState[(x+1) % nxC, (y-1) % nyC] + \
                      gameState[(x-1) % nxC, (y  ) % nyC] + \
                      gameState[(x+1) % nxC, (y  ) % nyC] + \
                      gameState[(x-1) % nxC, (y+1) % nyC] + \
                      gameState[(x  ) % nxC, (y+1) % nyC] + \
                      gameState[(x+1) % nxC, (y+1) % nyC]

            # Regla 1: Una celula muerta con 3 vecinos revive
            if gameState[x, y] == 0 and n_neigh == 3:
                newGameState[x, y] = 1
            # Regla 2: Una celula viva con menos de 2 o mas de 3 vecinos muere
            elif gameState[x, y] == 1 and (n_neigh < 2 or n_neigh > 3):
                newGameState[x, y] = 0

            poly = [(x * dimCW, y * dimCH),
                    ((x+1) * dimCW, y * dimCH),
                    ((x+1) * dimCW, (y+1) * dimCH),
                    (x * dimCW, (y+1) * dimCH)]

            if newGameState[x, y] == 0:
                pygame.draw.polygon(screen, (128, 128, 128), poly, width=1)
            else:
                pygame.draw.polygon(screen, (255, 255, 255), poly, width=0)
    gameState = np.copy(newGameState)
    pygame.display.flip() # Muestro lo dibujado
