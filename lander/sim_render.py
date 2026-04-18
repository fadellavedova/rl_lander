import pygame
import numpy as np

# Config
WIDTH, HEIGHT = 800, 600
SCALE = 50  # metros → pixels

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

background = pygame.image.load("Stars-fixed.jpg")
background = pygame.transform.scale(background, (WIDTH, HEIGHT))

def world_to_screen(x, z):
    px = WIDTH // 2 + int(x * SCALE)
    py = HEIGHT - int(z * SCALE)
    return px, py

def draw_lander(screen, state, thrust):
    x, z, vx, vz, theta, omega = state

    px, py = world_to_screen(x, z)

    # tamaño del lander
    w, h = 20, 40

    # cuerpo principal
    body = np.array([
        [-10, -15],
        [10, -15],
        [12, 10],
        [-12, 10]
    ])

    # patas
    legs = [
        np.array([[-12, 10], [-20, 20]]),
        np.array([[12, 10], [20, 20]])
    ]

    # ventana
    window = np.array([
        [-4, -10],
        [4, -10],
        [4, -2],
        [-4, -2]
    ])

    # rotación
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    def transform(shape):
        pts = shape @ R.T
        pts[:, 0] += px
        pts[:, 1] += py
        return pts

    # dibujar cuerpo
    pygame.draw.polygon(screen, (200, 200, 210), transform(body))

    # sombreado
    pygame.draw.polygon(screen, (150, 150, 160), transform(body * [0.9, 1.0]))

    # patas
    for leg in legs:
        pts = transform(leg)
        pygame.draw.line(screen, (220, 220, 220), pts[0], pts[1], 3)

    # ventana
    pygame.draw.polygon(screen, (100, 200, 255), transform(window))

    # fuego
    if thrust > 0:
        flame = np.array([
            [-5, 10],
            [5, 10],
            [0, 20 + np.random.rand()*10]
        ])
        pygame.draw.polygon(screen, (255, 120, 0), transform(flame))

def generate_terrain(width, scale=0.1, amplitude=2.0):
    xs = np.linspace(-width/2, width/2, width)
    #heights = amplitude * np.sin((xs + np.random.uniform(-10.0,10.0)) * scale) + 0.5 * np.sin((xs + 5) * scale * 3) + 2
    heights = np.ones_like(xs)
    return xs, heights

terrain_x, terrain_h = generate_terrain(2000)

def draw_terrain(screen):
    points = []
    for x, h in zip(terrain_x, terrain_h):
        px, py = world_to_screen(x, h)
        points.append((px, py))

    # cerrar hacia abajo
    points.append((WIDTH, HEIGHT))
    points.append((0, HEIGHT))

    pygame.draw.polygon(screen, (60, 60, 60), points)

def draw_background(screen):
    screen.blit(background, (0, 0))

def get_ground_height(x):
    # terrain_x y terrain_h vienen de generate_terrain
    return np.interp(x, terrain_x, terrain_h)

def render(state, thrust):
    screen.fill((0, 0, 0))

    # suelo
    draw_background(screen)
    draw_terrain(screen)
    draw_lander(screen, state, thrust[0])


    pygame.display.flip()