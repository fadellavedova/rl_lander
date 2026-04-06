import numpy as np
import sim_render
import pygame



def step(state, action, dt):
    x, z, vx, vz, theta, omega = state
    m = 1
    g = 10
    thrust, torque = action

    ax = (thrust / m) * np.sin(theta)
    az = (thrust / m) * np.cos(theta) - g

    vx += ax * dt
    vz += az * dt

    x += vx * dt
    z += vz * dt

    ground_z = sim_render.get_ground_height(x)

    if z-0.5 <= ground_z:
        z = ground_z + 0.5
        vz = 0
        collided = True
    else:
        collided = False

    omega += torque * dt
    theta += omega * dt

    new_state = x, z, vx, vz, theta, omega

    return new_state, collided



running = True
state = np.array([0, 9, -0.3, 0.1, 0, 0])  # x, z, vx, vz, theta, omega

dt = 0.02

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # acción random (placeholder)
    thrust = 9.0
    torque = 0.0

    state, collided = step(state, (thrust, torque), dt)

    if collided:
        if abs(state[3]) < 1.0 and abs(state[2]) < 1.0 and abs(state[4]) < 0.2:
            landed = True
        else:
            crashed = True

    sim_render.render(state, thrust)
    sim_render.clock.tick(60)

pygame.quit()