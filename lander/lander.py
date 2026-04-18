import numpy as np
import sim_render
import pygame
import train
import keyboard


def step(state, action, dt):
    x_, z_, vx_, vz_, theta_, omega_ = state
    m = 1
    g = 1.625
    thrust, torque = action

    ax = (thrust / m) * np.sin(theta_)
    az = (thrust / m) * np.cos(theta_) - g
    
    vx = vx_ + ax * dt
    vz = vz_ + az * dt

    x = x_ + vx * dt
    z = z_ + vz * dt

    omega = omega_ + torque * dt
    theta = theta_ + omega * dt
    
    success = None
    if z-0.5 <= 1 or abs(x) > 10:
        if vz > -3 and abs(vx) < 0.5 and abs(theta) < 0.1 :
            success = True

        z = np.array([1]) + 0.5
        vz = np.array([0])
        collided = True
    else:
        collided = False


    new_state = np.array([x, z, vx, vz, theta, omega])

    #Reward

    r_vel = np.exp(-0.1*np.abs(vz + 0.5))
    r_angular = np.exp(-2*np.abs(theta))
    r_omega = np.exp(-np.abs(omega))
    r_dist = - 0.05 * np.sqrt(z**2 + x**2)
    r = r_vel + r_angular + r_omega + r_dist
    if success:
        r += 100
    elif collided:
        pass

    return new_state, r, collided, success



running = True
state = np.array([0.0, 12, 0.0, -1, 0.0, 0.0]).reshape(-1, 1)  # x, z, vx, vz, theta, omega

dt = 0.02
low_limit = [0, 15, 0.0, -1.3, 0.01, 0]
high_limit = [0, 15, 0.0, -0.8, -0.01, 0]
buffer_size = 1000000
eps = np.array([[0.01, 0.01]]).T
batch_size = 256

td3 = train.TD3(step, low_limit, high_limit, buffer_size, 6, 2, 64)

print(eps.shape)

td3.train(2000, 1000, dt, eps, batch_size, 0.99, 0.95, 0.001, 0.0001, 5, sim_render.render, x_init = None)

while True:
    try:
        input('Press Enter')
        for t in range(4000):
            u = td3.policy(state)
            state = step(state, u, dt)
    except:
        break
pygame.quit()