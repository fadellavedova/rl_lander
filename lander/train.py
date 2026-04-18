import nets
import copy
import numpy as np


class TD3():

    def __init__(self, environment, low_limits, high_limits, buffer_size, n_s = 1, n_a = 1, hidden = 32):

        # Redes
        self.Q1 = nets.DDPG_Critic(n_s = n_s, n_a = n_a, hidden = hidden)
        self.Q1_target = copy.deepcopy(self.Q1)

        self.Q2 = nets.DDPG_Critic(n_s = n_s, n_a = n_a, hidden = hidden)
        self.Q2_target = copy.deepcopy(self.Q2)

        self.pi = nets.DDPG_Actor(n_s = n_s, n_a = n_a, hidden = hidden)
        self.pi_target = copy.deepcopy(self.pi)

        self.rate = 0.5
        self.buffer = nets.ReplayBuffer(int(buffer_size*self.rate), n_a, n_s)
        self.great_buffer = nets.ReplayBuffer(int(buffer_size*(1-self.rate)), n_a, n_s)

        # Variables de simulacion
        self.step = environment
        self.low_limits = low_limits
        self.high_limits = high_limits

        # Métricas
        self.hist_reward = []
        self.Q_std_history = []
        self.Q_mean_history = []
        self.dQda_history = []

    def train(self, episodes, steps_per_episode, dt, eps, batch_size, gamma, tau, lr_q, lr_pi, delay, render_callback=None, x_init=None):
        for i in range(episodes):
            if x_init is None:
                x = np.random.uniform(self.low_limits, self.high_limits)
            else:
                if x_init.shape != (6, 1):
                    print("Inadequate initial condition")
                    return
                x = x_init.copy()
            accum_reward = 0
            Q_mean = None
            for t in range(steps_per_episode):
                u = self.pi.forward(x.reshape(-1,1))
                u = np.clip(u + np.random.normal(0, eps, size = u.shape), -1, 1)
                action = u.copy()
                action[0] = action[0]*20
                x_next, r, done, success = self.step(x, action, dt)
                accum_reward += r
                if success:
                    print("Success")

                self.buffer.add(x.reshape(-1), u.reshape(-1), r, x_next.reshape(-1), done)
                self.great_buffer.add(x.reshape(-1), u.reshape(-1), r, x_next.reshape(-1), done)
                x = x_next.copy()

                if render_callback is not None:
                    render_callback(x.reshape(-1), u.reshape(-1))

                if self.buffer.size > batch_size:
                    
                    batch = self.buffer.sample(batch_size)
                    if self.great_buffer.size > batch_size*(1-self.rate):
                        batch = self.buffer.sample(int(batch_size*self.rate))
                        batch_g = self.great_buffer.sample(int(batch_size*(1-self.rate)))
                        batch = batch + batch_g

                    #Critic
                    self.Q1.forward(batch[0].T, batch[1].T)
                    self.Q2.forward(batch[0].T, batch[1].T)

                    
                    actions = self.pi_target.forward(batch[3].T)

                    smooth_noise = np.random.normal(0, 0.2, size=actions.shape)
                    smooth_noise = np.clip(smooth_noise, -0.5, 0.5)
                    actions_smooth = np.clip(actions + smooth_noise, -1, 1)

                    q1_next = self.Q1_target.forward(batch[3].T, actions_smooth)
                    q2_next = self.Q2_target.forward(batch[3].T, actions_smooth)

                    q_next = np.min(np.vstack([q1_next, q2_next]), axis=0)

                    td_target = batch[2].T + gamma * (1 - batch[4].T) * q_next
                    td_target = np.clip(td_target, -100, 100)

                    q_target = td_target

                    self.Q1.backward(q_target)
                    self.Q1.update_weights(lr_q)

                    self.Q2.backward(q_target)
                    self.Q2.update_weights(lr_q)

                    # Actor
                    if t % delay == 0:
                        a = self.pi.forward(batch[0].T)




                        self.Q1.forward(batch[0].T, a)
                        dQ_da = self.Q1.grad_a()
                        dQ_da = np.clip(dQ_da, -100, 100)

                        self.dQda_history.append(np.mean(dQ_da))

                        self.pi.backward(dQ_da)
                        self.pi.update_weights(lr_pi)

                        self.pi_target.soft_parameter_update(self.pi, tau)
                        self.Q1_target.soft_parameter_update(self.Q1, tau)
                        self.Q2_target.soft_parameter_update(self.Q2, tau)


                    q_for_metric = self.Q1.predict(batch[3].T, actions)
                    Q_mean = np.mean(q_for_metric)
                    Q_std = np.std(q_for_metric)

                    self.Q_std_history.append(Q_std)
                    self.Q_mean_history.append(Q_mean)

                if done:
                    break

            if accum_reward < 300:
                self.great_buffer.clean(t)

            print(f"[{i}] R = {accum_reward}, Q_mean = {Q_mean}")

            self.hist_reward.append(accum_reward)


    
    def policy(self, s):
        return self.pi.predict(s)

    

