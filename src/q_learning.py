import numpy as np
from tqdm import tqdm

from environment import compute_reward, simulate_system_param


class QLearningAgent:
    def __init__(self, alpha, gamma, epsilon, epsilon_decay, epsilon_min, initial_conditions=(.5, 1e-10, 1e-10, 1e-10),
                 q_table=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.initial_conditions = initial_conditions
        self.n_params = 5
        self.n_steps = 20
        self.epsilon_range = np.arange(0.05, 1.00 + 1e-5, (1 - 0.05) / (self.n_steps - 1))
        self.A_range = np.arange(1.0, 20.0 + 1e-5, (20.0 - 1.0) / (self.n_steps - 1))
        self.xi_num_range = np.linspace(5e-5, 5e-3, self.n_steps)
        self.fluid_damping_coefficient_gamma_range = np.arange(0.05, 1.0 + 1e-5, (1.0 - 0.05) / (self.n_steps - 1))
        self.nondimensional_mass_ratio_mu_range = np.arange(0.1, 5.0 + 1e-5, (5.0 - 0.1) / (self.n_steps - 1))
        self.params_range = np.stack((self.epsilon_range,
                                      self.A_range,
                                      self.xi_num_range,
                                      self.fluid_damping_coefficient_gamma_range,
                                      self.nondimensional_mass_ratio_mu_range))
        self.n_actions = 2 * self.n_params
        self.n_states = len(self.params_range)
        self.q_table = self.__initialize_q_table() if q_table is None else q_table

    def generate_new_state(self, state, action):
        if state.size != self.n_params:
            raise ValueError(f"Estado esperado com {self.n_params} parâmetros, mas recebeu {len(state)} parâmetros.")
        if action < 0 or action > (2 * self.n_params - 1):
            raise ValueError(f"Ação esperada pertence a [0,{2 * self.n_params - 1}] mas ação={action}.")

        new_state = state.copy()
        dir_action = 1 if action % 2 else -1
        which_param = action // 2

        new_state[which_param] = (
            (state[which_param] // self.n_steps * self.n_steps + (
                    state[which_param] % self.n_steps + self.n_steps - 1) % self.n_steps)
            if (dir_action > 0)
            else (state[which_param] // self.n_steps * self.n_steps + (state[which_param] + 1) % self.n_steps)
        )
        return new_state

    def __initialize_q_table(self):
        dimensions = (self.n_steps,) * self.n_params + (self.n_actions,)
        return np.zeros(dimensions) - 1

    def __select_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.randint(0, 2 * self.n_params)
        else:
            q_state = self.q_table[tuple(state[:self.n_params])]
            return np.argmax(q_state)

    def __update_q_value(self, state, action, reward, next_state):
        best_next_q_action = self.q_table[tuple(next_state[:self.n_params])].max()
        self.q_table[tuple(state[:self.n_params]) + (action,)] += self.alpha * (
                reward + self.gamma * best_next_q_action - self.q_table[tuple(state[:self.n_params]) + (action,)])

    def __get_new_params(self, new_state):
        new_params = np.array([
            self.params_range[i, new_state[i]] for i in range(self.n_params)
        ])
        return new_params

    def run(self, time, target_response, episodes=15000, steps_per_ep=1000):
        total_rewards = []

        for _ in tqdm(range(episodes), desc="Q-Learning Progress"):
            total_reward = 0
            state = np.random.randint(0, self.n_steps, self.n_params)

            for _ in range(steps_per_ep):
                action = self.__select_action(state)
                new_state = self.generate_new_state(state, action)
                new_params = self.__get_new_params(new_state)
                simulated_response = simulate_system_param(new_params, time)
                reward = compute_reward(simulated_response, target_response)
                total_reward += reward
                self.__update_q_value(state, action, reward, new_state)
                state = new_state.copy()

            total_rewards.append(total_reward)

        return total_rewards, self.q_table
