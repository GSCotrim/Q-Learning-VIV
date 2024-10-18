# %debug
# import random
from tqdm import tqdm
import numpy as np
from environment import simulate_system, compute_reward

class QLearningAgent:
    def __init__(self, alpha, gamma, epsilon, epsilon_decay, epsilon_min, initial_conditions=(.5, 1e-10, 1e-10, 1e-10), q_table=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.initial_conditions = initial_conditions
        ## 2 params possible DISCRETE states
        self.n_params = 2
        self.n_steps = 20
        self.epsilon_range = np.arange(0.05, 1.00+1e-5, (1-0.05)/(self.n_steps-1) )
        self.A_range = np.arange(1.0, 20.0 + 1e-5, (20.0-1.0)/(self.n_steps-1) )
        self.params_range = np.stack((self.epsilon_range, self.A_range))
        ## actions and states
        self.n_actions = 2*self.n_params
        self.n_states = len(self.params_range)

        ## learning table Q
        self.q_table = self.__initialize_q_table() if q_table==None else q_table


    ## Possible actions up or down epsilon, up or down A (periodical on the discrete list)
    ## epsilon up, down, then A up, down
    def generate_new_2state(self, state, action):
        if state.size != self.n_params:
            raise ValueError(f"Estado esperado com 2 parâmetros, mas recebeu {len(state)} parâmetros.")
        if action < 0 or action > (2*self.n_params-1):
            raise ValueError(f"Ação esperada pertence a [0,3] mas ação={action}.")

        # copia
        new_state = state.copy()
        # direcao da acao (up or down)
        dir_action = 1 if action%self.n_params else -1
        # qual param
        which_param = action//self.n_params

        ## acao atualiza state # right up or left down
        new_state[which_param] = (
            (state[which_param]//self.n_steps * self.n_steps + (state[which_param]%self.n_steps+self.n_steps-1)%self.n_steps ) if (dir_action>0)
            else ( state[which_param]//self.n_steps * self.n_steps + (state[which_param]+1)%self.n_steps )
        )
        return new_state

    def __initialize_q_table(self):
        # wind up Q_table as very negative value since maximum is 0
        return np.zeros( ( self.n_steps, self.n_steps, self.n_actions) ) - 1

    def __select_action(self, state):
        # check for exploration possibility
        if np.random.uniform() < self.epsilon:
            return np.random.randint(0, 2*self.n_params)
        else: ## or choose based on maximum o q for that state
            q_state = self.q_table[ state[0], state[1]]
            return np.argmax(q_state)

    def __update_q_value(self, state, action, reward, next_state):
        ## get best next
        best_next_q_action = self.q_table[next_state[0],next_state[1]].max()
        self.q_table[ state[0], state[1], action] += self.alpha * \
            ( reward + self.gamma * best_next_q_action - self.q_table[state[0], state[1], action])

    def run(self, time, target_response, episodes=15000, steps_per_ep=1000):
        total_rewards = []

        for _ in tqdm(range(episodes), desc="Q-Learning Progress"):
            ## initialize reward
            total_reward = 0
            ## random initial params
            state = np.random.randint(0, self.n_steps, self.n_params)

            for _ in range(steps_per_ep):
                action = self.__select_action(state)
                ## compute next state
                new_state = self.generate_new_2state(state, action)
                ## convert it to a set of params
                new_params = np.array( [
                    self.params_range[0, new_state[0]],
                    self.params_range[1, new_state[1]]
                ] )
                ## compute new solutions and reward
                simulated_response = simulate_system_2param(new_params, time)
                reward = compute_reward(simulated_response, target_response)
                ## store accumulated reward
                total_reward += reward
                ## update q_value
                self.__update_q_value(state, action, reward, new_state)
                ## jumpt to next_state
                state = new_state.copy()
                # ## update epsilon
                # self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            ## episodic reward store
            total_rewards.append(total_reward)

        return total_rewards, self.q_table

