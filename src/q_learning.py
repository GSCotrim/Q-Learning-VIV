import random

import numpy as np
from tqdm import tqdm

from environment import simulate_system, compute_reward


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def __select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(list(self.q_table[state].keys()))
        else:
            return max(self.q_table[state], key=self.q_table[state].get)

    def __update_q_value(self, state, action, reward, next_state):
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        self.q_table[state][action] += self.alpha * (reward + self.gamma * self.q_table[next_state][best_next_action] - self.q_table[state][action])

    def __initialize_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = {i: 0 for i in range(4)}

    def __determine_next_state(self, new_params):
        next_state = tuple(new_params)
        if next_state not in self.q_table:
            self.q_table[next_state] = {i: 0 for i in range(4)}
        return next_state

    def __compute_reward_and_next_state(self, new_params, initial_conditions, t, target_response):
        simulated_response = simulate_system(new_params, initial_conditions, t)
        reward = compute_reward(simulated_response, target_response, t)
        next_state = self.__determine_next_state(new_params)
        return reward, next_state

    def __take_action(self, state, initial_conditions, t, target_response):
        self.__initialize_q_values(state)
        action = self.__select_action(state)
        new_params = self.__generate_new_params(state, action)
        reward, next_state = self.__compute_reward_and_next_state(new_params, initial_conditions, t, target_response)
        self.__update_q_value(state, action, reward, next_state)
        return next_state, reward

    def __run_episode(self, initial_conditions, t, target_response):
        state = self.__initialize_state(initial_conditions)
        total_reward = 0
        done = False
        while not done:
            state, reward = self.__take_action(state, initial_conditions, t, target_response)
            total_reward += reward
            done = self.__is_done(total_reward)
        return total_reward

    def __initialize_state(self, initial_conditions):
        return tuple(initial_conditions)

    def __is_done(self, total_reward):
        return total_reward > -1.0

    def __generate_new_params(self, state, action):
        new_params = list(state)
        new_params[action] += random.uniform(-0.05, 0.05)
        return new_params

    def run(self, initial_conditions, t, target_response, episodes=5000):
        total_rewards = []
        with tqdm(total=episodes, desc="Q-Learning Progress", dynamic_ncols=True) as pbar:
            for _ in range(episodes):
                total_reward = self.__run_episode(initial_conditions, t, target_response)
                total_rewards.append(total_reward)
                pbar.update(1)
        return np.array(total_rewards), self.q_table
