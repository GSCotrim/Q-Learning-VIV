import random

from tqdm import tqdm

from environment import simulate_system, compute_reward


def generate_new_params(state, action):
    if len(state) != 7:
        raise ValueError(f"Estado esperado com 7 parâmetros, mas recebeu {len(state)} parâmetros.")

    new_params = list(state)
    if action < len(new_params):
        new_params[action] += random.uniform(-0.01, 0.01)
    else:
        raise ValueError(f"Ação {action} inválida. O número de parâmetros é {len(new_params)}.")

    return tuple(new_params)


def compute_reward_and_next_state(new_params, time, target_response):
    simulated_response = simulate_system(new_params, time)
    reward = compute_reward(simulated_response, target_response)
    next_state = tuple(new_params)
    return reward, next_state


class QLearningAgent:
    def __init__(self, alpha, gamma, epsilon, epsilon_decay, epsilon_min, initial_conditions):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.initial_conditions = initial_conditions
        self.q_table = {}

    def __select_action(self, state):
        if state not in self.q_table:
            self.__initialize_q_values(state)
        if random.uniform(0, 1) < self.epsilon or not self.q_table[state]:
            return random.choice(range(len(state)))
        else:
            return max(self.q_table[state], key=self.q_table[state].get)

    def __update_q_value(self, state, action, reward, next_state):
        if next_state not in self.q_table:
            self.__initialize_q_values(next_state)

        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        self.q_table[state][action] += self.alpha * (reward + self.gamma * self.q_table[next_state][best_next_action] - self.q_table[state][action])

    def __initialize_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = {i: random.uniform(-0.01, 0.01) for i in range(len(state))}

    def __take_action(self, state, time, target_response):
        self.__initialize_q_values(state)
        action = self.__select_action(state)

        try:
            new_params = generate_new_params(state, action)
            reward, next_state = compute_reward_and_next_state(new_params, time, target_response)
            self.__update_q_value(state, action, reward, next_state)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            return next_state, reward
        except Exception as e:
            print(f"Error during action: {e}")

            return state, -1

    def run(self, time, target_response, episodes=15000):
        total_rewards = []
        state = self.initial_conditions

        for _ in tqdm(range(episodes), desc="Q-Learning Progress"):
            total_reward = 0

            for _ in range(1000):
                action = self.__select_action(state)
                new_params = generate_new_params(state, action)
                simulated_response = simulate_system(new_params, time)
                reward = compute_reward(simulated_response, target_response)
                total_reward += reward
                self.__update_q_value(state, action, reward, new_params)
                state = new_params

            total_rewards.append(total_reward)

        return total_rewards, self.q_table

