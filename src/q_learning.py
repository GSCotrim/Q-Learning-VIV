import random

from tqdm import tqdm

from environment import simulate_system, compute_reward


def generate_new_params(state, action):
    # Verifique se o state contém 7 parâmetros
    if len(state) != 7:
        raise ValueError(f"Estado esperado com 7 parâmetros, mas recebeu {len(state)} parâmetros.")

    new_params = list(state)  # Certifique-se de transformar a tupla em lista

    # Ajusta o parâmetro com base na ação escolhida (0 a 6 representam os índices dos parâmetros)
    if action < len(new_params):
        new_params[action] += random.uniform(-0.05, 0.05)  # Pequena modificação no parâmetro escolhido
    else:
        raise ValueError(f"Ação {action} inválida. O número de parâmetros é {len(new_params)}.")

    return tuple(new_params)  # Retorna uma tupla com 7 parâmetros


def compute_reward_and_next_state(new_params, initial_conditions, time_vector, target_response):
    simulated_response = simulate_system(new_params, initial_conditions, time_vector)
    reward = compute_reward(simulated_response, target_response, time_vector)
    next_state = tuple(new_params)
    return reward, next_state


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.5, epsilon_decay=0.9995, epsilon_min=0.05):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}

    def __select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(list(self.q_table[state].keys()))
        else:
            return max(self.q_table[state], key=self.q_table[state].get)

    def __update_q_value(self, state, action, reward, next_state):
        # Inicializa o próximo estado na tabela Q, se não estiver presente
        if next_state not in self.q_table:
            self.__initialize_q_values(next_state)

        # Encontra a melhor ação possível para o próximo estado
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)

        # Atualiza o valor da Q-table com a equação de Q-learning
        self.q_table[state][action] += self.alpha * (
                reward + self.gamma * self.q_table[next_state][best_next_action] - self.q_table[state][action])

    def __initialize_q_values(self, state):
        # Se o estado não existir na Q-table, inicializa suas ações com valor 0
        if state not in self.q_table:
            self.q_table[state] = {i: 0 for i in range(len(state))}

    def __take_action(self, state, initial_conditions, time_vector, target_response):
        # Inicializa os valores da Q-table para o estado atual, se necessário
        self.__initialize_q_values(state)

        # Seleciona uma ação com base na política epsilon-greedy
        action = self.__select_action(state)

        # Gera novos parâmetros baseados na ação
        new_params = generate_new_params(state, action)

        # Calcula a recompensa e o próximo estado com base nos novos parâmetros
        reward, next_state = compute_reward_and_next_state(new_params, initial_conditions, time_vector, target_response)

        # Atualiza a tabela Q com a recompensa obtida
        self.__update_q_value(state, action, reward, next_state)

        # Decai o epsilon para reduzir a exploração ao longo do tempo
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return next_state, reward

    def run(self, initial_conditions, time_vector, target_response, episodes=15000, max_steps=1000):
        total_rewards = []
        for _ in tqdm(range(episodes), desc="Q-Learning Progress"):
            total_reward = 0
            state = tuple(initial_conditions)
            for _ in range(max_steps):
                state, reward = self.__take_action(state, initial_conditions, time_vector, target_response)
                total_reward += reward
            total_rewards.append(total_reward)
        return total_rewards, self.q_table
