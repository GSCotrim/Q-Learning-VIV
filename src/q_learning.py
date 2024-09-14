import numpy as np
from tqdm import tqdm
from simulation import environment_simulation

# Definindo possíveis valores de parâmetros µ, δ, γ
mu_values = [0.5, 1.0, 2.0]  # Aumentando a variação
delta_values = [0.7, 1.0, 1.3]  # Valores mais variados
gamma_values = [0.3, 0.5, 0.7]  # Explorando valores diferentes

def q_learning(num_episodes, learning_rate, discount_factor, epsilon, initial_conditions, num_states, num_actions, convergence_threshold=1e-3, patience=50):
    Q_table = np.zeros((num_states, num_actions))
    reward_history = []
    best_params = np.zeros((num_states, 3))
    best_total_reward = -np.inf

    for episode in tqdm(range(num_episodes), desc="Q-Learning Progress"):
        state = np.random.randint(0, num_states)
        total_reward = 0
        for _ in range(100):
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(0, num_actions)
            else:
                action = np.argmax(Q_table[state])

            mu = mu_values[action % len(mu_values)]
            delta = delta_values[(action // len(mu_values)) % len(delta_values)]
            gamma = gamma_values[action // (len(mu_values) * len(delta_values))]

            y = environment_simulation(state, mu, delta, gamma, initial_conditions)
            reward = -abs(0.5 - np.max(np.abs(y)))
            next_state = (state + 1) % num_states

            Q_table[state, action] = Q_table[state, action] + learning_rate * (
                reward + discount_factor * np.max(Q_table[next_state]) - Q_table[state, action]
            )
            state = next_state
            total_reward += reward

        reward_history.append(total_reward)

        # Verificar convergência com base nas recompensas recentes
        if len(reward_history) > patience:
            recent_rewards = reward_history[-patience:]
            avg_recent_reward = np.mean(recent_rewards)

            if abs(avg_recent_reward - best_total_reward) < convergence_threshold:
                print(f"Convergência atingida no episódio {episode + 1}")
                break

            best_total_reward = avg_recent_reward

    for state in range(num_states):
        best_action = np.argmax(Q_table[state])
        mu_opt = mu_values[best_action % len(mu_values)]
        delta_opt = delta_values[(best_action // len(mu_values)) % len(delta_values)]
        gamma_opt = gamma_values[(best_action // (len(mu_values) * len(delta_values)))]
        best_params[state] = [mu_opt, delta_opt, gamma_opt]

    return Q_table, reward_history, best_params
