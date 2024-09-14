import numpy as np
from tqdm import tqdm
from simulation import environment_simulation

# Definindo possíveis valores de parâmetros µ, δ, γ, Omega_f, e epsilon
mu_values = [0.5, 1.0, 2.0]  # Variando µ
delta_values = [0.7, 1.0, 1.3]  # Valores variados para δ
gamma_values = [0.3, 0.5, 0.7]  # Variando γ
Omega_f_values = [0.5, 1.0, 1.5]  # Frequências angulares
epsilon_values = [0.1, 0.2, 0.3]  # Parâmetro de van der Pol

# Valores adicionais relacionados ao modelo
U = 1.0  # Exemplo de valor para a velocidade do fluxo
D = 1.0  # Diâmetro do cilindro
St = 0.2  # Número de Strouhal típico para cilindros
m = 1.0  # Massa total (cilindro + massa adicionada do fluido)

def q_learning(num_episodes, learning_rate, discount_factor, epsilon, initial_conditions, num_states, num_actions, convergence_threshold=1e-3, patience=50):
    Q_table = np.zeros((num_states, num_actions))
    reward_history = []
    best_params = np.zeros((num_states, 5))  # Ajustar para armazenar µ, δ, γ, Omega_f, ε
    best_total_reward = -np.inf

    for episode in tqdm(range(num_episodes), desc="Q-Learning Progress"):
        state = np.random.randint(0, num_states)
        total_reward = 0
        for _ in range(100):  # Ciclo para cada episódio
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(0, num_actions)
            else:
                action = np.argmax(Q_table[state])

            # Definindo os parâmetros baseados na ação
            mu = mu_values[action % len(mu_values)]
            delta = delta_values[(action // len(mu_values)) % len(delta_values)]
            gamma = gamma_values[(action // (len(mu_values) * len(delta_values))) % len(gamma_values)]
            Omega_f = Omega_f_values[(action // (len(mu_values) * len(delta_values) * len(gamma_values))) % len(Omega_f_values)]
            epsilon_vp = epsilon_values[(action // (len(mu_values) * len(delta_values) * len(gamma_values) * len(Omega_f_values))) % len(epsilon_values)]

            # Executando a simulação do ambiente com todos os parâmetros
            y = environment_simulation(state, gamma, delta, mu, Omega_f, epsilon_vp, initial_conditions, U, D, St, m)

            # Função de recompensa baseada na amplitude
            reward = -abs(0.5 - np.max(np.abs(y)))
            next_state = (state + 1) % num_states

            # Atualizando a tabela Q
            Q_table[state, action] = Q_table[state, action] + learning_rate * (
                reward + discount_factor * np.max(Q_table[next_state]) - Q_table[state, action]
            )
            state = next_state
            total_reward += reward

        reward_history.append(total_reward)

        # Verificando convergência
        if len(reward_history) > patience:
            recent_rewards = reward_history[-patience:]
            avg_recent_reward = np.mean(recent_rewards)

            if abs(avg_recent_reward - best_total_reward) < convergence_threshold:
                print(f"Convergência atingida no episódio {episode + 1}")
                break

            best_total_reward = avg_recent_reward

    # Salvando os melhores parâmetros para cada estado
    for state in range(num_states):
        best_action = np.argmax(Q_table[state])
        mu_opt = mu_values[best_action % len(mu_values)]
        delta_opt = delta_values[(best_action // len(mu_values)) % len(delta_values)]
        gamma_opt = gamma_values[(best_action // (len(mu_values) * len(delta_values))) % len(gamma_values)]
        Omega_f_opt = Omega_f_values[(best_action // (len(mu_values) * len(delta_values) * len(gamma_values))) % len(Omega_f_values)]
        epsilon_opt = epsilon_values[(best_action // (len(mu_values) * len(delta_values) * len(gamma_values) * len(Omega_f_values))) % len(epsilon_values)]

        best_params[state] = [mu_opt, delta_opt, gamma_opt, Omega_f_opt, epsilon_opt]

    return Q_table, reward_history, best_params
