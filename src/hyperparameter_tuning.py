import numpy as np
from q_learning import q_learning


def refine_hyperparameters(num_states, num_actions, initial_conditions):
    best_hyperparams = {}
    best_reward = -np.inf

    # Listas de hiperparâmetros para testar
    learning_rates = [0.05, 0.1, 0.15]
    discount_factors = [0.9, 0.95, 0.99]
    epsilons = [0.05, 0.1, 0.15]

    for lr in learning_rates:
        for df in discount_factors:
            for eps in epsilons:
                # Executa Q-Learning com os hiperparâmetros atuais
                Q_table, reward_history, _ = q_learning(
                    num_episodes=500,
                    learning_rate=lr,
                    discount_factor=df,
                    epsilon=eps,
                    initial_conditions=initial_conditions,
                    num_states=num_states,
                    num_actions=num_actions
                )

                # Calcula a média das recompensas das últimas 100 iterações
                avg_reward = np.mean(reward_history[-100:])

                # Verifica se é a melhor recompensa encontrada
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    best_hyperparams = {'lr': lr, 'df': df, 'epsilon': eps}

    print(f"Melhores Hiperparâmetros: {best_hyperparams} com Recompensa Média: {best_reward}")
    return best_hyperparams


def test_robustness(best_hyperparams, num_states, num_actions):
    initial_conditions_list = [
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0, -1.0],
        [0.5, -0.5, 0.5, -0.5],
        [10.0, 10.0, 10.0, 10.0],
        [-10.0, -10.0, -10.0, -10.0],
        [0.1, 0.1, 0.1, 0.1],
        [-0.1, -0.1, -0.1, -0.1]
    ]

    for initial_conditions in initial_conditions_list:
        Q_table, reward_history, best_params = q_learning(
            1000,
            best_hyperparams['lr'],
            best_hyperparams['df'],
            best_hyperparams['epsilon'],
            initial_conditions,
            num_states,
            num_actions
        )

        # plot_q_table(Q_table)  # Removido para evitar geração de heatmaps desnecessários

