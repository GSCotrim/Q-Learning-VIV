from q_learning import q_learning
from hyperparameter_tuning import refine_hyperparameters, test_robustness
from plot_results import plot_reward_history, plot_q_table, save_plot
from simulation_analysis import simulate_and_plot
from utils import save_q_table

if __name__ == "__main__":
    num_episodes = 1000
    learning_rate = 0.1
    discount_factor = 0.9
    epsilon = 0.1
    initial_conditions = [0.0, 0.0, 1.0, 0.0]
    num_states = 10
    num_actions = 9

    # Refinamento de hiperparâmetros
    best_hyperparams = refine_hyperparameters(num_states, num_actions, initial_conditions)

    # Teste de robustez com novas condições extremas
    test_robustness(best_hyperparams, num_states, num_actions)

    # Executar Q-Learning com os melhores hiperparâmetros
    Q_table, reward_history, best_params = q_learning(
        num_episodes,
        best_hyperparams['lr'],
        best_hyperparams['df'],
        best_hyperparams['epsilon'],
        initial_conditions,
        num_states,
        num_actions
    )

    # Plotar e salvar resultados do Q-Learning
    plot_reward_history(reward_history)
    plot_q_table(Q_table)
    save_q_table(Q_table, '/home/gscotrim/PycharmProjects/Q-LearningVIV/data/Q_table.npy')
    save_plot('/home/gscotrim/PycharmProjects/Q-LearningVIV/data/q_table.png')
    save_plot('/home/gscotrim/PycharmProjects/Q-LearningVIV/data/reward_history.png')

    # Simular e gerar gráfico ŷ/D por U/Df₀ com os parâmetros otimizados
    simulate_and_plot(best_params, initial_conditions, num_states)
