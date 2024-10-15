import os
import matplotlib.pyplot as plt
import pandas as pd
from environment import simulate_system
from q_learning import QLearningAgent


def load_data(file_path):
    data = pd.read_csv(file_path)
    initial_conditions = eval(data['initial_conditions'][0])  # Como há uma única condição inicial
    t = data['t'].values
    target_response = data['target_response'].values
    return initial_conditions, t, target_response


def plot_results(total_rewards, t, simulated_response_final_refined, target_response):
    # Plot da resposta simulada vs resposta alvo
    plt.figure(figsize=(10, 6))
    plt.plot(t, simulated_response_final_refined[:, 0], label="Simulated Response")
    plt.plot(t, target_response, label="Target Response")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Simulated vs Target Oscillation")
    plt.show()

    # Plot da evolução das recompensas
    plt.figure(figsize=(10, 6))
    plt.plot(total_rewards, label="Total Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Evolution of Total Rewards During Training")
    plt.legend()
    plt.show()


def run_q_learning_process():
    file_path = os.path.abspath("data/dados_viv_simulados.csv")
    initial_conditions, t, target_response = load_data(file_path)

    agent = QLearningAgent(alpha=0.05, gamma=0.95, epsilon=0.2)
    total_rewards, q_table = agent.run(initial_conditions, t, target_response, episodes=500)

    final_params = max(q_table, key=lambda x: max(q_table[x].values()))
    print(f"Final Parameters: {final_params}")

    simulated_response = simulate_system(final_params, initial_conditions, t)
    plot_results(total_rewards, t, simulated_response, target_response)

if __name__ == "__main__":
    run_q_learning_process()
