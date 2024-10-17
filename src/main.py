import os
import matplotlib.pyplot as plt
import pandas as pd
from environment import simulate_system
from q_learning import QLearningAgent


def load_data(file_path):
    data = pd.read_csv(file_path)
    time = data['time'].values
    target_response = data['target_response'].values
    return time, target_response


def plot_results(total_rewards, t, simulated_response_final_refined, target_response):
    plt.figure(figsize=(10, 6))
    plt.plot(t, simulated_response_final_refined[:, 0], label="Simulated y(t) (Cylinder Position)")
    # plt.plot(t, simulated_response_final_refined[:, 1], label="Simulated q(t) (Fluid Dynamics)")
    plt.plot(t, target_response, label="Target Response")

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Simulated vs Target Oscillation (Coupled System)")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(total_rewards, label="Total Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Evolution of Total Rewards During Training")
    plt.legend()
    plt.show()


def run_q_learning_process():
    file_path = os.path.abspath("data/dados_viv_simulados.csv")
    time, target_response = load_data(file_path)

    agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.1, epsilon_decay=0.9995, epsilon_min=0.05, initial_conditions=(0,) * 7)
    total_rewards, q_table = agent.run(time, target_response, episodes=200)

    final_params = max(q_table, key=lambda x: max(q_table[x].values()))
    print(f"Final Parameters: {final_params}")

    simulated_response = simulate_system(final_params, time)
    plot_results(total_rewards, time, simulated_response, target_response)

if __name__ == "__main__":
    run_q_learning_process()
