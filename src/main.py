import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from q_learning import QLearningAgent
from src.environment import simulate_system_param


def load_data(file_path):
    data = pd.read_csv(file_path)
    time = data['time'].values
    target_response = data['target_response'].values
    return time, target_response


def plot_results(total_rewards, t, simulated_response_final_refined, target_response):
    plt.figure()
    plt.plot(t, simulated_response_final_refined[:, 0], label="Simulated y(t) (Cylinder Position)")
    plt.plot(t, target_response, label="Target Response")

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Simulated vs Target Oscillation (Coupled System)")
    plt.show()

    plt.figure()
    plt.plot(total_rewards, label="Total Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Evolution of Total Rewards During Training")
    plt.legend()
    plt.show()


def build_target_response(time):
    target_response = np.array((0.4, 10.0, .07e-2, 0.47746, 2.8274, 0.9381))
    reference_response = simulate_system_param(target_response, time)
    noise_lvl = 0.1
    reference_response[:, 0] += np.random.normal(0, noise_lvl, reference_response.shape[0])
    reference_response[:, 1] += np.random.normal(0, noise_lvl / .15, reference_response.shape[0])
    return reference_response[:, 0].copy()


def run_q_learning_process():
    time = np.linspace(0, 150, 250 + 1)
    target_response = build_target_response(time)

    agent = QLearningAgent(
        alpha=0.8,  ## learning rate
        gamma=0.95,  ## discount parameter
        epsilon=0.25,  ## exploration probability initially
        epsilon_decay=0.9995,  ## decay of exploration probability
        epsilon_min=0.05,  ## final asymptotic value of exploration probability
    )
    total_rewards, q_table = agent.run(time, target_response, episodes=1000, steps_per_ep=4)

    ijk = np.unravel_index(q_table.argmax(), q_table.shape)
    print(f"Final STATE: {ijk}")
    final_params = np.array([
        f'epsilon: {agent.params_range[0, ijk[0]]}; ',
        f'A: {agent.params_range[1, ijk[1]]}; ',
        f'Xi: {agent.params_range[2, ijk[2]]}; ',
        f'gamma: {agent.params_range[3, ijk[3]]}; ',
        f'mu: {agent.params_range[4, ijk[4]]}; ',
        f'delta: {agent.params_range[5, ijk[5]]}; ',
    ])
    print(f"Final PARAMS: {final_params}")

    simulated_response = simulate_system_param(final_params, time)
    plot_results(total_rewards, time, simulated_response, target_response)


if __name__ == "__main__":
    run_q_learning_process()
