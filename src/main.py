import numpy as np

from q_learning import QLearningAgent
from src.target_response_generator import build_target_response
from src.environment import simulate_system_param
from src.results_plotter import plot_results


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
    total_rewards, q_table = agent.run(time, target_response, episodes=1500, steps_per_ep=10)

    ijk = np.unravel_index(q_table.argmax(), q_table.shape)
    print(f"Final STATE: {ijk}")

    final_params = np.array([
        agent.params_range[0, ijk[0]],
        agent.params_range[1, ijk[1]],
        agent.params_range[2, ijk[2]],
        agent.params_range[3, ijk[3]],
        agent.params_range[4, ijk[4]],
        agent.params_range[5, ijk[5]],
        agent.params_range[6, ijk[6]]
    ])

    params_map = {
        'epsilon_num': final_params[0],
        'a_num': final_params[1],
        'xi_num': final_params[2],
        'fluid_damping_coefficient_gamma': final_params[3],
        'nondimensional_mass_ratio_mu': final_params[4],
        'structure_reduced_angular_frequency_delta': final_params[5],
        'mass_number_M': final_params[6]
    }

    print(params_map)
    simulated_response = simulate_system_param(final_params, time)
    plot_results(total_rewards, time, simulated_response, target_response)


if __name__ == "__main__":
    run_q_learning_process()
