import numpy as np

from src.environment import simulate_system_param


def build_target_response(time):
    target_response = np.array((0.4, 10.0, .07e-2, 0.47746, 2.8274, 0.9381, 0.01680))
    reference_response = simulate_system_param(target_response, time)
    noise_lvl = 0.1
    reference_response[:, 0] += np.random.normal(0, noise_lvl, reference_response.shape[0])
    reference_response[:, 1] += np.random.normal(0, noise_lvl / .15, reference_response.shape[0])
    return reference_response[:, 0].copy()
