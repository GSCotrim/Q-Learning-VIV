from typing import List, Any

import numpy as np
from numpy import ndarray, dtype, floating
from scipy.fft import fft


def compute_dydt(y, y_dot, delta, gamma, mu, s, xi):
    dydt = y_dot
    dydt_dot = -(2 * xi * delta + (gamma / mu if mu != 0 else 0)) * y_dot - (delta ** 2) * y + s
    return dydt, dydt_dot


def compute_dqdt(q, q_dot, epsilon, f):
    dqdt = q_dot
    dqdt_dot = - epsilon * ((q ** 2) - 1) * q_dot - q + f
    return dqdt, dqdt_dot


def compute_system_dynamics(yq, params):
    y, y_dot, q, q_dot = yq
    if len(params) == 5:
        epsilon, delta, gamma, mu, xi = params
        s, f = 0, 0  # Default values for missing parameters
    else:
        epsilon, delta, gamma, mu, s, f, xi = params

    dydt, dydt_dot = compute_dydt(y, y_dot, delta, gamma, mu, s, xi)
    dqdt, dqdt_dot = compute_dqdt(q, q_dot, epsilon, f)

    return [dydt, dydt_dot, dqdt, dqdt_dot]


def simulate_system(params: List[float], initial_conditions: List[float], t: np.ndarray) -> np.ndarray:
    # Simulate the system using a simplified model
    y = initial_conditions[0] + params[0] * np.sin(t)
    y_dot = initial_conditions[1] + params[1] * np.cos(t)
    q = initial_conditions[2] + params[2] * np.sin(t)
    q_dot = initial_conditions[3] + params[3] * np.cos(t)
    return np.array([y, y_dot, q, q_dot]).T


def compute_dominant_frequency(response: np.ndarray, t: np.ndarray) -> ndarray[Any, dtype[floating[Any]]]:
    response = response[:, 0] if response.ndim > 1 else response  # Ensure response is 1D
    response_fft = fft(response)
    freqs = np.fft.fftfreq(len(response), d=(t[1] - t[0]))
    dominant_freq = freqs[np.argmax(np.abs(response_fft))]
    return dominant_freq


def compute_amplitude_variance(simulated_response):
    return np.var(simulated_response[:, 0]) if simulated_response.ndim > 1 else np.var(simulated_response)


def compute_frequency_difference(simulated_response, target_response, t):
    simulated_freq = compute_dominant_frequency(simulated_response, t)
    target_freq = compute_dominant_frequency(target_response, t)
    return np.abs(simulated_freq - target_freq)


def compute_reward(simulated_response, target_response, t):
    amplitude_variance = compute_amplitude_variance(simulated_response)
    freq_difference = compute_frequency_difference(simulated_response, target_response, t)
    reward = -(amplitude_variance + freq_difference)
    return reward
