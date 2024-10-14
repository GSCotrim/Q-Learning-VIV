from typing import Any

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
    if len(params) == 4:
        epsilon, delta, gamma, mu = params
        s, f, xi = 0, 0, 0
    elif len(params) == 5:
        epsilon, delta, gamma, mu, xi = params
        s, f = 0, 0
    elif len(params) == 7:
        epsilon, delta, gamma, mu, s, f, xi = params
    else:
        raise ValueError(f"Número incorreto de parâmetros. Esperado 4, 5 ou 7, mas recebeu {len(params)}.")

    dydt, dydt_dot = compute_dydt(y, y_dot, delta, gamma, mu, s, xi)
    dqdt, dqdt_dot = compute_dqdt(q, q_dot, epsilon, f)

    return [dydt, dydt_dot, dqdt, dqdt_dot]


def simulate_system(params: list, initial_conditions: list, time_vector: np.ndarray,
                    target_response: np.ndarray) -> np.ndarray:
    if isinstance(initial_conditions[0], (list, tuple)) and len(initial_conditions[0]) == 4:
        y0, y_dot0, q0, q_dot0 = initial_conditions[0]
    else:
        raise ValueError("initial_conditions tem um formato inesperado. Esperava uma lista ou tupla de 4 valores.")

    y0 = target_response[0]
    y = np.zeros_like(time_vector)
    y_dot = np.zeros_like(time_vector)
    q = np.zeros_like(time_vector)
    q_dot = np.zeros_like(time_vector)

    y[0], y_dot[0], q[0], q_dot[0] = y0, y_dot0, q0, q_dot0
    epsilon, delta, gamma, mu, s, f, xi = params

    for i in range(1, len(time_vector)):
        dydt, dydt_dot = compute_dydt(y[i - 1], y_dot[i - 1], delta, gamma, mu, s, xi)
        dqdt, dqdt_dot = compute_dqdt(q[i - 1], q_dot[i - 1], epsilon, f)

        y[i] = y[i - 1] + dydt * (time_vector[i] - time_vector[i - 1])
        y_dot[i] = y_dot[i - 1] + dydt_dot * (time_vector[i] - time_vector[i - 1])
        q[i] = q[i - 1] + dqdt * (time_vector[i] - time_vector[i - 1])
        q_dot[i] = q_dot[i - 1] + dqdt_dot * (time_vector[i] - time_vector[i - 1])

    return np.array([y, y_dot, q, q_dot]).T


def compute_dominant_frequency(response: np.ndarray, time_vector: np.ndarray) -> ndarray[Any, dtype[floating[Any]]]:
    response = response[:, 0] if response.ndim > 1 else response
    response_fft = fft(response)
    freqs = np.fft.fftfreq(len(response), d=(time_vector[1] - time_vector[0]))
    dominant_freq = freqs[np.argmax(np.abs(response_fft))]
    return dominant_freq


def compute_amplitude_variance(simulated_response):
    return np.var(simulated_response[:, 0]) if simulated_response.ndim > 1 else np.var(simulated_response)


def compute_frequency_difference(simulated_response, target_response, time_vector):
    simulated_freq = compute_dominant_frequency(simulated_response, time_vector)
    target_freq = compute_dominant_frequency(target_response, time_vector)
    return np.abs(simulated_freq - target_freq)


def compute_reward(simulated_response, target_response, time_vector):
    mse = np.mean((simulated_response[:, 0] - target_response) ** 2)
    amplitude_difference = np.abs(np.mean(simulated_response[:, 0]) - np.mean(target_response))
    freq_difference = compute_frequency_difference(simulated_response, target_response, time_vector)
    reward = -(mse + 1.0 * amplitude_difference + 0.5 * freq_difference)
    return reward
