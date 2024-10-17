from typing import Any

import numpy as np
from numpy import ndarray, dtype, floating
from scipy.fft import fft

initial_conditions = (0,) * 7

def compute_dydt(y, y_dot, delta, gamma, mu, s, xi):
    mu = max(mu, 1e-3)
    dydt = y_dot
    dydt_dot = -(2 * xi * delta + (gamma / mu)) * y_dot - (delta ** 2) * y + s
    return dydt, dydt_dot


def compute_dqdt(q, q_dot, epsilon, f):
    epsilon = max(epsilon, 1e-3)
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


def simulate_system(params, time):
    epsilon, delta, gamma, mu, s, f, xi = params
    mu = max(mu, 1e-6)

    y0, y_dot0, q0, q_dot0 = initial_conditions[:4]

    y = np.zeros_like(time)
    q = np.zeros_like(time)

    y[0], q[0] = y0, q0

    for i in range(1, len(time)):
        dt = time[i] - time[i - 1]

        dydt_dot = -(2 * xi * delta + (gamma / mu)) * y_dot0 - (delta ** 2) * y0 + s

        dqdt_dot = - epsilon * ((q0 ** 2) - 1) * q_dot0 - q0 + f

        y[i] = y0 + dydt_dot * dt
        q[i] = q0 + dqdt_dot * dt

        y0, y_dot0 = y[i], dydt_dot
        q0, q_dot0 = q[i], dqdt_dot

    return np.array([y, q]).T


def compute_dominant_frequency(response: np.ndarray, time: np.ndarray) -> ndarray[Any, dtype[floating[Any]]]:
    response = response[:, 0] if response.ndim > 1 else response
    response_fft = fft(response)
    freqs = np.fft.fftfreq(len(response), d=(time[1] - time[0]))
    dominant_freq = freqs[np.argmax(np.abs(response_fft))]
    return dominant_freq


def compute_amplitude_variance(simulated_response):
    return np.var(simulated_response[:, 0]) if simulated_response.ndim > 1 else np.var(simulated_response)


def compute_frequency_difference(simulated_response, target_response, time):
    simulated_freq = compute_dominant_frequency(simulated_response, time)
    target_freq = compute_dominant_frequency(target_response, time)
    return np.abs(simulated_freq - target_freq)


def compute_reward(simulated_response, target_response):
    simulated_y = simulated_response[:, 0]
    mse = np.mean((simulated_y - target_response) ** 2)
    reward = -mse
    return reward
