from typing import Any

import numpy as np
from numpy import ndarray, dtype, floating
from scipy.fft import fft
from scipy.integrate import odeint

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
    epsilon, delta, gamma, mu, s, f, xi = params
    mu = max(mu, 1e-6)

    y, y_dot, q, q_dot = yq

    dydt_dot = -(2 * xi * delta + (gamma / mu)) * y_dot - (delta ** 2) * y + s
    dqdt_dot = - epsilon * ((q ** 2) - 1) * q_dot - q + f

    return [y_dot, dydt_dot, q_dot, dqdt_dot]


def simulate_system(params, time):
    initial_state = [0.0, 0.0, 0.0, 0.0]
    try:
        system_dynamics = lambda yq, t: compute_system_dynamics(yq, params)
        solution = odeint(system_dynamics, initial_state, time, atol=1e-7, rtol=1e-5, hmin=1e-5)
        if np.any(np.isnan(solution)) or np.any(np.isinf(solution)):
            raise ValueError("A solução contém valores inválidos.")

        return solution
    except Exception as e:
        print(f"Erro na simulação com parâmetros {params}: {e}")
        return None


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


def compute_reward(simulated_response, target_response, time):
    if simulated_response is None:
        return -10000

    simulated_y = simulated_response[:, 0]
    mse = np.mean((simulated_y - target_response) ** 2)
    shape_penalty = np.mean(np.abs(np.gradient(simulated_y) - np.gradient(target_response)))
    simulated_freq = compute_dominant_frequency(simulated_y, time)
    target_freq = compute_dominant_frequency(target_response, time)
    freq_diff = np.abs(simulated_freq - target_freq)
    reward = -(mse + 0.5 * shape_penalty + 0.5 * freq_diff)

    return reward


