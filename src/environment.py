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


def simulate_system(params, initial_conditions, time_vector):
    # Certifique-se de que initial_conditions contém pelo menos 4 valores (y, y_dot, q, q_dot)
    if len(initial_conditions) < 4:
        raise ValueError(f"Condições iniciais devem conter pelo menos 4 valores, mas recebeu {len(initial_conditions)}")

    # Extrair as condições iniciais relevantes (ignorando os outros parâmetros extras)
    y0, y_dot0, q0, q_dot0 = initial_conditions[:4]

    # Inicializar os vetores para armazenar os dados ao longo do tempo
    y = np.zeros_like(time_vector)
    y_dot = np.zeros_like(time_vector)
    q = np.zeros_like(time_vector)
    q_dot = np.zeros_like(time_vector)

    # Definir as condições iniciais
    y[0], y_dot[0], q[0], q_dot[0] = y0, y_dot0, q0, q_dot0

    # Verificar se params tem 7 valores (epsilon, delta, gamma, mu, s, f, xi)
    if len(params) != 7:
        raise ValueError(f"Esperado 7 parâmetros, mas recebeu {len(params)}")

    # Extrair os parâmetros
    epsilon, delta, gamma, mu, s, f, xi = params

    # Simular a dinâmica ao longo do tempo
    for i in range(1, len(time_vector)):
        dydt, dydt_dot = compute_dydt(y[i - 1], y_dot[i - 1], delta, gamma, mu, s, xi)
        dqdt, dqdt_dot = compute_dqdt(q[i - 1], q_dot[i - 1], epsilon, f)

        dt = time_vector[i] - time_vector[i - 1]
        y[i] = y[i - 1] + dydt * dt
        y_dot[i] = y_dot[i - 1] + dydt_dot * dt
        q[i] = q[i - 1] + dqdt * dt
        q_dot[i] = q_dot[i - 1] + dqdt_dot * dt

    # Retornar a trajetória simulada
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

    # Exclui temporariamente a penalidade de forma
    reward = -(mse + 0.5 * amplitude_difference)
    return reward

