import numpy as np

def reward_function(state, action, y, dy_dt, desired_amplitude=1.0, amplitude_weight=1.0, derivative_weight=0.5):
    amplitude_error = -abs(desired_amplitude - np.max(np.abs(y)))
    derivative_penalty = -abs(dy_dt[-1])  # Penalizar grandes variações no final

    # Ajuste dinâmico dos pesos durante o treinamento
    amplitude_weight = max(0.5, amplitude_weight * 0.99)
    derivative_weight = min(1.0, derivative_weight * 1.01)

    reward = amplitude_weight * amplitude_error + derivative_weight * derivative_penalty
    return reward
