from scipy.integrate import odeint
import numpy as np

# Modelo acoplado: van der Pol + oscilador estrutural
def coupled_oscillator(y, t, mu, delta, gamma):
    y1, dy1_dt, q, dq_dt = y
    dy2_dt = q - 2 * delta * gamma * dy1_dt - delta**2 * y1
    dq2_dt = mu * (1 - q**2) * dq_dt - q
    return [dy1_dt, dy2_dt, dq_dt, dq2_dt]

# Função de simulação do ambiente com parâmetros µ, δ, γ
def environment_simulation(state, mu, delta, gamma, initial_conditions):
    t = np.linspace(0, 10, 100)  # Tempo de simulação
    solution = odeint(coupled_oscillator, initial_conditions, t, args=(mu, delta, gamma))

    y = solution[:, 0]  # Considerar o movimento y para a amplitude de oscilação
    return y
