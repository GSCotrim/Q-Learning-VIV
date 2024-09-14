import numpy as np
from scipy.integrate import odeint

# Função s(q) com base na equação (3.7)
def s(q, U, D, St, m):
    # Aplicando a equação fornecida
    return (D * q) / (4 * np.pi**2 * St**2 * U**2 * m)

# Função que define o sistema acoplado (corrigido para receber todos os parâmetros)
def coupled_oscillator(y, t, gamma, delta, mu, Omega_f, epsilon, U, D, St, m):
    # Variáveis de estado
    y1, dy1_dt, q, dq_dt = y

    # Equações diferenciais acopladas do modelo
    dy2_dt = -(2 * gamma + (gamma / mu) * Omega_f) * dy1_dt - delta ** 2 * y1 + s(q, U, D, St, m)
    dq2_dt = -epsilon * (q ** 2 - 1) * dq_dt - Omega_f * q + dy1_dt

    return [dy1_dt, dy2_dt, dq_dt, dq2_dt]

# Função de simulação do ambiente (corrigido)
def environment_simulation(state, gamma, delta, mu, Omega_f, epsilon, initial_conditions, U, D, St, m):
    t = np.linspace(0, 10, 100)  # Tempo de simulação
    # Solução numérica da ODE com as condições iniciais e os parâmetros ajustados
    solution = odeint(coupled_oscillator, initial_conditions, t, args=(gamma, delta, mu, Omega_f, epsilon, U, D, St, m))

    # Verificar se a solução é 1D ou 2D
    if solution.ndim == 2:
        # Se for 2D, extraímos as variáveis de estado
        y, dy_dt, q, dq_dt = solution.T
    else:
        # Se for 1D, extraímos diretamente
        y = solution[0]
        dy_dt = solution[1]
        q = solution[2]
        dq_dt = solution[3]

    return y
