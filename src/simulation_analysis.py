import matplotlib.pyplot as plt
import numpy as np
from simulation import environment_simulation

def simulate_and_plot(best_params, initial_conditions, num_states):
    U_Df0_values = np.linspace(1, 12, num_states)  # Exemplo de variação de U/Df0
    y_D_values = []

    # Valores adicionais necessários para a simulação
    U = 1.0  # Velocidade do fluxo
    D = 1.0  # Diâmetro do cilindro
    St = 0.2  # Número de Strouhal típico para cilindros
    m = 1.0  # Massa total (inclui massa adicionada do fluido)

    for state in range(num_states):
        mu_opt, delta_opt, gamma_opt, Omega_f_opt, epsilon_opt = best_params[state]

        # Simular o sistema com os parâmetros ótimos, passando também U, D, St e m
        y = environment_simulation(state, gamma_opt, delta_opt, mu_opt, Omega_f_opt, epsilon_opt, initial_conditions, U, D, St, m)

        # Calcular a amplitude da oscilação (ŷ/D) baseada na solução simulada
        y_D_values.append(np.max(np.abs(y)))

    # Plotar o gráfico ŷ/D por U/Df₀
    plt.figure(figsize=(10, 6))
    plt.plot(U_Df0_values, y_D_values, 'o-', markersize=8, label="Amplitude de Oscilação")
    plt.xlabel("U / Df₀")
    plt.ylabel("ŷ / D")
    plt.title("Amplitude de Oscilação vs U / Df₀")
    plt.grid(True)
    plt.legend()
    plt.show()
