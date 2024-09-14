import matplotlib.pyplot as plt
import numpy as np
from simulation import environment_simulation

def simulate_and_plot(best_params, initial_conditions, num_states):
    U_Df0_values = np.linspace(1, 12, num_states)  # Exemplo de variação de U/Df0
    y_D_values = []

    for state in range(num_states):
        mu_opt, delta_opt, gamma_opt = best_params[state]

        # Simular o sistema com os parâmetros ótimos
        y = environment_simulation(state, mu_opt, delta_opt, gamma_opt, initial_conditions)

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
