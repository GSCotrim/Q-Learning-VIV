import numpy as np
import pandas as pd

def generate_mock_data(num_samples=100, seed=42, output_path='/home/gscotrim/Documents/POLI/TCC/rl/Q-Learning-VIV/src/data/dados_viv_simulados.csv'):
    np.random.seed(seed)
    t = np.linspace(0, 10, num_samples)  # Adiciona a coluna 't'
    velocidade_fluxo = np.linspace(0.3, 2.5, num_samples)
    amplitude_oscilacao = 0.8 * np.sin(2 * np.pi * velocidade_fluxo / 5) + np.random.normal(0, 0.05, num_samples)
    frequencia_shedding = 0.2 * velocidade_fluxo + np.random.normal(0, 0.01, num_samples)
    coeficiente_sustentacao = 0.4 * np.cos(2 * np.pi * velocidade_fluxo / 3) + np.random.normal(0, 0.02, num_samples)
    numero_reynolds = 1e5 * velocidade_fluxo / 0.1
    target_response = 0.5 * velocidade_fluxo + np.random.normal(0, 0.1, num_samples)  # Example target response

    # Example initial conditions
    initial_conditions = [0.5, 0.1, 0.2, 0.3]

    dados_viv = pd.DataFrame({
        't': t,  # Adiciona a coluna 't'
        'Velocidade_Fluxo (m/s)': velocidade_fluxo,
        'Amplitude_Oscilacao (m)': amplitude_oscilacao,
        'Frequencia_Shedding (Hz)': frequencia_shedding,
        'Coeficiente_Sustentacao (Clv)': coeficiente_sustentacao,
        'Numero_Reynolds (Re)': numero_reynolds,
        'target_response': target_response,
        'initial_conditions': [initial_conditions] * num_samples  # Add initial_conditions column
    })

    try:
        dados_viv.to_csv(output_path, index=False)
        print(f"Dados salvos com sucesso em {output_path}")
    except Exception as e:
        print(f"Erro ao salvar os dados: {e}")

    return dados_viv

if __name__ == "__main__":
    generate_mock_data()