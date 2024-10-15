import os

import numpy as np
import pandas as pd


def generate_harmonic_data(num_samples=100, seed=42, output_path=os.path.abspath("data/dados_viv_simulados.csv")):
    np.random.seed(seed)
    t = np.linspace(0, 10, num_samples)  # Tempo de 0 a 10 segundos

    # Gera dados de oscilação harmônica simples (padrão senoidal)
    amplitude_oscilacao = 0.8 * np.sin(2 * np.pi * 0.5 * t)  # Seno de 0.5 Hz

    # Define a resposta-alvo para o Q-Learning igual à oscilação gerada
    target_response = amplitude_oscilacao

    # Condições iniciais para o agente de RL (7 parâmetros)
    initial_conditions = [0.5, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0]  # Certifique-se de incluir os 7 parâmetros

    # Crie o DataFrame
    data = pd.DataFrame({
        't': t,
        'Amplitude_Oscilacao (m)': amplitude_oscilacao,
        'target_response': target_response,
        'initial_conditions': [initial_conditions] * num_samples
    })

    try:
        data.to_csv(output_path, index=False)
        print(f"Dados salvos com sucesso em {output_path}")
    except Exception as e:
        print(f"Erro ao salvar os dados: {e}")

    return data


if __name__ == "__main__":
    generate_harmonic_data()
