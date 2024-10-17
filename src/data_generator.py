import os

import numpy as np
import pandas as pd


def generate_harmonic_data(num_samples=100, seed=42, output_path=os.path.abspath("data/dados_viv_simulados.csv")):
    np.random.seed(seed)
    time = np.linspace(0, 10, num_samples)
    oscilacao_pura = 0.8 * np.sin(2 * np.pi * 0.5 * time)
    target_response = oscilacao_pura

    data = pd.DataFrame({
        'time': time,
        'oscilacao_pura (m)': oscilacao_pura,
        'target_response': target_response,
    })

    try:
        data.to_csv(output_path, index=False)
        print(f"Dados salvos com sucesso em {output_path}")
    except Exception as e:
        print(f"Erro ao salvar os dados: {e}")

    return data


if __name__ == "__main__":
    generate_harmonic_data()
