import os
import matplotlib.pyplot as plt
import pandas as pd

from environment import simulate_system
from q_learning import QLearningAgent


def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        if 'initial_conditions' not in data.columns:
            print("Erro: A coluna 'initial_conditions' não foi encontrada no arquivo CSV.")
            return None, None, None
        initial_conditions = data['initial_conditions'].apply(eval).apply(lambda x: tuple(map(float, x))).values
        t = data['t'].values
        if 'target_response' in data.columns:
            target_response = data['target_response'].values
        else:
            print("Erro: A coluna 'target_response' não foi encontrada no arquivo CSV.")
            return None, None, None
    except FileNotFoundError:
        print("Arquivo não encontrado. Verifique o caminho do arquivo.")
        return None, None, None
    except pd.errors.EmptyDataError:
        print("Arquivo CSV está vazio.")
        return None, None, None
    except Exception as e:
        print(f"Ocorreu um erro ao ler o arquivo: {e}")
        return None, None, None

    return initial_conditions, t, target_response


def initialize_agent():
    return QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1)


def run_q_learning(agent, initial_conditions, t, target_response, episodes=5000):
    total_rewards, q_table = agent.run(initial_conditions, t, target_response, episodes)
    return total_rewards, q_table


def plot_results(total_rewards, t, simulated_response_final_refined, target_response):
    plt.plot(total_rewards)
    plt.title("Evolução das Recompensas ao Longo dos Episódios (Recompensa Refinada)")
    plt.xlabel("Episódios")
    plt.ylabel("Recompensa Total")
    plt.show()

    plt.plot(t, simulated_response_final_refined[:, 0], label="Resposta Simulada (Ajustada - Refinada)")
    plt.plot(t, target_response, label="Resposta Alvo (Lock-in)")
    plt.xlabel("Tempo")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("Comparação da Resposta Simulada com a Resposta Alvo (Lock-in - Refinado)")
    plt.show()


def run_q_learning_process():
    file_path = os.path.abspath("data/dados_viv_simulados.csv")
    initial_conditions, t, target_response = load_data(file_path)
    if initial_conditions is None or t is None or target_response is None:
        return

    agent = initialize_agent()
    total_rewards, q_table = run_q_learning(agent, initial_conditions, t, target_response)

    final_params_refined = max(q_table, key=lambda x: max(q_table[x].values()))
    print(
        f"Parâmetros ajustados após o treinamento refinado: y={final_params_refined[0]}, y_dot={final_params_refined[1]}, q={final_params_refined[2]}, q_dot={final_params_refined[3]}")

    simulated_response_final_refined = simulate_system(final_params_refined, initial_conditions, t)
    plot_results(total_rewards, t, simulated_response_final_refined, target_response)


if __name__ == "__main__":
    run_q_learning_process()