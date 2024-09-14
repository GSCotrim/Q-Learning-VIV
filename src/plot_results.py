import matplotlib.pyplot as plt

def plot_reward_history(reward_history):
    plt.figure()
    plt.plot(reward_history)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Convergence of Q-Learning')
    plt.grid(True)
    plt.show()

def plot_q_table(Q_table):
    plt.figure()
    plt.imshow(Q_table, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Q-Table')
    plt.xlabel('Actions')
    plt.ylabel('States')
    plt.show()

def save_plot(filepath):
    plt.savefig(filepath)
    plt.close()  # Fecha o gráfico para evitar acumulação de memória
