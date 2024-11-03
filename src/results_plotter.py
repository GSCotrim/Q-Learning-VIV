from matplotlib import pyplot as plt


def plot_results(total_rewards, t, simulated_response_final_refined, target_response):
    plt.figure()
    plt.plot(t, simulated_response_final_refined[:, 0], label="Simulated y(t) (Cylinder Position)")
    plt.plot(t, target_response, label="Target Response")

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Simulated vs Target Oscillation (Coupled System)")
    plt.show()

    plt.figure()
    plt.plot(total_rewards, label="Total Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Evolution of Total Rewards During Training")
    plt.legend()
    plt.show()