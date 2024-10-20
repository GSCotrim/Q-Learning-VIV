import numpy as np
from scipy.integrate import odeint

from src.ModelParameters import ModelParameters


# oscilador de Facchinetti com van der Pol
def cylinder_wake_oscillator(x, t, csi, delta, gamma, mu, epsilon, M, MMinv):
    # 4 positions x = [ y, ydot, q, qdot ]
    BB = np.array([[0.0, 1.0, 0, 0],
                   [-delta ** 2, -(2.0 * csi * delta + gamma / mu), M, 0.0],
                   [0.0, 0.0, 0.0, 1.0],
                   [0, 0, -1, -epsilon * (x[2] ** 2.0 - 1.0)]])
    ## MM inv eh definido com parametros A
    res = MMinv.dot(BB.dot(x))
    return res


def simulate_system_param(params, ts):
    model_parameters = ModelParameters(epsilon_num = params[0],
                                       a_num = params[1],
                                       xi_num = params[2],
                                       fluid_damping_coefficient_gamma = params[3],
                                       nondimensional_mass_ratio_mu = params[4]
                                       )

    # matrix for van der pol - dependends on A_num == params[1]
    MM = np.eye(4)
    MM[3, 1] = -model_parameters.a_num
    MMinv = np.linalg.inv(MM)

    # call ODE solver
    xs = ode_solver(ts, model_parameters.xi_num,
                    model_parameters.structure_reduced_angular_frequency_delta,
                    model_parameters.fluid_damping_coefficient_gamma,
                    model_parameters.nondimensional_mass_ratio_mu,
                    model_parameters.epsilon_num,
                    model_parameters.mass_number_M, MMinv)

    return xs[:, [0, 2]]  # this returns y and q


def compute_reward(simulated_response, target_response):
    simulated_y = simulated_response[:, 0]
    mse = np.mean((simulated_y - target_response) ** 2)
    reward = -mse
    return reward


def ode_solver(ts, xi_num, structure_reduced_angular_frequency, fluid_damping_coefficient, nondimensional_mass_ratio,
               epsilon_num, mass_number_M, MMinv):
    return odeint(
        cylinder_wake_oscillator,
        (.5, 1e-10, 1e-10, 1e-10),  # initial_conditions
        ts,
        args=(
            xi_num, structure_reduced_angular_frequency,
            fluid_damping_coefficient,
            nondimensional_mass_ratio,
            epsilon_num,
            mass_number_M,
            MMinv)
    )
