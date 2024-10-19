import numpy as np
from scipy.integrate import odeint

from src.constants import U_RED, CILINDER_DIAMETER, STROUHAL, FLUID_DAMPING_CDosc, ADDED_MASS_COEFICIENT_CM, \
    WATER_DENSITY_RHO, COUPLING_CL0, CILINDER_FREQUENCY_OSCILATION_FS, NONDIMENSIONAL_MASS_M_STAR


# oscilador de Facchinetti com van der Pol
def cylinder_wake_oscillator(x, t, csi, delta, gamma, mu, epsilon, M, MMinv):
    # xi_num, delta_num, gamma_num, mu_num, epsilon_num, A_num, M_num,
    # 4 positions x = [ y, ydot, q, qdot ]
    BB = np.array([[0.0, 1.0, 0, 0],
                   [-delta ** 2, -(2.0 * csi * delta + gamma / mu), M, 0.0],
                   [0.0, 0.0, 0.0, 1.0],
                   [0, 0, -1, -epsilon * (x[2] ** 2.0 - 1.0)]])
    ## MM inv eh definido com parametros A
    res = MMinv.dot(BB.dot(x))
    return res


def simulate_system_param(params, ts):
    epsilon_num, A_num, xi_num = params

    # other parameters derived/calculated from the above ones -- see facchinetti 2004
    structural_angular_frequency = set_structural_angular_frequency()
    infinity_flow_velocity = set_infinity_flow_velocity(structural_angular_frequency)
    angular_frequency = set_angular_frequency(infinity_flow_velocity)
    structure_reduced_angular_frequency = set_structure_reduced_angular_frequency(structural_angular_frequency, angular_frequency)
    fluid_damping_coefficient = set_fluid_damping_coefficient()
    fluid_mass = set_fluid_mass()
    structure_mass = set_structure_mass(fluid_mass)
    nondimensional_mass_ratio = set_nondimensional_mass_ratio(fluid_mass, structure_mass)
    mass_number_M = set_M(nondimensional_mass_ratio)

    # matrix for van der pol - dependends on A_num == params[1]
    MM = np.eye(4)
    MM[3, 1] = -A_num
    MMinv = np.linalg.inv(MM)

    # call ODE solver
    xs = ode_solver(ts, xi_num, structure_reduced_angular_frequency, fluid_damping_coefficient,
                    nondimensional_mass_ratio, epsilon_num, mass_number_M, MMinv)

    return xs[:, [0, 2]]  # this returns y and q


def compute_reward(simulated_response, target_response):
    simulated_y = simulated_response[:, 0]
    mse = np.mean((simulated_y - target_response) ** 2)
    reward = -mse
    return reward


def set_structural_angular_frequency():
    return 2.0 * np.pi * CILINDER_FREQUENCY_OSCILATION_FS


def set_infinity_flow_velocity(structural_angular_frequency):
    return U_RED * CILINDER_DIAMETER * structural_angular_frequency / 2.0 / np.pi


def set_angular_frequency(infinity_flow_velocity):
    return 2.0 * np.pi * (STROUHAL * infinity_flow_velocity / CILINDER_DIAMETER)


def set_structure_reduced_angular_frequency(structural_angular_frequency, angular_frequency):
    return structural_angular_frequency / angular_frequency


def set_fluid_damping_coefficient():
    return FLUID_DAMPING_CDosc / (4.0 * np.pi * STROUHAL)


def set_fluid_mass():
    return ADDED_MASS_COEFICIENT_CM * WATER_DENSITY_RHO * CILINDER_DIAMETER ** 2 * np.pi / 4.0


def set_structure_mass(fluid_mass):
    return NONDIMENSIONAL_MASS_M_STAR * fluid_mass


def set_nondimensional_mass_ratio(fluid_mass, structure_mass):
    return (fluid_mass + structure_mass) / (WATER_DENSITY_RHO * CILINDER_DIAMETER ** 2)


def set_M(nondimensional_mass_ratio):
    return 0.5 * COUPLING_CL0 / (8.0 * np.pi ** 2 * STROUHAL ** 2 * nondimensional_mass_ratio)


def ode_solver(ts, xi_num, structure_reduced_angular_frequency, fluid_damping_coefficient, nondimensional_mass_ratio,
               epsilon_num, mass_number_M, MMinv):
    return odeint(
        cylinder_wake_oscillator,
        (.5, 1e-10, 1e-10, 1e-10),  # initial_conditions
        ts,
        args=(xi_num, structure_reduced_angular_frequency, fluid_damping_coefficient, nondimensional_mass_ratio, epsilon_num, mass_number_M, MMinv)
    )