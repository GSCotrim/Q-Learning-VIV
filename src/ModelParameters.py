from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.constants import U_RED, CILINDER_DIAMETER, STROUHAL, FLUID_DAMPING_CDosc, ADDED_MASS_COEFICIENT_CM, \
    WATER_DENSITY_RHO, COUPLING_CL0, CILINDER_FREQUENCY_OSCILATION_FS, NONDIMENSIONAL_MASS_M_STAR


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


@dataclass
class ModelParameters:
    epsilon_num: float
    a_num: float
    xi_num: float
    fluid_damping_coefficient_gamma: float
    structural_angular_frequency: Optional[float] = set_structural_angular_frequency()
    fluid_mass: Optional[float] = set_fluid_mass()
    infinity_flow_velocity: Optional[float] = set_infinity_flow_velocity(structural_angular_frequency)
    structure_mass: Optional[float] = set_structure_mass(fluid_mass)
    angular_frequency: Optional[float] = set_angular_frequency(infinity_flow_velocity)
    structure_reduced_angular_frequency_delta: Optional[float] = set_structure_reduced_angular_frequency(structural_angular_frequency, angular_frequency)
    nondimensional_mass_ratio: Optional[float] = set_nondimensional_mass_ratio(fluid_mass, structure_mass)
    mass_number_M: Optional[float] = set_M(nondimensional_mass_ratio)
