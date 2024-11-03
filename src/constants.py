## hardcoded evaluation at a single Ured <-> Uinf
U_RED = 5.33  ## should results in 2.0*np.pi*Uinf/Omegas/D Uinf = 0.08

## harcoded VIV known parameters (not params!)
# From Assi 2010 experiments
CILINDER_FREQUENCY_OSCILATION_FS = 0.3  # structural - in Hertz
CILINDER_DIAMETER = 50e-3  # meters
NONDIMENSIONAL_MASS_M_STAR = 2.6
WATER_DENSITY_RHO = 1000.0  # kg/m^3
STROUHAL = 0.2 # Assumed 0.2 usual for a wide range of Reynolds
ADDED_MASS_COEFICIENT_CM = 1.0  # cylinder - potential solution
# xi_num = .07e-2  # structural damping coefficient

# Coupling scales @ Violette 2007
COUPLING_CL0 = 0.3  # Gioria has chosen this value for coupling
FLUID_DAMPING_CDosc = 1.2  # Gioria has chosen this value for fluid damping