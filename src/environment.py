## imports local
from typing import Any
import numpy as np
from scipy.fft import fft
from scipy.integrate import odeint


# oscilador de Facchinetti com van der Pol
def Cylinder_Wake_oscillator (x, t, csi, delta, gamma, mu, epsilon, M, MMinv):
  # xi_num, delta_num, gamma_num, mu_num, epsilon_num, A_num, M_num,
  # 4 positions x = [ y, ydot, q, qdot ]
  BB = np.array([ [0.0,1.0,0,0] ,
                  [-delta**2,-(2.0*csi*delta+gamma/mu),M,0.0] ,
                  [0.0,0.0,0.0,1.0] ,
                  [0,0,-1,-epsilon * (x[2] ** 2.0 - 1.0)] ])
  ## MM inv eh definido com parametros A
  res = MMinv.dot(BB.dot(x))
  return res

## global initial condition for system y, y_dot, q, q_dot
# initial_conditions = (.5, 1e-10, 1e-10, 1e-10)

# # facchinetti 2004 /wake oscillator
# A_num = 12.0
# epsilon_num = 0.3
# # time run 1000 time units y/d   adn t=T*omegaf
# ts = np.linspace(0.0, 50, 500+1)
def simulate_system_2param(params, ts):
  epsilon_num, A_num = params

  ## hardcoded evaluation at a single Ured <-> Uinf
  Ured = 5.33 ## should results in 2.0*np.pi*Uinf/Omegas/D Uinf = 0.08

  ## harcoded VIV known parameters (not params!)
  # From Assi 2010 experiments
  xi_num = .07e-2 # structural damping coefficient
  fs = 0.3        # structural - in Hertz
  D = 50e-3       # diameter in meters
  mstar = 2.6     # nondimensional
  rho = 1000.0    # kg/m^3

  # Strouhal (assumed 0.2 usual for a wide range of Reynolds)
  St = 0.2

  # added mass coefficient
  Cm = 1.0        # cylinder - potential solution

  # Coupling scales @ Violette 2007
  CL0 = 0.3     # gioria has chosen this value for coupling
  CDosc = 1.2   # gioria has chosen this value for fluid damping

  # other parameters derived/calculated from the above ones -- see facchinetti 2004
  Omegas = 2.0*np.pi*fs
  # Ured = 2.0*np.pi*Uinf/Omegas/D
  Uinf = Ured*D*Omegas/2.0/np.pi
  Omegaf = 2.0*np.pi* (St*Uinf/D)
  delta_num = Omegas/Omegaf
  gamma_num = CDosc/(4.0*np.pi*St)
  mf = Cm*rho*D**2*np.pi/4.0
  ms = mstar*mf
  m = ms+mf
  mu_num = (mf+ms)/(rho*D**2)
  M_num = 0.5*CL0/(8.0*np.pi**2*St**2*mu_num)

  # matrix for van der pol - dependends on A_num == params[1]
  MM = np.eye(4)
  MM[3,1] = -A_num
  MMinv = np.linalg.inv(MM)

  # call ODE solver
  xs = odeint(
      Cylinder_Wake_oscillator,
      (.5, 1e-10, 1e-10, 1e-10),  # initial_conditions
      ts,
      args = (xi_num, delta_num, gamma_num, mu_num, epsilon_num, M_num, MMinv)
  )

  return xs[:,[0,2]]  # this returns y and q

def compute_reward(simulated_response, target_response):
    simulated_y = simulated_response[:, 0]
    mse = np.mean((simulated_y - target_response) ** 2)
    reward = -mse
    return reward