"""
Created on Mon Jun  5 12:29:24 2023

@author: Parham Oveissi
"""

import PyRCAC
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[0.3, 0.2],[0.1, 0.6]])
B = np.array([[0],[1]])
C = np.array([[1,0]])
D = 0

# ------------------------------------------------------------ Simulation ------------------------------------------------------------------

my_system = PyRCAC.System_definition(A, B, C, D)                                                     # First Step: Define your system (A, B, C and D matrices)

# my_system_tf = my_system.get_TF_from_ABCD()


my_controller = PyRCAC.RCAC(Nc = 4, Rz = 1, Ru = 0e+1, RegZ = 1, FF = 0, Integrator = 1, FIR = 0, ContType = "dense", R0 = 1e5, Lambda = 0.9995)          # Second Step: Setup your RCAC control structure



my_filter = PyRCAC.Filter_initializer(my_system)                                                  # Third Step: Initialize your filter



x, u, y0, z, theta, ref_sig = my_controller.simulate(my_system, my_filter, 500, 'Step_Input')   # Fourth Step: Run the simulation(Step_Input, Double_Step_Input, ...)



# ------------------------------------------------------------ Plots ---------------------------------------------------------------------------
plt.plot(x[0,:])
plt.plot(x[1,:])
plt.ylabel('x')
plt.figure()
plt.plot(y0[0,:])
plt.plot(ref_sig[0,:])
plt.ylabel('y')
plt.figure()
plt.plot(u[0,:])
plt.ylabel('u')