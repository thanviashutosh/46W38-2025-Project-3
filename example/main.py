# -*- coding: utf-8 -*-
"""
@author: Ashutosh Thanvi s252568
"""

"""
This module is the driver used for the simulation of wind turbine modelling using 
Blade Element Momentum Theory. This module creates the required folders and 
simulates BEM at all the aerofoils along the blade span and calculatue Power and Thrust 
for wind speeds.It also use operational strategy to calculate pitch angle and omega for 
various wind speeds. 

The module also plots shapes of 50 aerofoil located along the blade span. 

The results are saved in the output folder.

The Power and Thrust results  are then plotted as a function of wind speeds and saved.

This module bem_module created for the wind turbine modelling. 
"""
import matplotlib.pyplot as plt

import os
import sys

blade_file = "../inputs/IEA-15-240-RWT/IEA-15-240-RWT_AeroDyn15_blade.dat"
airfoil_folder = "../inputs/IEA-15-240-RWT/Airfoils"
operational_file = "../inputs/IEA-15-240-RWT/IEA_15MW_RWT_Onshore.opt"

sys.path.insert(0, os.path.abspath('../src')) # locating the folder for the module
from bem_module import WindTurbineBEM # importing the bem_module

# Initialize BEM model
bem_model = WindTurbineBEM(blade_file, airfoil_folder, operational_file)

# Plot all airfoil shapes
bem_model.plot_airfoils()

# Compute power and thrust curves
V0s, pitch_curve, omega_curve, P_curve, T_curve = bem_model.compute_optimal_curve()

#creating output folder for saving plots
output_dir = "../output"
os.makedirs(output_dir, exist_ok=True)


# Plot results
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.plot(V0s, P_curve / 1e6, 'r-o')
plt.xlabel("Wind Speed [m/s]")
plt.ylabel("Power [MW]")
plt.title("Rotor Power Curve")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(V0s, T_curve / 1000.0, 'b-s')
plt.xlabel("Wind Speed [m/s]")
plt.ylabel("Thrust [kN]")
plt.title("Rotor Thrust Curve")
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "power_thrust_curves.png"), dpi=300)
plt.show()



