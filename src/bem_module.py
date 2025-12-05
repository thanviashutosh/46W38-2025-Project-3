# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 15:37:34 2025

@author: ASHTHA
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d

class WindTurbineBEM:
    """
    This is a Class for BEM implementation :
    Attributes:
        Blade file, aerofoil folder and operational file
        
    Methods
    - load blade file
    - Airfoil polar interpolation
    - Tip-loss correction (Prandtl)
    - Operational strategy reading
    - plot aerofoil shapes
    """

    def __init__(self, blade_file, airfoil_folder, operational_file):
        self.blade_file = blade_file
        self.airfoil_folder = airfoil_folder
        self.operational_file = operational_file

        # Load data
        self.load_blade()
        self.airfoils = self.load_airfoils()
        self.operational = self.load_operational_strategy()

        self.rho = 1.225 # density in kg/m3
        self.B = 3  # blades
        self.rated_power = 15e6  # 15 MW in Watts

    # *********************
    # LOAD BLADE GEOMETRY
    # *********************
    def load_blade(self):
        """Loads span, chord, twist, airfoil ID from blade file."""
        data = np.loadtxt(self.blade_file, skiprows=6)
        self.r = data[:, 0]       # BlSpn
        self.chord = data[:, 5]   # BlChord
        self.twist = data[:, 4]  # twist
        self.af_id = data[:, 6].astype(int)  # 1–50

        self.R = np.max(self.r)

    
    
    # ******************************
    # LOAD AIRFOILS (SHAPES + POLARS)
    # ******************************
    def load_airfoils(self):
        airfoil_data = {}
    
        # Shape files AF00..AF49
        shape_files = sorted([f for f in os.listdir(self.airfoil_folder) 
                          if f.endswith("_Coords.txt")])

        for idx, fname in enumerate(shape_files):
            af_id = idx  # index 0–49

        # Load coordinates
            coords_path = os.path.join(self.airfoil_folder, fname)
            coords = np.loadtxt(coords_path, skiprows=8)
           

        # Load polar file with matching index
            polar_name = f"IEA-15-240-RWT_AeroDyn15_Polar_{idx:02d}.dat"
            polar_path = os.path.join(self.airfoil_folder, polar_name)

        # Correct skiprows rule
            skip = 20 if idx <= 4 else 54
            polar_data = np.loadtxt(polar_path, skiprows=skip)

            airfoil_data[af_id] = {
                "coords": coords,
                "polar": polar_data,
                "alpha": polar_data[:, 0],
                "Cl": polar_data[:, 1],
                "Cd": polar_data[:, 2]
                }

        return airfoil_data


    # *********************
    # PLOT AIRFOIL SHAPES
    # *********************
    def plot_airfoils(self):
        plt.figure(figsize=(8, 6))
    
        for af_id, af_data in self.airfoils.items():
            coords = np.array(af_data["coords"])
            plt.plot(coords[:, 0], coords[:, 1], label=f"AF {af_id}")
    
        plt.xlabel("x/c")
        plt.ylabel("y/c")
        plt.title("Airfoil Shapes")
        plt.axis("equal")
        plt.grid(True)
        plt.legend(fontsize='small', ncol=10, loc='center', bbox_to_anchor=(0.5, -0.15))
        plt.tight_layout()
        output_dir = "../Output_aerofoil_shapes"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "aerofoil_shapes.png"), dpi=300)
        
        

    # *****************
    # PRANDTL TIP LOSS
    # *****************
    def prandtl_tip_loss(self, r, phi):
        F = (2/np.pi) * np.arccos(
            np.exp(-self.B/2 * (self.R - r) / (r * np.sin(phi)))
        )
        return np.maximum(F, 1e-6)

    # ********************************
    # COMPUTE BEM FOR ONE WIND SPEED
    # *******************************
    def compute_BEM(self, V0, pitch, omega):

        dr = np.gradient(self.r)
        T_total = 0.0
        M_total = 0.0

        for i in range(len(self.r)):
            ri = self.r[i]
            chord = self.chord[i]
            twist = self.twist[i]
            af = self.airfoils[self.af_id[i] - 1]

            sigma = self.B * chord / (2 * np.pi * ri)

            a = 0.0
            a_prime = 0.0

            for _ in range(100):

                phi = np.arctan2(V0*(1-a), omega*ri*(1+a_prime))

                alpha = phi - (np.deg2rad(pitch) + np.deg2rad(twist))
                alpha_deg = np.rad2deg(alpha)

                # Interpolate Cl/Cd correctly
                Cl = np.interp(alpha_deg, af["alpha"], af["Cl"])
                Cd = np.interp(alpha_deg, af["alpha"], af["Cd"])

                # Force coefficients
                Cn = Cl*np.cos(phi) + Cd*np.sin(phi)
                Ct = Cl*np.sin(phi) - Cd*np.cos(phi)


                # Tip-loss
                F = self.prandtl_tip_loss(ri, phi)

                # Updated induction factors (with F)
                a_new = 1 / (((4*F*np.sin(phi)**2)/(sigma*Cn)) + 1)
                a_prime_new = 1 / (4*F*np.sin(phi)*np.cos(phi)/(sigma*Ct) - 1)

                if abs(a_new - a) < 1e-5 and abs(a_prime_new - a_prime) < 1e-5:
                    a = a_new
                    a_prime = a_prime_new
                    break

                a, a_prime = a_new, a_prime_new

           
            # Forces
            dT = 4 * np.pi * ri * self.rho * V0**2 * a * (1 - a) * dr[i]
            dM = 4 * np.pi * ri**3 * self.rho * V0 * omega * a_prime * (1 - a) * dr[i]

            T_total += dT
            M_total += dM

        P = min(M_total * omega, self.rated_power)
    
        return T_total, M_total, P

 # *****************************
 # LOAD OPERATIONAL STRATEGY
 # ******************************
    def load_operational_strategy(self):
          """
          Reads the operational file:
             V0 [m/s], pitch [deg], rpm, aero power, thrust
             Converts rpm → rad/s
             """
          data = np.loadtxt(self.operational_file, skiprows=1)

          V0 = data[:, 0]
          pitch_deg = data[:, 1]
          rpm = data[:, 2]

          omega = rpm * (2*np.pi / 60)  # rad/s
     
           # Produce continuous functions (piecewise linear is recommended)
          self.pitch_opt = interp1d(V0, pitch_deg, fill_value="extrapolate")
          self.omega_opt = interp1d(V0, omega, fill_value="extrapolate")

          return {
            "V0": V0,
            "pitch": np.deg2rad(pitch_deg),
            "omega": omega
            }

    def optimize_operational_point(self, V0):
         """
         operational strategy curves(interpolated pitch and omega
                                     from the input .txt file).

         Returns:
             pitch_opt [deg],
             omega_opt [rad/s],
             T_total [N],
             P_total [W]
             """

        # Interpolate from operational file
         pitch_deg = float(self.pitch_opt(V0))      # degrees
         omega = float(self.omega_opt(V0))          # rad/s

        # Compute BEM at this operating point
         T_total, M_total, P_total = self.compute_BEM(V0, pitch_deg, omega)

         return pitch_deg, omega, T_total, P_total   

 #**********************
 # COMPUTE OPTIMAL CURVES
 #*************************
    def compute_optimal_curve(self):
        V0_list = np.arange(0,30,1)

        pitch_curve = []
        omega_curve = []
        P_curve = []
        T_curve = []

        for V0 in V0_list:
            pitch_opt, omega_opt, T_opt, P_opt = self.optimize_operational_point(V0)
            

            pitch_curve.append(pitch_opt)
            omega_curve.append(omega_opt)
            T_curve.append(T_opt)
            P_curve.append(P_opt)
            

        return (V0_list, 
                np.array(pitch_curve), 
                np.array(omega_curve),
                np.array(P_curve), 
                np.array(T_curve)) 

    
