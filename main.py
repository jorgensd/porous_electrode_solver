#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 00:51:26 2025
@author: yuhe wang

Main script for solving Galvanostatic/Potentiostatic potential fields
(electrode/electrolyte) using Decoupled or Fully-Coupled methods.
This script sets up the grid, physical parameters, conductivity fields
(homogeneous/heterogeneous),boundary conditions, reference potential settings,
and then calls the solver. It also plots the results and saves selected data.

Modified by Amirhossein Aghabarari November 2025

"""

import time


NNN = 31
import numpy as np
from newton_raphson import newton_raphson
from bultervolmerclassic import bultervolmerclassic
from generate_bimodal_field import generate_bimodal_field
from generate_channelized_field import generate_channelized_field
from plot_functions import plot_sigma_kappa, plot_potential_results
from line_search import update
import os

# ----------------------------
# General Setup
# ----------------------------
homogeneous = True  # True: homogeneous; False: heterogeneous
heterogeneous_type = "bimodal"  # Options: "bimodal" or "channelized"
operating_mode = 1  # 1: Galvanostatic; 2: Potentiostatic sweeping; 3: Potentiostatic current sweeping
reference_method = 2  # 1: LCM; 2: DSM; 3: No reference potential enforcement
numerical_scheme = 2  # 1: Decoupled; 2: Fully-Coupled

# ----------------------------
# Grid Information
# ----------------------------


def solve_problem(Nxy: int):
    """
    Args:
        NNN: Resolution in x and y directions."""
    start_time = time.perf_counter()

    Nx, Ny, Nz = Nxy, Nxy, 1
    N = Nx * Ny * Nz
    Lx, Ly, Lz = 5.0e-3, 1.0e-1, 1.0e-1
    V = Lx * Ly * Lz

    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
    dV = dx * dy * dz

    # ----------------------------
    # Physical Parameters
    # ----------------------------
    D2, D3 = 2.4e-10, 2.4e-10  # Diffusivity (m^2/s)
    poro_val = 0.78  # Porosity
    sigma_val = 1000.0  # Electrode conductivity (S/m)
    As = 1.64e4  # Specific surface area (1/m)
    E = -0.255  # Reference electrode potential (V)
    I = 4.0  # Applied current (A)
    J0 = I / (Ly * Lz)  # Current density (A/m^2)

    c2, c3 = 27.0, 1053.0  # Concentrations (mol/m^3)
    F, R_const, T = 96485, 8.314, 298.15
    k_rate = 1.7e-7
    j0 = (F * k_rate) * (c3**0.5) * (c2**0.5)
    E_eq = E + (R_const * T / F) * np.log(c3 / c2)

    D23, c23, z23 = [D2, D3], [c2, c3], [2, 3]

    # ----------------------------
    # Conductivity Fields
    # ----------------------------
    sigma, kappa = np.zeros((Ny, Nx)), np.zeros((Ny, Nx))

    def compute_conductivity():
        temp_sum = sum((z23[p] ** 2) * D23[p] * c23[p] for p in range(len(z23)))
        kappa_val = (F**2 / (R_const * T)) * temp_sum

        if not homogeneous:
            if heterogeneous_type == "bimodal":
                porosity = generate_bimodal_field(Nx, Ny, Lx, Ly, 0.8, 0.2, 0.4)
            elif heterogeneous_type == "channelized":
                porosity = generate_channelized_field(Nx, Ny, 0.8, 0.2, 0.2)
            else:
                raise ValueError(
                    "Invalid heterogeneous_type. Choose 'bimodal' or 'channelized'."
                )
        else:
            porosity = np.full((Ny, Nx), poro_val)

        for i in range(Ny):
            for j in range(Nx):
                sigma[i, j] = ((1 - porosity[i, j]) ** 1.5) * sigma_val
                kappa[i, j] = (porosity[i, j] ** 1.5) * kappa_val

        if not homogeneous:
            plot_sigma_kappa(Lx, Ly, Nx, Ny, sigma, kappa)

    compute_conductivity()

    # ----------------------------
    # Running Cases Setup
    # ----------------------------
    def determine_run_case():
        if operating_mode == 1:  # Galvanostatic
            if numerical_scheme == 1:
                return {1: 1, 2: 2}.get(reference_method, None)
            elif numerical_scheme == 2:
                return {1: 3, 2: 4, 3: 5}.get(reference_method, None)
        elif operating_mode == 2 and numerical_scheme == 2:
            return 6  # Potentiostatic sweeping
        elif operating_mode == 3 and numerical_scheme == 2:
            return 7  # Potentiostatic current sweeping
        raise ValueError("Invalid combination of parameters.")

    run_case = determine_run_case()

    # ----------------------------
    # Reference Potential Setup
    # ----------------------------
    if run_case == 1:
        ref_value_e = 0.0
        ref_value_l = -E_eq
        reference_cells_e = [(i) * Nx + 0 for i in range(Ny)]
        reference_cells_l = [(i) * Nx + (Nx - 1) + N for i in range(Ny)]  # global index
    elif run_case == 2:
        ref_value_e = 0.0
        ref_value_l = -E_eq
    elif run_case == 3:
        ref_value_e = 0.0
        reference_cells_e = [(i) * Nx + 0 for i in range(Ny)]
    elif run_case == 4:
        ref_value_e = 0.0
    elif run_case == 5:
        # No reference potentials
        pass
    elif run_case == 6:
        v_applied = 0.0
        v_sweep = np.arange(0.1, 0.5 + 0.03, 0.03)
    elif run_case == 7:
        v_applied = 0.0
        I_sweep = np.arange(0.2, 9.0 + 1.0, 1.0)
        J_sweep = I_sweep / (Ly * Lz)

    # ----------------------------
    # Boundary Conditions Setup
    # ----------------------------
    if run_case in [1, 3, 5]:
        # All-Neumann: electrode left flux = J0; electrolyte right flux = J0.
        e_left, e_right, e_top, e_bottom = J0, 0.0, 0.0, 0.0
        l_left, l_right, l_top, l_bottom = 0.0, J0, 0.0, 0.0
    elif run_case == 2:
        # DSM & Picard: electrode left Dirichlet = ref_value_e; electrolyte right Dirichlet = ref_value_l.
        e_left, e_right, e_top, e_bottom = ref_value_e, 0.0, 0.0, 0.0
        l_left, l_right, l_top, l_bottom = 0.0, ref_value_l, 0.0, 0.0
    elif run_case == 4:
        # DSM & Fully-Coupled: electrode left Dirichlet = ref_value_e; electrolyte remains Neumann.
        e_left, e_right, e_top, e_bottom = ref_value_e, 0.0, 0.0, 0.0
        l_left, l_right, l_top, l_bottom = 0.0, J0, 0.0, 0.0
    elif run_case == 6:
        # Potentiostatic sweeping: electrode left Dirichlet = v_applied; electrolyte right Dirichlet = first element of v_sweep.
        e_left, e_right, e_top, e_bottom = v_applied, 0.0, 0.0, 0.0
        l_left, l_right, l_top, l_bottom = 0.0, v_sweep[0], 0.0, 0.0
    elif run_case == 7:
        # Current sweeping: electrode left Dirichlet = v_applied; electrolyte right remains Neumann but with J_sweep.
        e_left, e_right, e_top, e_bottom = v_applied, 0.0, 0.0, 0.0
        l_left, l_right, l_top, l_bottom = 0.0, J_sweep[0], 0.0, 0.0

    # ----------------------------
    # Initialization of Potentials and Source Terms
    # ----------------------------
    def initialize_potentials(run_case):
        """Initializes potential fields and state vector x based on the run case."""
        phi_ev = np.full(
            (N, 1),
            ref_value_e
            if run_case in [1, 2, 3, 4]
            else (v_applied if run_case in [6, 7] else 0.0),
        )
        phi_lv = np.full((N, 1), ref_value_l if run_case in [1, 2] else -E_eq)

        lambda_vec = None
        if run_case == 1:
            lambda_vec = np.zeros((len(reference_cells_e) + len(reference_cells_l), 1))
        elif run_case == 3:
            lambda_vec = np.zeros((len(reference_cells_e), 1))

        if lambda_vec is not None:
            x = np.vstack((phi_ev, phi_lv, lambda_vec))
        else:
            x = np.vstack((phi_ev, phi_lv))

        return phi_ev, phi_lv, x, lambda_vec

    phi_ev, phi_lv, x, lambda_vec = initialize_potentials(run_case)

    # For decoupled cases, initialize the local current density field
    if run_case in [1, 2]:
        aj_init = -I / V
        aj = aj_init * np.ones((Ny, Nx))
        aj_old = aj.copy()

    phi_ev_old = phi_ev.copy()
    phi_lv_old = phi_lv.copy()
    x_old = x.copy()

    # ----------------------------
    # Solve Potential Fields
    # ----------------------------
    if run_case in [1, 2]:
        max_iter_outer, max_iter_picard = 100, 1
        iter_outer, iter_picard = 0, 0
        tol_totalcharge, tol_picard = 1e-5, 1e-1
        omega = 1.0
        iter_count_picard = np.zeros((max_iter_picard + 100, 1))
        iter_count_linesearch = np.zeros((max_iter_outer, 1))
        iter_count_picard_ = np.zeros((max_iter_outer, 1))

        while iter_outer < max_iter_outer:
            if iter_outer > 0:
                charge_total = np.sum(np.abs(aj) * dV)
                res_totalcharge = abs(charge_total - I) / I
                print(
                    f"Outer iteration {iter_outer}, Charge Balance Residual = {res_totalcharge:e}"
                )
                if res_totalcharge < tol_totalcharge:
                    print("\n")
                    print("-----------------------------------------------")
                    print(
                        f"| {'Total Outer Iterations:':<30} | {int(iter_outer):<10} |"
                    )
                    print(
                        f"| {'Total Line Search Iterations:':<30} | {int(sum(iter_count_linesearch)):<10} |"
                    )
                    print(
                        f"| {'Total Picard Iterations:':<30} | {int(sum(iter_count_picard) + sum(iter_count_picard_)):<10} |"
                    )
                    print(f"| {'Total Charge:':<30} | {float(charge_total):<10.3e} |")
                    print(
                        f"| {'Charge Balance Residual:':<30} | {float(res_totalcharge):<10.3e} |"
                    )
                    print("-----------------------------------------------")
                    break
                else:
                    ref_value_l, iter_linesearch, iter_picard_ = update(
                        ref_value_l,
                        Nx,
                        Ny,
                        dx,
                        dy,
                        dV,
                        e_left,
                        e_right,
                        e_top,
                        e_bottom,
                        ref_value_e,
                        l_left,
                        l_right if run_case == 1 else [],
                        l_top,
                        l_bottom,
                        c3,
                        c2,
                        I,
                        aj_old,
                        phi_ev_old,
                        phi_lv_old,
                        sigma,
                        kappa,
                        reference_cells_e if run_case == 1 else [],
                        reference_cells_l if run_case == 1 else [],
                        lambda_vec if run_case == 1 else [],
                        run_case,
                    )

            iter_outer += 1
            if iter_outer > 1:
                iter_count_linesearch[iter_outer - 1] = iter_linesearch
                iter_count_picard_[iter_outer - 1] = iter_picard_

            while iter_picard < max_iter_picard:
                iter_picard += 1
                x, iter_nr, res_nr = newton_raphson(
                    Nx,
                    Ny,
                    dx,
                    dy,
                    sigma,
                    kappa,
                    phi_ev_old,
                    phi_lv_old,
                    e_top,
                    e_bottom,
                    e_left,
                    e_right,
                    l_top,
                    l_bottom,
                    l_left,
                    ref_value_l if run_case == 2 else l_right,
                    run_case,
                    reference_cells_e=reference_cells_e if run_case == 1 else None,
                    reference_cells_l=reference_cells_l if run_case == 1 else None,
                    ref_value_e=ref_value_e,
                    ref_value_l=ref_value_l,
                    lambda_vec=lambda_vec if run_case == 1 else None,
                )

                phi_ev_old, phi_lv_old = x[:N], x[N : 2 * N]
                phi_e = phi_ev_old.reshape((Ny, Nx))
                phi_l = phi_lv_old.reshape((Ny, Nx))
                aj, jr, eta = bultervolmerclassic(phi_e, phi_l, c3, c2)
                # Relaxation
                aj = omega * aj + (1.0 - omega) * aj_old
                res_aj = np.linalg.norm(aj - aj_old) / (np.linalg.norm(aj_old) + 1e-12)
                aj_old = aj.copy()
                if res_aj < tol_picard:
                    break
            iter_count_picard[iter_outer - 1] = iter_picard
            iter_picard = 0

    else:
        # Fully-Coupled cases (3,4,5,6,7)
        iter, end_iter = 0, 1 if run_case in [3, 4, 5] else 1000
        while iter < end_iter:
            iter += 1
            x, iter_nr, res_nr = newton_raphson(
                Nx,
                Ny,
                dx,
                dy,
                sigma,
                kappa,
                phi_ev_old,
                phi_lv_old,
                e_top,
                e_bottom,
                e_left,
                e_right,
                l_top,
                l_bottom,
                l_left,
                l_right
                if run_case not in [6, 7]
                else (v_sweep[iter - 1] if run_case == 6 else J_sweep[iter - 1]),
                run_case,
                reference_cells_e=reference_cells_e if run_case == 3 else None,
                ref_value_e=ref_value_e if run_case == 3 else None,
                lambda_vec=lambda_vec if run_case == 3 else None,
            )

            phi_ev_old, phi_lv_old = x[:N], x[N : 2 * N]
            phi_e = phi_ev_old.reshape((Ny, Nx))
            phi_l = phi_lv_old.reshape((Ny, Nx))
            aj, jr, eta = bultervolmerclassic(phi_e, phi_l, c3, c2)
            charge_total = np.sum(np.abs(aj) * dV)
            if run_case == 7:
                res_totalcharge = (
                    abs(charge_total - I_sweep[iter - 1]) / I_sweep[iter - 1]
                )
            elif run_case == 6:
                res_totalcharge = None
            else:
                res_totalcharge = abs(charge_total - I) / I
            """   
            print("\n")
            print("-----------------------------------------------")
            print(f"| {'Total Newton-Raphson Iterations:':<35} | {iter_nr:<10d} |")
            print(f"| {'Newton-Raphson Residual:':<35} | {res_nr:<10.3e} |")
            print(f"| {'Total Charge:':<35} | {charge_total:<10.3e} |")
            print(f"| {'Charge Balance Residual:':<35} | {res_totalcharge:<10.3e} |")
            print("-----------------------------------------------")
            """
    # ----------------------------
    # Plot Results
    # ----------------------------
    # ==== SAVE η(x) AT y = H/2 ======================================
    # Requires: Nx, Ny, Lx, and the 2D array 'eta' already defined.
    # 'eta' is created by: aj, jr, eta = bultervolmerclassic(phi_e, phi_l, c3, c2)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.6f} seconds")

    # x-coordinates in meters (Nx nodes from 0 to Lx)
    x_m = np.linspace(0.0, Lx, Nx)

    # pick the mid-height line (y = H/2); if Ny is even, average the two middle rows
    if Ny % 2 == 1:
        jmid = Ny // 2
        eta_line = eta[jmid, :]  # shape: (Nx,)
    else:
        jlo, jhi = Ny // 2 - 1, Ny // 2
        eta_line = 0.5 * (eta[jlo, :] + eta[jhi, :])
    return x_m, eta_line, elapsed_time


# save to CSV
os.makedirs("exports", exist_ok=True)
out_csv = os.path.join("exports", "Wang_eta_vs_x_yH2.csv")


resolutions = [8, 16, 32, 64]
runtime = np.zeros(len(resolutions))

for i, Nxy in enumerate(resolutions):
    x_m, eta_line, elapsed_time = solve_problem(Nxy)
    runtime[i] = elapsed_time
    # NOTE: Do error computation here

np.savetxt(
    out_csv,
    np.column_stack([resolutions, runtime]),
    delimiter=",",
    header="N,runtime",
    comments="",
    fmt="%.10e",
)
print(f"[export] wrote {out_csv} with rows.")
# ===============================================================

# plot_potential_results(Lx, Ly, Nx, Ny, phi_e, phi_l, eta, jr)
# print(j0)
# print(E_eq)
# print(J0)
