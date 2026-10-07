import json
import os.path as osp
import time

import numpy as np
import pyvista as pv
import ufl
from dolfinx import fem, plot
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
from petsc4py import PETSc

from utils_simulations import project_to_cg1

try:
    from dolfinx import default_scalar_type
except Exception:
    default_scalar_type = PETSc.ScalarType


def calc_rel_l2_error_same_mesh(u_approx, u_exact):
    """
    Relative L2 error for two functions on the same mesh and FE layout.
    """
    domain = u_exact.function_space.mesh

    e_fun = fem.Function(u_exact.function_space)
    e_fun.x.array[:] = u_approx.x.array - u_exact.x.array

    error_sq = fem.form(ufl.inner(e_fun, e_fun) * ufl.dx)
    error_local = fem.assemble_scalar(error_sq)
    error_global = domain.comm.allreduce(error_local, op=MPI.SUM)

    norm_sq = fem.form(ufl.inner(u_exact, u_exact) * ufl.dx)
    norm_local = fem.assemble_scalar(norm_sq)
    norm_global = domain.comm.allreduce(norm_local, op=MPI.SUM)

    return float(np.sqrt(error_global) / (np.sqrt(norm_global) + 1e-15))


def run_mf_simulations_solver_error(
    mesh,
    element_order,
    fx,
    fy,
    E,
    nu,
    solver_max_its_list,
    metadatas_list,
    outfolder=None,
    lag_jacobian=-2,
    lag_preconditioner=-2,
):
    """
    Hyperelastic solver-error study on a fixed mesh.

    Solver fidelity is controlled by nonlinear iteration budget (`solver_max_its_list`),
    using Modified Newton / chord-style behavior by lagging Jacobian and preconditioner.
    Defaults use snes_lag_jacobian=-2, snes_lag_preconditioner=-2 and
    snes_linesearch_damping=0.05.

    Notes:
    - No load stepping is used; full load is applied directly.
    - Relative L2 errors are computed against the last fidelity
      (typically the highest max-iteration budget).
    """
    if len(solver_max_its_list) != len(metadatas_list):
        raise ValueError("solver_max_its_list and metadatas_list must have the same length.")
    if len(solver_max_its_list) == 0:
        raise ValueError("solver_max_its_list cannot be empty.")

    solver_max_its_list = [int(v) for v in solver_max_its_list]
    if any(v <= 0 for v in solver_max_its_list):
        raise ValueError("All solver max-iteration values must be positive integers.")
    if any(a > b for a, b in zip(solver_max_its_list[:-1], solver_max_its_list[1:])):
        raise ValueError("solver_max_its_list must be non-decreasing.")

    domain, facet_tags = mesh.mesh, mesh.facet_tags
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, 0)

    LEFT_ID = 2
    RIGHT_ID = 3

    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", element_order, (gdim,)))
    u = fem.Function(V, name="u")
    v = ufl.TestFunction(V)

    left_facets = facet_tags.indices[facet_tags.values == LEFT_ID].astype(np.int32)
    right_facets = facet_tags.indices[facet_tags.values == RIGHT_ID].astype(np.int32)
    if left_facets.size == 0:
        raise RuntimeError("No left boundary facets (tag=2) found for Dirichlet BC.")
    if right_facets.size == 0:
        raise RuntimeError("No right boundary facets (tag=3) found for traction BC.")

    left_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
    bc = fem.dirichletbc(np.zeros(gdim, dtype=default_scalar_type), left_dofs, V)

    dx = ufl.Measure("dx", domain=domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

    # Compressible Neo-Hookean model (same as discretization study).
    mu = E / (2.0 * (1.0 + nu))
    lambda_ = 2.0 * mu * nu / (1.0 - nu)

    I = ufl.variable(ufl.Identity(gdim))
    F = ufl.variable(I + ufl.grad(u))
    C = ufl.variable(F.T * F)
    J = ufl.variable(ufl.det(F))
    Ic = ufl.variable(ufl.tr(C))

    psi = (mu / 2.0) * (Ic - 3.0) - mu * ufl.ln(J) + (lambda_ / 2.0) * (ufl.ln(J)) ** 2

    # Full-load solve only (no load stepping).
    T = fem.Constant(domain, default_scalar_type((fx, fy)))

    total_potential = psi * dx - ufl.dot(T, u) * ds(RIGHT_ID)
    F_form = ufl.derivative(total_potential, u, v)
    J_form = ufl.derivative(F_form, u, ufl.TrialFunction(V))

    u_solutions = []
    sigma_solutions = []
    epsilon_solutions = []
    vm_solutions = []

    for fidelity_idx, (solver_max_it, metadatas) in enumerate(
        zip(solver_max_its_list, metadatas_list)
    ):
        # Cold start to keep solver-cost fidelity meaningful.
        u.x.array[:] = 0.0
        u.x.scatter_forward()

        prefix = f"hyperelasticity_modified_newton_fid_{fidelity_idx}_"
        problem = NonlinearProblem(
            F_form,
            u,
            bcs=[bc],
            J=J_form,
            petsc_options_prefix=prefix,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "snes_type": "newtonls",
                "snes_linesearch_type": "basic",
                "snes_linesearch_damping": 0.05,
                "snes_rtol": 1e-50,
                "snes_atol": 1e-50,
                "snes_max_it": int(solver_max_it),
                "snes_lag_jacobian": int(lag_jacobian),
                "snes_lag_preconditioner": int(lag_preconditioner),
            },
        )
        problem.solver.setConvergenceHistory()

        domain.comm.Barrier()
        t_start = time.perf_counter()
        problem.solve()
        domain.comm.Barrier()
        t_end = time.perf_counter()

        converged_reason = int(problem.solver.getConvergedReason())
        converged = converged_reason > 0
        num_iterations = int(problem.solver.getIterationNumber())

        conv_history = problem.solver.getConvergenceHistory()
        residuals = conv_history[0] if isinstance(conv_history, tuple) else conv_history
        if len(residuals) > 0:
            abs_residual = float(residuals[-1])
            rel_residual = float(abs_residual / (residuals[0] + 1e-15))
        else:
            if hasattr(problem.solver, "getFunctionNorm"):
                abs_residual = float(problem.solver.getFunctionNorm())
                rel_residual = np.nan
            else:
                abs_residual = np.nan
                rel_residual = np.nan

        # Export fields
        P = ufl.diff(psi, F)
        sigma_expr = (1.0 / J) * P * F.T
        epsilon_expr = 0.5 * (C - I)

        sxx, syy, sxy = sigma_expr[0, 0], sigma_expr[1, 1], sigma_expr[0, 1]
        vm_expr = ufl.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2)
        strain_mag_expr = ufl.sqrt(ufl.inner(epsilon_expr, epsilon_expr))

        V_cg1_vec = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
        uh_cg1 = fem.Function(V_cg1_vec, name="u")
        uh_cg1.interpolate(u)

        sigma_cg1 = project_to_cg1(sigma_expr, domain, tensor_shape=(gdim, gdim))
        sigma_cg1.name = "sigma"
        epsilon_cg1 = project_to_cg1(epsilon_expr, domain, tensor_shape=(gdim, gdim))
        epsilon_cg1.name = "epsilon"
        vm_cg1 = project_to_cg1(vm_expr, domain)
        vm_cg1.name = "vm"
        strain_mag_cg1 = project_to_cg1(strain_mag_expr, domain)
        strain_mag_cg1.name = "strain_mag"

        topology, cell_types, geometry = plot.vtk_mesh(V_cg1_vec)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)

        u_vals = uh_cg1.x.array.real.reshape((-1, gdim))
        if gdim == 2:
            u_vals = np.hstack([u_vals, np.zeros((u_vals.shape[0], 1))])

        grid.point_data["u"] = u_vals
        grid.point_data["sigma"] = sigma_cg1.x.array.real.reshape((-1, gdim * gdim))
        grid.point_data["epsilon"] = epsilon_cg1.x.array.real.reshape((-1, gdim * gdim))
        grid.point_data["vm"] = vm_cg1.x.array.real

        fidelity_id = str(fidelity_idx).zfill(len(str(len(solver_max_its_list))))
        outfile = osp.join(outfolder, f"{fidelity_id}.vtu")
        grid.save(outfile)

        vm_vals = vm_cg1.x.array.real
        vm_max = float(np.max(vm_vals)) if vm_vals.size else np.nan
        vm_mean = float(np.mean(vm_vals)) if vm_vals.size else np.nan
        vm_p99 = float(np.percentile(vm_vals, 99.0)) if vm_vals.size else np.nan

        strain_vals = strain_mag_cg1.x.array.real
        max_strain = float(np.max(strain_vals)) if strain_vals.size else np.nan

        local_strain_energy = fem.assemble_scalar(fem.form(psi * dx))
        strain_energy_fem = domain.comm.allreduce(local_strain_energy, op=MPI.SUM)

        local_compliance = fem.assemble_scalar(fem.form(ufl.dot(T, u) * ds(RIGHT_ID)))
        compliance_fem = domain.comm.allreduce(local_compliance, op=MPI.SUM)

        # Store copies for relative error calculation
        u_save = fem.Function(V_cg1_vec)
        u_save.x.array[:] = uh_cg1.x.array
        u_solutions.append(u_save)

        sigma_save = fem.Function(sigma_cg1.function_space)
        sigma_save.x.array[:] = sigma_cg1.x.array
        sigma_solutions.append(sigma_save)

        epsilon_save = fem.Function(epsilon_cg1.function_space)
        epsilon_save.x.array[:] = epsilon_cg1.x.array
        epsilon_solutions.append(epsilon_save)

        vm_save = fem.Function(vm_cg1.function_space)
        vm_save.x.array[:] = vm_cg1.x.array
        vm_solutions.append(vm_save)

        metadatas.update(
            {
                "solver_max_its": int(solver_max_it),
                "solver_method": "modified_newton",
                "snes_lag_jacobian": int(lag_jacobian),
                "snes_lag_preconditioner": int(lag_preconditioner),
                # fixed step length of the damped chord iteration; this sets the
                # linear convergence rate (rho ~ 1 - damping)
                "snes_linesearch_type": "basic",
                "snes_linesearch_damping": 0.05,
                "dofs": int(V.dofmap.index_map.size_global * V.dofmap.index_map_bs),
                "n_nodes": int(domain.topology.index_map(0).size_global),
                "n_cells": int(domain.topology.index_map(domain.topology.dim).size_global),
                "simulation_time": t_end - t_start,
                "converged": bool(converged),
                "converged_reason": converged_reason,
                "num_iterations": num_iterations,
                "abs_residual": float(abs_residual) if np.isfinite(abs_residual) else np.nan,
                "rel_residual": float(rel_residual) if np.isfinite(rel_residual) else np.nan,
                "strain_energy_fem": float(strain_energy_fem),
                "compliance_fem": float(compliance_fem),
                "vm_max": vm_max,
                "vm_mean": vm_mean,
                "vm_p99": vm_p99,
                "max_strain": max_strain,
            }
        )

    # Relative L2 errors against highest solver budget (last fidelity)
    if len(u_solutions) > 1:
        for i in range(len(u_solutions) - 1):
            rel_u = calc_rel_l2_error_same_mesh(u_solutions[i], u_solutions[-1])
            rel_sigma = calc_rel_l2_error_same_mesh(sigma_solutions[i], sigma_solutions[-1])
            rel_epsilon = calc_rel_l2_error_same_mesh(
                epsilon_solutions[i], epsilon_solutions[-1]
            )
            rel_vm = calc_rel_l2_error_same_mesh(vm_solutions[i], vm_solutions[-1])

            metadatas_list[i]["rel_l2_error_u"] = rel_u
            metadatas_list[i]["rel_l2_error_sigma"] = rel_sigma
            metadatas_list[i]["rel_l2_error_epsilon"] = rel_epsilon
            metadatas_list[i]["rel_l2_error_vm"] = rel_vm
            # Backward compatibility alias
            metadatas_list[i]["rel_l2_error"] = rel_u

        metadatas_list[-1]["rel_l2_error_u"] = 0.0
        metadatas_list[-1]["rel_l2_error_sigma"] = 0.0
        metadatas_list[-1]["rel_l2_error_epsilon"] = 0.0
        metadatas_list[-1]["rel_l2_error_vm"] = 0.0
        metadatas_list[-1]["rel_l2_error"] = 0.0
    else:
        metadatas_list[0]["rel_l2_error_u"] = 0.0
        metadatas_list[0]["rel_l2_error_sigma"] = 0.0
        metadatas_list[0]["rel_l2_error_epsilon"] = 0.0
        metadatas_list[0]["rel_l2_error_vm"] = 0.0
        metadatas_list[0]["rel_l2_error"] = 0.0

    for i in range(len(u_solutions)):
        fidelity_id = str(i).zfill(len(str(len(solver_max_its_list))))
        metadata_file = osp.join(outfolder, f"{fidelity_id}_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadatas_list[i], f, indent=2)

    return None
