import json
import os.path as osp
import time

import numpy as np
import pyvista as pv
import ufl
from dolfinx import fem, plot
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc

from utils_simulations import project_to_cg1

try:
    from dolfinx import default_scalar_type
except Exception:
    default_scalar_type = PETSc.ScalarType


def calc_rel_l2_error_same_mesh(u_approx, u_exact):
    """
    Relative L2 error for two functions on the same mesh and same FE layout.
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
):
    """
    Linear-elasticity solver-error study on a fixed mesh.
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
    u_trial = ufl.TrialFunction(V)
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

    mu = E / (2.0 * (1.0 + nu))
    lambda_ = 2.0 * mu * nu / (1.0 - nu)

    def epsilon(w):
        return ufl.sym(ufl.nabla_grad(w))

    def sigma(w):
        return lambda_ * ufl.nabla_div(w) * ufl.Identity(gdim) + 2.0 * mu * epsilon(w)

    T = fem.Constant(domain, default_scalar_type((fx, fy)))

    a = ufl.inner(sigma(u_trial), epsilon(v)) * dx
    L = ufl.dot(T, v) * ds(RIGHT_ID)

    # Build CG1 spaces once for all exported/reference fields.
    V_cg1_vec = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))

    u_solutions = []
    sigma_solutions = []
    epsilon_solutions = []
    vm_solutions = []

    for fidelity_idx, (solver_max_it, metadatas) in enumerate(
        zip(solver_max_its_list, metadatas_list)
    ):
        prefix = f"linear_elasticity_solver_fid_{fidelity_idx}_"
        problem = LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options={
                "ksp_type": "richardson",  # Richardson iteration to drive the stationary method
                "pc_type": "sor",  # Successive Over-Relaxation
                "pc_sor_omega": 1.0,  # relaxation factor omega=1.0 to make SOR mathematically identical to standard Gauss-Seidel
                "pc_sor_symmetric": 0,  # standard forward GS sweep
                "ksp_max_it": solver_max_it,
                "ksp_rtol": 1e-50,
                "ksp_atol": 1e-50,
                "ksp_divtol": 1e12,
                "ksp_initial_guess_nonzero": 0,
                "ksp_norm_type": "unpreconditioned",
            },
            petsc_options_prefix=prefix,
        )

        ksp = problem.solver
        ksp.setConvergenceHistory()
        ksp.setComputeSingularValues(True)

        domain.comm.Barrier()
        t_start = time.perf_counter()
        uh = problem.solve()
        domain.comm.Barrier()
        t_end = time.perf_counter()

        converged_reason = int(ksp.getConvergedReason())
        converged = converged_reason > 0
        num_iterations = int(ksp.getIterationNumber())
        abs_residual = float(ksp.getResidualNorm())

        try:
            rhs_norm = float(ksp.getRhs().norm())
            rel_residual = abs_residual / (rhs_norm + 1e-15)
        except Exception:
            rel_residual = np.nan

        conv_history = ksp.getConvergenceHistory()
        residuals = conv_history[0] if isinstance(conv_history, tuple) else conv_history
        if len(residuals) > 0:
            rel_residual_history = float(residuals[-1] / (residuals[0] + 1e-15))
        else:
            rel_residual_history = np.nan

        try:
            # computeExtremeSingularValues() returns (emax, emin)
            emax, emin = ksp.computeExtremeSingularValues()
            condition_number = float(emax / emin) if emin > 0 else np.nan
        except Exception:
            condition_number = np.nan
            emax, emin = np.nan, np.nan

        uh.name = "u"
        eps_expr = epsilon(uh)
        sig_expr = sigma(uh)

        sxx, syy, sxy = sig_expr[0, 0], sig_expr[1, 1], sig_expr[0, 1]
        vm_expr = ufl.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2)
        strain_mag_expr = ufl.sqrt(ufl.inner(eps_expr, eps_expr))

        # Export fields on CG1 for consistent postprocessing.
        uh_cg1 = fem.Function(V_cg1_vec, name="u")
        uh_cg1.interpolate(uh)

        sigma_cg1 = project_to_cg1(sig_expr, domain, tensor_shape=(gdim, gdim))
        sigma_cg1.name = "sigma"
        epsilon_cg1 = project_to_cg1(eps_expr, domain, tensor_shape=(gdim, gdim))
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

        local_strain_energy = fem.assemble_scalar(
            fem.form(0.5 * ufl.inner(sig_expr, eps_expr) * dx)
        )
        strain_energy_fem = domain.comm.allreduce(local_strain_energy, op=MPI.SUM)

        local_compliance = fem.assemble_scalar(fem.form(ufl.dot(T, uh) * ds(RIGHT_ID)))
        compliance_fem = domain.comm.allreduce(local_compliance, op=MPI.SUM)

        # Store copies for error computation after all fidelities are solved.
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
                # Richardson + SOR(omega=1, forward sweep) == plain Gauss-Seidel
                "ksp_type": "richardson",
                "pc_type": "sor",
                "pc_sor_omega": 1.0,
                "iterative_scheme": "gauss_seidel_forward",
                "error_reference": "last_level_same_mesh",
                "dofs": int(V.dofmap.index_map.size_global * V.dofmap.index_map_bs),
                "n_nodes": int(domain.topology.index_map(0).size_global),
                "n_cells": int(domain.topology.index_map(domain.topology.dim).size_global),
                "simulation_time": t_end - t_start,
                "converged": bool(converged),
                "converged_reason": converged_reason,
                "num_iterations": num_iterations,
                "abs_residual": abs_residual,
                "rel_residual": float(rel_residual) if np.isfinite(rel_residual) else np.nan,
                "rel_residual_history": float(rel_residual_history)
                if np.isfinite(rel_residual_history)
                else np.nan,
                "strain_energy_fem": float(strain_energy_fem),
                "compliance_fem": float(compliance_fem),
                "vm_max": vm_max,
                "vm_mean": vm_mean,
                "vm_p99": vm_p99,
                "max_strain": max_strain,
                "condition_number": condition_number,
                "eigenvalue_max": float(emax),
                "eigenvalue_min": float(emin),
            }
        )

    # Relative L2 errors against the highest-iteration (last) level on the same mesh.
    for i in range(len(u_solutions)):
        rel_u = calc_rel_l2_error_same_mesh(u_solutions[i], u_solutions[-1])
        rel_sigma = calc_rel_l2_error_same_mesh(sigma_solutions[i], sigma_solutions[-1])
        rel_epsilon = calc_rel_l2_error_same_mesh(epsilon_solutions[i], epsilon_solutions[-1])
        rel_vm = calc_rel_l2_error_same_mesh(vm_solutions[i], vm_solutions[-1])

        metadatas_list[i]["rel_l2_error_u"] = rel_u
        metadatas_list[i]["rel_l2_error_sigma"] = rel_sigma
        metadatas_list[i]["rel_l2_error_epsilon"] = rel_epsilon
        metadatas_list[i]["rel_l2_error_vm"] = rel_vm
        # for backwards compatibility
        metadatas_list[i]["rel_l2_error"] = rel_u

    for i in range(len(u_solutions)):
        fidelity_id = str(i).zfill(len(str(len(solver_max_its_list))))
        metadata_file = osp.join(outfolder, f"{fidelity_id}_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadatas_list[i], f, indent=2)

    return None
