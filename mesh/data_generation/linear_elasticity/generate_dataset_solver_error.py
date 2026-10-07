import os
import os.path as osp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyvista as pv
from tqdm import tqdm

from simulation_solver_error import run_mf_simulations_solver_error
from utils_postprocessing import postprocess_simulation
from utils_simulations import (
    build_mesh,
    sample_geometry,
    sample_mesh_params_from_fidelity,
    save_image,
)


def job(
    rectangle,
    meshing_params,
    element_order,
    fx,
    fy,
    E,
    nu,
    solver_max_its_list,
    metadatas_list,
    outfolder,
):
    # One fixed mesh per sample (solver-error study).
    mesh = build_mesh(rectangle=rectangle, **meshing_params, order=element_order)

    return run_mf_simulations_solver_error(
        mesh=mesh,
        element_order=element_order,
        fx=fx,
        fy=fy,
        E=E,
        nu=nu,
        solver_max_its_list=solver_max_its_list,
        metadatas_list=metadatas_list,
        outfolder=outfolder,
    )


def run_simulations(
    output_path,
    n_simulations,
    lc_min_coarse,
    lc_min_fine,
    bulk_multiplicator,
    band_fac_coarse,
    band_fac_fine,
    order,
    plate_width,
    plate_height,
    min_r_frac,
    max_r_frac,
    edge_margin,
    min_gap,
    max_holes,
    force_min,
    force_max,
    E,
    nu,
    solver_max_its=(1, 2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256),
    mesh_fidelity=0.9,
    save_images_every=100,
    n_workers=32,
):
    solver_max_its = [int(v) for v in solver_max_its]
    if len(solver_max_its) == 0:
        raise ValueError("solver_max_its cannot be empty.")
    if any(v <= 0 for v in solver_max_its):
        raise ValueError("solver_max_its must contain positive integers.")
    if any(a > b for a, b in zip(solver_max_its[:-1], solver_max_its[1:])):
        raise ValueError("solver_max_its must be non-decreasing.")

    # Warm-up run to compile kernels.
    print("WARMUP: Compiling FEniCS kernels (running one dummy linear-elasticity solve)...")
    warmup_rectangle = sample_geometry(
        plate_width=plate_width,
        plate_height=plate_height,
        min_r_frac=min_r_frac,
        max_r_frac=max_r_frac,
        edge_margin=edge_margin,
        min_gap=min_gap,
        max_holes=min(max_holes, 10),
    )
    junk_path = osp.join(output_path, "warmup_junk")
    os.makedirs(junk_path, exist_ok=True)

    warmup_meshing_params = sample_mesh_params_from_fidelity(
        lc_min_fine=lc_min_fine,
        lc_min_coarse=lc_min_coarse,
        band_fac_fine=band_fac_fine,
        band_fac_coarse=band_fac_coarse,
        bulk_multiplicator=bulk_multiplicator,
        fidelity=float(mesh_fidelity),
    )
    warmup_mesh = build_mesh(
        rectangle=warmup_rectangle, **warmup_meshing_params, order=order
    )
    _ = run_mf_simulations_solver_error(
        mesh=warmup_mesh,
        element_order=order,
        fx=1e5,
        fy=0.0,
        E=E,
        nu=nu,
        solver_max_its_list=[max(2, solver_max_its[0])],
        metadatas_list=[{}],
        outfolder=junk_path,
    )

    futures = {}
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for i in range(1, n_simulations + 1):
            str_idx = str(i).zfill(len(str(n_simulations)))
            sim_folder = osp.join(output_path, str_idx)
            os.makedirs(sim_folder, exist_ok=True)

            rectangle = sample_geometry(
                plate_width=plate_width,
                plate_height=plate_height,
                min_r_frac=min_r_frac,
                max_r_frac=max_r_frac,
                edge_margin=edge_margin,
                min_gap=min_gap,
                max_holes=max_holes,
            )

            # Random traction on right boundary.
            tmag = np.random.uniform(force_min, force_max)
            theta = np.random.uniform(0.0, 2.0 * np.pi)
            fx = float(tmag * np.cos(theta))
            fy = float(tmag * np.sin(theta))

            # Fixed mesh settings for all solver fidelities.
            meshing_params = sample_mesh_params_from_fidelity(
                lc_min_fine=lc_min_fine,
                lc_min_coarse=lc_min_coarse,
                band_fac_fine=band_fac_fine,
                band_fac_coarse=band_fac_coarse,
                bulk_multiplicator=bulk_multiplicator,
                fidelity=float(mesh_fidelity),
            )

            metadatas_list = []
            for max_it in solver_max_its:
                metadatas_list.append(
                    {
                        "geometry": {
                            "width": rectangle.width,
                            "height": rectangle.height,
                            "holes": [
                                {"x": h.x, "y": h.y, "r": h.r}
                                for h in (rectangle.holes or [])
                            ],
                        },
                        "fidelity_raw": int(max_it),
                        "solver_max_its": int(max_it),
                        "mesh_fidelity": float(mesh_fidelity),
                        "meshing_params": meshing_params,
                        "fx": fx,
                        "fy": fy,
                        "E": float(E),
                        "nu": float(nu),
                    }
                )

            futures[i] = ex.submit(
                job,
                rectangle=rectangle,
                meshing_params=meshing_params,
                element_order=order,
                fx=fx,
                fy=fy,
                E=E,
                nu=nu,
                solver_max_its_list=solver_max_its,
                metadatas_list=metadatas_list,
                outfolder=sim_folder,
            )

        visualizations_path = osp.join(output_path, "vizualizations")
        os.makedirs(visualizations_path, exist_ok=True)

        for i, future in tqdm(
            futures.items(),
            desc="Running linear-elasticity solver-error simulations",
            total=n_simulations,
        ):
            try:
                _ = future.result()
            except Exception as e:
                print(f"Simulation {i} generated an exception: {e}")

            if i % save_images_every == 0:
                sim_path = osp.join(output_path, str(i).zfill(len(str(n_simulations))))
                vm_min, vm_max = float("inf"), float("-inf")
                data_list = []
                names_list = []
                for file in os.listdir(sim_path):
                    if file.endswith(".vtu"):
                        sim_file_path = osp.join(sim_path, file)
                        data = pv.read(sim_file_path)
                        vm = data.point_data["vm"]
                        vm_min = min(vm_min, vm.min())
                        vm_max = max(vm_max, vm.max())
                        data_list.append(data)
                        names_list.append(osp.splitext(file)[0])

                for data, name in zip(data_list, names_list):
                    image_path = osp.join(
                        visualizations_path,
                        f"{osp.basename(sim_path)}",
                        f"{name}.png",
                    )
                    os.makedirs(osp.dirname(image_path), exist_ok=True)
                    save_image(
                        grid=data,
                        var_name="vm",
                        min_val=vm_min,
                        max_val=vm_max,
                        img_path=image_path,
                    )


def postprocess_simulations(
    raw_path,
    output_path,
    summary_metadata_keys,
    num_workers=32,
):
    metadata_list = []

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {}
        metadata = {}

        for raw_folder in os.listdir(raw_path):
            if raw_folder in ["warmup_junk", "vizualizations"]:
                continue

            raw_folder_path = osp.join(raw_path, raw_folder)
            out_folder = osp.join(output_path, raw_folder)
            os.makedirs(out_folder, exist_ok=True)

            for file in os.listdir(raw_folder_path):
                if file.endswith(".vtu"):
                    sim_path = osp.join(raw_folder_path, file)
                    out_path = osp.join(out_folder, f"{file[:-4]}.h5")
                    futures[out_path] = ex.submit(
                        postprocess_simulation,
                        sim_path=sim_path,
                        out_path=out_path,
                        metadata_keys_to_keep=summary_metadata_keys,
                    )
                    metadata[out_path] = {
                        "sample_id": raw_folder,
                        "fidelity_id": str(file[:-4]),
                    }

        for out_path, future in tqdm(
            futures.items(),
            desc="Postprocessing linear-elasticity solver-error simulations",
            total=len(futures),
        ):
            _metadata = future.result()
            _metadata["sample_id"] = metadata[out_path]["sample_id"]
            _metadata["fidelity_id"] = metadata[out_path]["fidelity_id"]
            metadata_list.append(_metadata)

    metadata_df = pd.DataFrame(metadata_list)
    metadata_df.to_csv(osp.join(output_path, "metadata.csv"), index=False)


if __name__ == "__main__":
    np.random.seed(42)
    n_simulations = 90000
    save_images_every = 500
    num_workers = 92

    raw_simulations_path = "_data_test/linear_elasticity/solver_truncation/raw"
    processed_simulations_path = "_data_test/linear_elasticity/solver_truncation/processed"

    os.makedirs(raw_simulations_path, exist_ok=True)
    os.makedirs(processed_simulations_path, exist_ok=True)

    solver_max_its = [25000, 26505, 28188, 30095, 32297, 34902, 38089, 42199, 47991, 57893, 262144]

    summary_metadata_keys = [
        "sample_id",
        "fidelity_id",
        "fidelity_raw",
        "solver_max_its",
        "mesh_fidelity",
        "dofs",
        "n_nodes",
        "n_cells",
        "simulation_time",
        "converged",
        "converged_reason",
        "num_iterations",
        "abs_residual",
        "rel_residual",
        "rel_residual_history",
        "rel_l2_error",
        "rel_l2_error_u",
        "rel_l2_error_sigma",
        "rel_l2_error_epsilon",
        "rel_l2_error_vm",
        "strain_energy_fem",
        "compliance_fem",
        "vm_max",
        "vm_p99",
        "vm_mean",
        "max_strain",
        "condition_number",
        "fx",
        "fy",
        "E",
        "nu",
    ]
    order = 1

    os.makedirs(raw_simulations_path, exist_ok=True)
    os.makedirs(processed_simulations_path, exist_ok=True)

    # Meshing parameters
    lc_min_coarse = 0.2
    lc_min_fine = 0.01
    bulk_multiplicator = 5
    band_fac_coarse = 1
    band_fac_fine = 3

    # Geometry parameters
    plate_width = 2.0
    plate_height = 1.0
    min_r_frac = 0.02
    max_r_frac = 0.15
    edge_margin = 0.05
    min_gap = 0.05
    max_holes = 20

    # Loading / material
    force_min = 1e5
    force_max = 1e7
    E = 210e9
    nu = 0.3

    run_simulations(
        output_path=raw_simulations_path,
        n_simulations=n_simulations,
        lc_min_coarse=lc_min_coarse,
        lc_min_fine=lc_min_fine,
        bulk_multiplicator=bulk_multiplicator,
        band_fac_coarse=band_fac_coarse,
        band_fac_fine=band_fac_fine,
        order=order,
        plate_width=plate_width,
        plate_height=plate_height,
        min_r_frac=min_r_frac,
        max_r_frac=max_r_frac,
        edge_margin=edge_margin,
        min_gap=min_gap,
        max_holes=max_holes,
        force_min=force_min,
        force_max=force_max,
        E=E,
        nu=nu,
        solver_max_its=solver_max_its,
        mesh_fidelity=0.9,
        save_images_every=save_images_every,
        n_workers=num_workers,
    )

    postprocess_simulations(
        raw_path=raw_simulations_path,
        output_path=processed_simulations_path,
        summary_metadata_keys=summary_metadata_keys,
        num_workers=num_workers,
    )
