import os
import os.path as osp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyvista as pv
from tqdm import tqdm

from simulation_discretization_error import run_mf_simulations_mesh_discretization
from utils_simulations import (
    sample_geometry,
    sample_mesh_params_from_fidelity,
    build_mesh,
    save_image,
)
from utils_postprocessing import postprocess_simulation


def job(
    rectangle,
    meshing_params_list,
    element_order,
    fx,
    fy,
    E,
    nu,
    metadatas_list,
    outfolder,
):
    meshes_list = []
    for meshing_params in meshing_params_list:
        mesh = build_mesh(rectangle=rectangle, **meshing_params, order=element_order)
        meshes_list.append(mesh)

    return run_mf_simulations_mesh_discretization(
        meshes_list=meshes_list,
        element_order=element_order,
        fx=fx,
        fy=fy,
        E=E,
        nu=nu,
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
    meshing_fidelities=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    save_images_every=100,
    n_workers=32,
):
    # warmup JIT so compiled kernels are cached
    print("WARMUP: Compiling FEniCS kernels (running one dummy hyperelastic simulation)...")
    warmup_rectangle = sample_geometry(
        plate_width=plate_width,
        plate_height=plate_height,
        min_r_frac=min_r_frac,
        max_r_frac=max_r_frac,
        edge_margin=edge_margin,
        min_gap=min_gap,
        max_holes=10,
    )
    junk_path = osp.join(output_path, "warmup_junk")
    os.makedirs(junk_path, exist_ok=True)
    meshing_params = sample_mesh_params_from_fidelity(
        lc_min_fine=lc_min_fine,
        lc_min_coarse=lc_min_coarse,
        band_fac_fine=band_fac_fine,
        band_fac_coarse=band_fac_coarse,
        bulk_multiplicator=bulk_multiplicator,
        fidelity=0.0,
    )
    mesh = build_mesh(rectangle=warmup_rectangle, **meshing_params, order=order)
    _ = run_mf_simulations_mesh_discretization(
        meshes_list=[mesh],
        element_order=order,
        fx=1e5,
        fy=0.0,
        E=E,
        nu=nu,
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

            # sample traction vector
            tmag = np.random.uniform(force_min, force_max)
            # theta = np.random.uniform(0.0, 2.0 * np.pi)
            theta = np.random.uniform(-np.pi / 6, np.pi / 6)
            fx = float(tmag * np.cos(theta))
            fy = float(tmag * np.sin(theta))

            meshing_params_list = []
            metadatas_list = []
            for meshing_fidelity in meshing_fidelities:
                meshing_params = sample_mesh_params_from_fidelity(
                    lc_min_fine=lc_min_fine,
                    lc_min_coarse=lc_min_coarse,
                    band_fac_fine=band_fac_fine,
                    band_fac_coarse=band_fac_coarse,
                    bulk_multiplicator=bulk_multiplicator,
                    fidelity=meshing_fidelity,
                )
                meshing_params_list.append(meshing_params)
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
                        "fidelity_raw": float(meshing_fidelity),
                        "meshing_fidelity": float(meshing_fidelity),
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
                meshing_params_list=meshing_params_list,
                element_order=order,
                fx=fx,
                fy=fy,
                E=E,
                nu=nu,
                metadatas_list=metadatas_list,
                outfolder=sim_folder,
            )

        visualizations_path = osp.join(output_path, "vizualizations")
        os.makedirs(visualizations_path, exist_ok=True)

        for i, future in tqdm(
            futures.items(),
            desc="Running hyperelastic simulations",
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
            desc="Postprocessing hyperelastic simulations",
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
    n_simulations = 10000
    save_images_every = 10
    num_workers = 92

    raw_simulations_path = "_data_test/hyperelasticity/discretization/raw"
    processed_simulations_path = "_data_test/hyperelasticity/discretization/processed"

    meshing_fidelities = [0.0, 0.246, 0.354, 0.419, 0.478, 0.527, 0.566, 0.61, 0.645, 0.68, 0.713, 0.745, 0.774, 0.803, 0.833, 0.861, 0.888, 0.916, 0.943, 0.971, 1.0]

    summary_metadata_keys = [
        "sample_id",
        "fidelity_id",
        "fidelity_raw",
        "meshing_fidelity",
        "dofs",
        "n_nodes",
        "n_cells",
        "simulation_time",
        "converged",
        "num_iterations",
        "rel_residual",
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
        "fx",
        "fy",
        "E",
        "nu",
    ]
    order = 1

    os.makedirs(raw_simulations_path, exist_ok=True)
    os.makedirs(processed_simulations_path, exist_ok=True)

    # meshing
    lc_min_coarse = 0.2
    lc_min_fine = 0.005
    bulk_multiplicator = 5
    band_fac_coarse = 1
    band_fac_fine = 3

    # geometry
    plate_width = 2.0
    plate_height = 1.0
    min_r_frac = 0.02
    max_r_frac = 0.15
    edge_margin = 0.05
    min_gap = 0.05
    max_holes = 20

    # loading and material
    E = 2e6
    nu = 0.3
    force_min = 1e3
    force_max = 5e4


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
        meshing_fidelities=meshing_fidelities,
        save_images_every=save_images_every,
        n_workers=num_workers,
    )

    postprocess_simulations(
        raw_path=raw_simulations_path,
        output_path=processed_simulations_path,
        summary_metadata_keys=summary_metadata_keys,
        num_workers=num_workers,
    )
