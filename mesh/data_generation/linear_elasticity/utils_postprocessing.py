import json
import os.path as osp

import h5py
import numpy as np
import pyvista as pv


def get_features(data, metadata, dtype=np.float64, eps=1e-8):
    # boundary nodes
    data.point_data["original_point_ids"] = np.arange(data.n_points, dtype=int)
    boundary_nodes = data.extract_feature_edges(boundary_edges=True)

    if "vtkOriginalPointIds" in boundary_nodes.point_data:
        boundary_node_ids = np.asarray(
            boundary_nodes.point_data["vtkOriginalPointIds"], dtype=int
        )
    elif "original_point_ids" in boundary_nodes.point_data:
        boundary_node_ids = np.asarray(
            boundary_nodes.point_data["original_point_ids"], dtype=int
        )
    else:
        raise KeyError("Could not identify boundary point id array in PyVista output.")

    is_boundary = np.zeros(data.n_points, dtype=bool)
    is_boundary[boundary_node_ids] = True

    # signed-distance-like feature to nearest boundary component
    distance_left = np.abs(data.points[:, 0] - 0)
    distance_bottom = np.abs(data.points[:, 1] - 0)
    distance_right = np.abs(data.points[:, 0] - metadata["geometry"]["width"])
    distance_top = np.abs(data.points[:, 1] - metadata["geometry"]["height"])
    distances = np.stack([distance_left, distance_bottom, distance_right, distance_top], axis=1)

    distances_holes = []
    for hole in metadata["geometry"]["holes"]:
        center = np.array([hole["x"], hole["y"]])
        radius = hole["r"]
        distance_hole = np.linalg.norm(data.points[:, :2] - center[None, :], axis=1) - radius
        distances_holes.append(distance_hole)

    if len(distances_holes) > 0:
        distances_holes = np.stack(distances_holes, axis=1)
        distances_all = np.concatenate([distances, distances_holes], axis=1)
    else:
        distances_all = distances

    sdf = np.min(distances_all, axis=1)
    sdf[sdf < eps] = 0

    # boundary normals
    normals = np.zeros((data.n_points, 2), dtype=dtype)
    closest_boundary = np.argmin(distances_all, axis=1)

    # hole normals
    if len(metadata["geometry"]["holes"]) > 0:
        hole_edge_indices = is_boundary & (closest_boundary >= 4)
        centers = np.array(
            [[h["x"], h["y"]] for h in metadata["geometry"]["holes"]], dtype=dtype
        )
        for i in range(len(metadata["geometry"]["holes"])):
            indices = np.where(hole_edge_indices & (closest_boundary == i + 4))[0]
            if len(indices) == 0:
                continue
            vecs = data.points[indices, :2] - centers[i][None, :]
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + eps
            normals[indices, :] = vecs

    # outer boundary normals
    left_indices = np.abs(data.points[:, 0] - 0) < eps
    bottom_indices = np.abs(data.points[:, 1] - 0) < eps
    right_indices = np.abs(data.points[:, 0] - metadata["geometry"]["width"]) < eps
    top_indices = np.abs(data.points[:, 1] - metadata["geometry"]["height"]) < eps
    normals[left_indices, 0] = 1
    normals[bottom_indices, 1] = 1
    normals[right_indices, 0] = -1
    normals[top_indices, 1] = -1

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= norms + eps

    return is_boundary.astype(np.uint8), sdf.astype(dtype), normals.astype(dtype)


def postprocess_simulation(
    sim_path,
    out_path,
    metadata_keys_to_keep=("simulation_time", "dofs"),
    compute_edge_index=False,
    storing_dtype=np.float64,
    x_tol=1e-5,
):
    data = pv.get_reader(sim_path).read()

    # adjacency list (optional)
    if compute_edge_index:
        edges = data.extract_all_edges()
        lines = edges.lines.reshape(-1, 3)
        if not np.all(lines[:, 0] == 2):
            raise ValueError("Unexpected line cell format from PyVista (expected [2, i, j]).")
        edge_index_directed = lines[:, 1:]
        n_edges = edge_index_directed.shape[0]
        edge_index_undirected = np.empty((2, 2 * n_edges), dtype=np.int64)
        edge_index_undirected[:, 0::2] = edge_index_directed.T
        edge_index_undirected[:, 1::2] = edge_index_directed[:, ::-1].T

    # node coordinates
    coords = np.array(data.points, dtype=storing_dtype)[:, :2]

    # displacement (keep only x/y)
    u_raw = np.array(data.point_data["u"], dtype=storing_dtype)
    if u_raw.ndim == 1:
        u = u_raw[:, None]
    elif u_raw.shape[1] >= 2:
        u = u_raw[:, :2]
    else:
        raise ValueError(f"Unexpected displacement shape {u_raw.shape} in {sim_path}")

    # Cauchy stress tensor
    sigma = np.array(data.point_data["sigma"], dtype=storing_dtype)
    if sigma.ndim != 2:
        raise ValueError(f"Unexpected sigma shape {sigma.shape} in {sim_path}")
    if sigma.shape[1] == 4:
        if np.allclose(sigma[:, 1], sigma[:, 2]):
            sigma = sigma[:, [0, 1, 3]]
        else:
            raise ValueError("Expected sigma_xy == sigma_yx for symmetric stress tensor.")
    elif sigma.shape[1] != 3:
        raise ValueError(f"Unexpected sigma channel count {sigma.shape[1]} in {sim_path}")

    # Green-Lagrange strain tensor
    epsilon = np.array(data.point_data["epsilon"], dtype=storing_dtype)
    if epsilon.ndim != 2:
        raise ValueError(f"Unexpected epsilon shape {epsilon.shape} in {sim_path}")
    if epsilon.shape[1] == 4:
        if np.allclose(epsilon[:, 1], epsilon[:, 2]):
            epsilon = epsilon[:, [0, 1, 3]]
        else:
            raise ValueError("Expected epsilon_xy == epsilon_yx for symmetric strain tensor.")
    elif epsilon.shape[1] != 3:
        raise ValueError(f"Unexpected epsilon channel count {epsilon.shape[1]} in {sim_path}")

    # Von-Mises stress
    vm = np.array(data.point_data["vm"], dtype=storing_dtype)
    if vm.ndim == 1:
        vm = vm[:, None]
    elif vm.ndim == 2 and vm.shape[1] == 1:
        pass
    else:
        raise ValueError(f"Unexpected vm shape {vm.shape} in {sim_path}")

    # mesh connectivity for plotting/evaluation
    tri_conn = data.cells_dict[pv.CellType.TRIANGLE].reshape(-1, 3).astype(np.int32)

    # metadata
    metadata_path = osp.join(
        osp.dirname(sim_path),
        osp.splitext(osp.basename(sim_path))[0] + "_metadata.json",
    )
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # nodal force field on right boundary via edge lumping
    fx = float(metadata.get("fx", 0.0))
    fy = float(metadata.get("fy", 0.0))
    traction = np.array([fx, fy], dtype=storing_dtype)

    x_max = coords[:, 0].max()
    right_node_indices = np.where(np.abs(coords[:, 0] - x_max) < x_tol)[0]

    edges = np.concatenate(
        [
            tri_conn[:, [0, 1]],
            tri_conn[:, [1, 2]],
            tri_conn[:, [2, 0]],
        ],
        axis=0,
    )
    edges = np.sort(edges, axis=1)

    mask_0 = np.isin(edges[:, 0], right_node_indices)
    mask_1 = np.isin(edges[:, 1], right_node_indices)
    right_edges = np.unique(edges[mask_0 & mask_1], axis=0)

    force_field = np.zeros_like(coords)
    if right_edges.size > 0:
        p1 = coords[right_edges[:, 0]]
        p2 = coords[right_edges[:, 1]]
        lengths = np.linalg.norm(p1 - p2, axis=1)
        force_per_edge_node = 0.5 * lengths[:, None] * traction[None, :]
        np.add.at(force_field, right_edges[:, 0], force_per_edge_node)
        np.add.at(force_field, right_edges[:, 1], force_per_edge_node)

    fidelity_raw = metadata.get("fidelity_raw", 0.0)

    # engineered geometric features
    is_boundary, sdf, normals = get_features(data=data, metadata=metadata, dtype=storing_dtype)

    # save h5
    with h5py.File(out_path, "w") as h5f:
        cond = h5f.create_group("cond")
        cond.create_dataset(
            "fidelity_raw",
            data=fidelity_raw,
            dtype=storing_dtype,
            compression=None,
            shuffle=False,
        )

        fields_list = [
            coords,
            u,
            sigma,
            epsilon,
            vm,
            force_field,
            is_boundary[:, None],
            sdf[:, None],
            normals,
        ]
        fields_names = [
            "coords",
            "u",
            "sigma",
            "epsilon",
            "vm",
            "force",
            "is_boundary",
            "sdf",
            "normals",
        ]

        channels = h5f.create_group("channels")
        curr_idx = 0
        for name, field in zip(fields_names, fields_list):
            _slice = np.arange(curr_idx, curr_idx + field.shape[-1])
            channels.create_dataset(
                name,
                data=_slice,
                dtype=np.int64,
                compression=None,
                shuffle=False,
            )
            curr_idx += field.shape[-1]

        data_group = h5f.create_group("data")
        data_group.create_dataset(
            "fields",
            data=np.concatenate(fields_list, axis=-1),
            dtype=storing_dtype,
            compression=None,
            shuffle=False,
        )

        if compute_edge_index:
            data_group.create_dataset(
                "edge_index",
                data=edge_index_undirected,
                dtype=np.int64,
                compression=None,
                shuffle=False,
            )

        mesh_group = h5f.create_group("mesh")
        mesh_group.create_dataset(
            "connectivity",
            data=tri_conn,
            dtype=np.int32,
            shuffle=False,
        )

        h5f.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)

    return {key: metadata.get(key, None) for key in metadata_keys_to_keep}
