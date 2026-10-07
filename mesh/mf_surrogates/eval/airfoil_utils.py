import torch

from mf_surrogates.train.loss import calc_mse, calc_rel_l2_loss
from mf_surrogates.eval.utils import add_channelwise_field_splits, add_field_metric_splits


# Air properties
MOL = 28.965338e-3
P_REF = 1.01325e5
R_GAS = 8.3144621


def _get_surface_mask(sample, dataset):
    if not hasattr(dataset, "x_channels") or "on_surface" not in dataset.x_channels:
        raise KeyError("Dataset must provide x_channels['on_surface'] for surface/volume metric split.")

    on_surface_slice = dataset.x_channels["on_surface"]
    # expected to be a single channel, but average if multiple channels are provided.
    on_surface_vals = sample.x[:, on_surface_slice]
    if on_surface_vals.ndim == 2 and on_surface_vals.shape[1] > 1:
        on_surface_vals = on_surface_vals.mean(dim=1)
    else:
        on_surface_vals = on_surface_vals.reshape(-1)
    return on_surface_vals > 0.5


def _compute_subset_metrics(preds, gt, batch_index, mask, y_channels, eps=1e-8):
    n_graphs = int(batch_index.max().item()) + 1
    n_channels = preds.shape[-1]
    device = preds.device
    dtype = preds.dtype

    mask_f = mask.to(dtype=dtype)
    # number of selected nodes per graph.
    n_sel = torch.zeros(n_graphs, device=device, dtype=dtype)
    n_sel.scatter_add_(0, batch_index, mask_f)
    valid_graph = n_sel > 0

    expanded_batch = batch_index.view(-1, 1).expand(-1, n_channels)
    se = (preds - gt) ** 2
    gt_sq = gt ** 2

    se_sum = torch.zeros(n_graphs, n_channels, device=device, dtype=dtype)
    gt_sq_sum = torch.zeros(n_graphs, n_channels, device=device, dtype=dtype)
    se_sum.scatter_add_(0, expanded_batch, se * mask_f.view(-1, 1))
    gt_sq_sum.scatter_add_(0, expanded_batch, gt_sq * mask_f.view(-1, 1))

    mse_per_channel = torch.full_like(se_sum, float("nan"))
    if valid_graph.any():
        mse_per_channel[valid_graph] = se_sum[valid_graph] / n_sel[valid_graph].view(-1, 1)

    rel_l2_per_channel = torch.full_like(se_sum, float("nan"))
    valid_ch = valid_graph.view(-1, 1) & (gt_sq_sum > eps)
    rel_l2_per_channel[valid_ch] = torch.sqrt(
        se_sum[valid_ch] / (gt_sq_sum[valid_ch] + eps)
    )

    rel_l2_fields = []
    for _slice in y_channels.values():
        field_err = se_sum[:, _slice].sum(dim=1)
        field_gt = gt_sq_sum[:, _slice].sum(dim=1)
        rel_f = torch.full((n_graphs,), float("nan"), device=device, dtype=dtype)
        valid_f = valid_graph & (field_gt > eps)
        rel_f[valid_f] = torch.sqrt(field_err[valid_f] / (field_gt[valid_f] + eps))
        rel_l2_fields.append(rel_f)
    rel_l2_per_field = torch.stack(rel_l2_fields, dim=1)

    return mse_per_channel, rel_l2_per_field, rel_l2_per_channel


def _stack_nanmean(per_graph_tensors):
    return torch.nanmean(torch.cat(per_graph_tensors, dim=0), dim=0)


def _fluid_density(temp_k: float) -> float:
    return P_REF * MOL / (R_GAS * float(temp_k))


def _r2_score_1d(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-12) -> float:
    y_true = y_true.to(torch.float64).reshape(-1)
    y_pred = y_pred.to(torch.float64).reshape(-1)
    if y_true.numel() == 0 or y_true.numel() != y_pred.numel():
        return float("nan")

    ss_res = torch.sum((y_true - y_pred) ** 2)
    y_true_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_true_mean) ** 2)
    if ss_tot.item() <= eps:
        return float("nan")
    return (1.0 - (ss_res / (ss_tot + eps))).item()


def _compute_cd_cl_from_graph_denorm(
    y_graph_denorm: torch.Tensor,
    y_channels: dict,
    simulation_params: dict,
    aero_to_internal: torch.Tensor,
    aero_line_connectivity: torch.Tensor,
    aero_line_length: torch.Tensor,
    aero_line_normals: torch.Tensor,
):
    device = y_graph_denorm.device
    dtype = torch.float64

    aoa_deg = float(simulation_params["aoa"])
    u_inf = float(simulation_params["Uinf"])
    temperature_k = float(simulation_params.get("temperature", 298.15))
    rho = _fluid_density(temperature_k)

    p_internal = y_graph_denorm[:, y_channels["p"]].squeeze(-1).to(dtype=dtype)
    wss_internal = y_graph_denorm[:, y_channels["wss"]].to(dtype=dtype)

    a2i = aero_to_internal.long().to(device=device)
    conn = aero_line_connectivity.long().to(device=device)
    length = aero_line_length.to(device=device, dtype=dtype)
    normals = aero_line_normals.to(device=device, dtype=dtype)

    p_surf = p_internal[a2i]
    tau_surf = -wss_internal[a2i]

    i = conn[:, 0]
    j = conn[:, 1]
    p_cell = 0.5 * (p_surf[i] + p_surf[j])
    tau_cell = 0.5 * (tau_surf[i] + tau_surf[j])

    wp_int = -(p_cell[:, None] * normals)
    wp_int = (wp_int * length[:, None]).sum(dim=0)
    wss_int = (tau_cell * length[:, None]).sum(dim=0)

    force_p = -wp_int * rho
    force_v = wss_int * rho

    aoa = torch.deg2rad(torch.tensor(aoa_deg, dtype=dtype, device=device))
    basis = torch.stack(
        [
            torch.stack([torch.cos(aoa), torch.sin(aoa)]),
            torch.stack([-torch.sin(aoa), torch.cos(aoa)]),
        ]
    )

    fp_rot = basis @ force_p
    fv_rot = basis @ force_v
    q_inf = max(0.5 * rho * u_inf * u_inf, 1e-15)

    cp = fp_rot / q_inf
    cv = fv_rot / q_inf
    cd = (cp[0] + cv[0]).item()
    cl = (cp[1] + cv[1]).item()
    return cd, cl


@torch.no_grad()
def eval_epoch(model, valset, valloader, device):
    # validation during training. The model-selection criterion "val/rel_l2_loss_fields_avg" is
    # the sum of the per-channel rel-L2 on volume nodes (u, p) and on surface nodes (p, wss).
    model.to(device)
    model.eval()

    rel_l2_vol_list = []
    rel_l2_surf_list = []

    for sample in valloader:
        sample = sample.to(device)
        preds = model(
            x=sample.x,
            mesh_coords=sample.coords,
            cond=sample.cond,
            batch_index=sample.batch_index,
        )

        preds_denorm = valset.denormalize(preds)
        gt_denorm = valset.denormalize(sample.y)

        on_surface_idx = valset.x_channels["on_surface"]
        surface_mask = (sample.x[:, on_surface_idx] > 0.5).squeeze()
        volume_mask = ~surface_mask

        u_idx = valset.y_channels["u"]
        p_idx = valset.y_channels["p"]
        wss_idx = valset.y_channels["wss"]
        vol_channels = torch.cat([u_idx, p_idx])
        surf_channels = torch.cat([p_idx, wss_idx])

        rel_l2_vol = calc_rel_l2_loss(
            preds=preds_denorm[volume_mask][:, vol_channels],
            gt=gt_denorm[volume_mask][:, vol_channels],
            batch_index=sample.batch_index[volume_mask],
            kind="per_channel",
            eps=1e-8,
        )
        rel_l2_surf = calc_rel_l2_loss(
            preds=preds_denorm[surface_mask][:, surf_channels],
            gt=gt_denorm[surface_mask][:, surf_channels],
            batch_index=sample.batch_index[surface_mask],
            kind="per_channel",
            eps=1e-8,
        )

        rel_l2_vol_list.append(rel_l2_vol)
        rel_l2_surf_list.append(rel_l2_surf)

    vol_avg = torch.mean(torch.cat(rel_l2_vol_list, dim=0)).item()
    surf_avg = torch.mean(torch.cat(rel_l2_surf_list, dim=0)).item()
    total = vol_avg + surf_avg

    log_dict = {
        "val/rel_l2_volume": vol_avg,
        "val/rel_l2_surface": surf_avg,
        "val/rel_l2_total": total,
        "val/rel_l2_loss_fields_avg": total,
    }

    return total, log_dict


@torch.no_grad()
def eval_epoch_with_coeffs(model, valset, valloader, device):
    # full test evaluation: field metrics (all/surface/volume nodes) + Cd/Cl.
    model.to(device)
    model.eval()

    nmse, nrel_l2_loss_fields, nrel_l2_loss_channels, mse, rel_l2_loss_fields, rel_l2_loss_channels = [], [], [], [], [], []
    surface_nmse, surface_nrel_l2_loss_fields, surface_nrel_l2_loss_channels = [], [], []
    surface_mse, surface_rel_l2_loss_fields, surface_rel_l2_loss_channels = [], [], []
    volume_nmse, volume_nrel_l2_loss_fields, volume_nrel_l2_loss_channels = [], [], []
    volume_mse, volume_rel_l2_loss_fields, volume_rel_l2_loss_channels = [], [], []

    rel_err_cd, rel_err_cl = [], []
    abs_err_cd, abs_err_cl = [], []
    cd_calc_all, cl_calc_all = [], []
    cd_gt_all, cl_gt_all = [], []

    dataset_offset = 0

    for sample in valloader:
        sample = sample.to(device)
        preds = model(
            x=sample.x,
            mesh_coords=sample.coords,
            cond=sample.cond,
            batch_index=sample.batch_index,
        )
        # normalized metrics
        _nmse = calc_mse(preds, sample.y, sample.batch_index)
        _nrel_l2_loss_fields = calc_rel_l2_loss(preds, sample.y, batch_index=sample.batch_index, kind="per_field", y_channels=valset.y_channels, eps=1e-8)
        _nrel_l2_loss_channels = calc_rel_l2_loss(preds, sample.y, batch_index=sample.batch_index, kind="per_channel", y_channels=valset.y_channels, eps=1e-8)
        nmse.append(_nmse)
        nrel_l2_loss_fields.append(_nrel_l2_loss_fields)
        nrel_l2_loss_channels.append(_nrel_l2_loss_channels)
        # denormalized metrics
        preds_denormalized = valset.denormalize(preds)
        y_denormalized = valset.denormalize(sample.y)
        _mse = calc_mse(preds_denormalized, y_denormalized, sample.batch_index)
        _rel_l2_loss_fields = calc_rel_l2_loss(preds_denormalized, y_denormalized, batch_index=sample.batch_index, kind="per_field", y_channels=valset.y_channels, eps=1e-8)
        _rel_l2_loss_channels = calc_rel_l2_loss(preds_denormalized, y_denormalized, batch_index=sample.batch_index, kind="per_channel", y_channels=valset.y_channels, eps=1e-8)
        mse.append(_mse)
        rel_l2_loss_fields.append(_rel_l2_loss_fields)
        rel_l2_loss_channels.append(_rel_l2_loss_channels)

        # surface/volume split metrics
        surface_mask = _get_surface_mask(sample, valset)
        volume_mask = ~surface_mask

        _surface_nmse, _surface_nrel_f, _surface_nrel_c = _compute_subset_metrics(
            preds, sample.y, sample.batch_index, surface_mask, valset.y_channels, eps=1e-8
        )
        _volume_nmse, _volume_nrel_f, _volume_nrel_c = _compute_subset_metrics(
            preds, sample.y, sample.batch_index, volume_mask, valset.y_channels, eps=1e-8
        )
        surface_nmse.append(_surface_nmse)
        surface_nrel_l2_loss_fields.append(_surface_nrel_f)
        surface_nrel_l2_loss_channels.append(_surface_nrel_c)
        volume_nmse.append(_volume_nmse)
        volume_nrel_l2_loss_fields.append(_volume_nrel_f)
        volume_nrel_l2_loss_channels.append(_volume_nrel_c)

        _surface_mse, _surface_rel_f, _surface_rel_c = _compute_subset_metrics(
            preds_denormalized, y_denormalized, sample.batch_index, surface_mask, valset.y_channels, eps=1e-8
        )
        _volume_mse, _volume_rel_f, _volume_rel_c = _compute_subset_metrics(
            preds_denormalized, y_denormalized, sample.batch_index, volume_mask, valset.y_channels, eps=1e-8
        )
        surface_mse.append(_surface_mse)
        surface_rel_l2_loss_fields.append(_surface_rel_f)
        surface_rel_l2_loss_channels.append(_surface_rel_c)
        volume_mse.append(_volume_mse)
        volume_rel_l2_loss_fields.append(_volume_rel_f)
        volume_rel_l2_loss_channels.append(_volume_rel_c)

        # Cd/Cl metrics (cell normals mode, per graph in batch)
        batch_size = int(sample.batch_index.max().item()) + 1
        counts = torch.bincount(sample.batch_index, minlength=batch_size).tolist()
        pred_chunks = torch.split(preds_denormalized, counts, dim=0)

        for b_idx, pred_graph in enumerate(pred_chunks):
            ds_idx = dataset_offset + b_idx
            if ds_idx >= len(valset):
                continue

            if (
                valset.aero_line_connectivity[ds_idx] is None
                or valset.aero_line_length[ds_idx] is None
                or valset.aero_line_normals[ds_idx] is None
                or valset.aero_to_internal[ds_idx] is None
                or valset.cd_openfoam[ds_idx] is None
                or valset.cl_openfoam[ds_idx] is None
                or "simulation_params" not in valset.metadata[ds_idx]
            ):
                continue

            cd_calc, cl_calc = _compute_cd_cl_from_graph_denorm(
                y_graph_denorm=pred_graph,
                y_channels=valset.y_channels,
                simulation_params=valset.metadata[ds_idx]["simulation_params"],
                aero_to_internal=valset.aero_to_internal[ds_idx],
                aero_line_connectivity=valset.aero_line_connectivity[ds_idx],
                aero_line_length=valset.aero_line_length[ds_idx],
                aero_line_normals=valset.aero_line_normals[ds_idx],
            )

            cd_gt = float(valset.cd_openfoam[ds_idx])
            cl_gt = float(valset.cl_openfoam[ds_idx])

            abs_cd = abs(cd_calc - cd_gt)
            abs_cl = abs(cl_calc - cl_gt)
            rel_cd = abs_cd / max(abs(cd_gt), 1e-15)
            rel_cl = abs_cl / max(abs(cl_gt), 1e-15)

            cd_calc_all.append(cd_calc)
            cl_calc_all.append(cl_calc)
            cd_gt_all.append(cd_gt)
            cl_gt_all.append(cl_gt)
            abs_err_cd.append(abs_cd)
            abs_err_cl.append(abs_cl)
            rel_err_cd.append(rel_cd)
            rel_err_cl.append(rel_cl)

        dataset_offset += batch_size

    nmse = torch.mean(torch.cat(nmse, dim=0), dim=0)
    nrel_l2_loss_fields = torch.mean(torch.cat(nrel_l2_loss_fields, dim=0), dim=0)
    nrel_l2_loss_channels = torch.mean(torch.cat(nrel_l2_loss_channels, dim=0), dim=0)
    mse = torch.mean(torch.cat(mse, dim=0), dim=0)
    rel_l2_loss_fields = torch.mean(torch.cat(rel_l2_loss_fields, dim=0), dim=0)
    rel_l2_loss_channels = torch.mean(torch.cat(rel_l2_loss_channels, dim=0), dim=0)

    surface_nmse = _stack_nanmean(surface_nmse)
    surface_nrel_l2_loss_fields = _stack_nanmean(surface_nrel_l2_loss_fields)
    surface_nrel_l2_loss_channels = _stack_nanmean(surface_nrel_l2_loss_channels)
    surface_mse = _stack_nanmean(surface_mse)
    surface_rel_l2_loss_fields = _stack_nanmean(surface_rel_l2_loss_fields)
    surface_rel_l2_loss_channels = _stack_nanmean(surface_rel_l2_loss_channels)

    volume_nmse = _stack_nanmean(volume_nmse)
    volume_nrel_l2_loss_fields = _stack_nanmean(volume_nrel_l2_loss_fields)
    volume_nrel_l2_loss_channels = _stack_nanmean(volume_nrel_l2_loss_channels)
    volume_mse = _stack_nanmean(volume_mse)
    volume_rel_l2_loss_fields = _stack_nanmean(volume_rel_l2_loss_fields)
    volume_rel_l2_loss_channels = _stack_nanmean(volume_rel_l2_loss_channels)

    log_dict = {
        "val/nmse_avg": torch.mean(nmse).item(),
        "val/nrel_l2_loss_fields_avg": torch.mean(nrel_l2_loss_fields).item(),
        "val/nrel_l2_loss_channels_avg": torch.mean(nrel_l2_loss_channels).item(),
        "val/mse_avg": torch.mean(mse).item(),
        "val/rel_l2_loss_fields_avg": torch.mean(rel_l2_loss_fields).item(),
        "val/rel_l2_loss_channels_avg": torch.mean(rel_l2_loss_channels).item(),
        "val/surface_nmse_avg": torch.nanmean(surface_nmse).item(),
        "val/surface_nrel_l2_loss_fields_avg": torch.nanmean(surface_nrel_l2_loss_fields).item(),
        "val/surface_nrel_l2_loss_channels_avg": torch.nanmean(surface_nrel_l2_loss_channels).item(),
        "val/surface_mse_avg": torch.nanmean(surface_mse).item(),
        "val/surface_rel_l2_loss_fields_avg": torch.nanmean(surface_rel_l2_loss_fields).item(),
        "val/surface_rel_l2_loss_channels_avg": torch.nanmean(surface_rel_l2_loss_channels).item(),
        "val/volume_nmse_avg": torch.nanmean(volume_nmse).item(),
        "val/volume_nrel_l2_loss_fields_avg": torch.nanmean(volume_nrel_l2_loss_fields).item(),
        "val/volume_nrel_l2_loss_channels_avg": torch.nanmean(volume_nrel_l2_loss_channels).item(),
        "val/volume_mse_avg": torch.nanmean(volume_mse).item(),
        "val/volume_rel_l2_loss_fields_avg": torch.nanmean(volume_rel_l2_loss_fields).item(),
        "val/volume_rel_l2_loss_channels_avg": torch.nanmean(volume_rel_l2_loss_channels).item(),
    }

    # overall splits (per field)
    add_channelwise_field_splits(log_dict, "val/nmse", nmse, valset.y_channels)
    add_field_metric_splits(log_dict, "val/nrel_l2_loss_fields", nrel_l2_loss_fields, valset.y_channels)
    add_channelwise_field_splits(log_dict, "val/nrel_l2_loss_channels", nrel_l2_loss_channels, valset.y_channels)
    add_channelwise_field_splits(log_dict, "val/mse", mse, valset.y_channels)
    add_field_metric_splits(log_dict, "val/rel_l2_loss_fields", rel_l2_loss_fields, valset.y_channels)
    add_channelwise_field_splits(log_dict, "val/rel_l2_loss_channels", rel_l2_loss_channels, valset.y_channels)

    # surface splits (per field)
    add_channelwise_field_splits(log_dict, "val/surface_nmse", surface_nmse, valset.y_channels)
    add_field_metric_splits(log_dict, "val/surface_nrel_l2_loss_fields", surface_nrel_l2_loss_fields, valset.y_channels)
    add_channelwise_field_splits(log_dict, "val/surface_nrel_l2_loss_channels", surface_nrel_l2_loss_channels, valset.y_channels)
    add_channelwise_field_splits(log_dict, "val/surface_mse", surface_mse, valset.y_channels)
    add_field_metric_splits(log_dict, "val/surface_rel_l2_loss_fields", surface_rel_l2_loss_fields, valset.y_channels)
    add_channelwise_field_splits(log_dict, "val/surface_rel_l2_loss_channels", surface_rel_l2_loss_channels, valset.y_channels)

    # volume splits (per field)
    add_channelwise_field_splits(log_dict, "val/volume_nmse", volume_nmse, valset.y_channels)
    add_field_metric_splits(log_dict, "val/volume_nrel_l2_loss_fields", volume_nrel_l2_loss_fields, valset.y_channels)
    add_channelwise_field_splits(log_dict, "val/volume_nrel_l2_loss_channels", volume_nrel_l2_loss_channels, valset.y_channels)
    add_channelwise_field_splits(log_dict, "val/volume_mse", volume_mse, valset.y_channels)
    add_field_metric_splits(log_dict, "val/volume_rel_l2_loss_fields", volume_rel_l2_loss_fields, valset.y_channels)
    add_channelwise_field_splits(log_dict, "val/volume_rel_l2_loss_channels", volume_rel_l2_loss_channels, valset.y_channels)

    if len(rel_err_cd) > 0:
        rel_cd_t = torch.tensor(rel_err_cd, dtype=torch.float64)
        rel_cl_t = torch.tensor(rel_err_cl, dtype=torch.float64)
        abs_cd_t = torch.tensor(abs_err_cd, dtype=torch.float64)
        abs_cl_t = torch.tensor(abs_err_cl, dtype=torch.float64)
        cd_calc_t = torch.tensor(cd_calc_all, dtype=torch.float64)
        cl_calc_t = torch.tensor(cl_calc_all, dtype=torch.float64)
        cd_gt_t = torch.tensor(cd_gt_all, dtype=torch.float64)
        cl_gt_t = torch.tensor(cl_gt_all, dtype=torch.float64)

        log_dict.update(
            {
                "val/coeff_eval_count": float(len(rel_err_cd)),
                "val/rel_err_cd_avg": rel_cd_t.mean().item(),
                "val/rel_err_cl_avg": rel_cl_t.mean().item(),
                "val/r2_cd": _r2_score_1d(cd_gt_t, cd_calc_t),
                "val/r2_cl": _r2_score_1d(cl_gt_t, cl_calc_t),
                "val/abs_err_cd_avg": abs_cd_t.mean().item(),
                "val/abs_err_cl_avg": abs_cl_t.mean().item(),
                "val/cd_calc_avg": cd_calc_t.mean().item(),
                "val/cl_calc_avg": cl_calc_t.mean().item(),
                "val/cd_gt_avg": cd_gt_t.mean().item(),
                "val/cl_gt_avg": cl_gt_t.mean().item(),
            }
        )
    else:
        log_dict.update(
            {
                "val/coeff_eval_count": 0.0,
                "val/rel_err_cd_avg": float("nan"),
                "val/rel_err_cl_avg": float("nan"),
                "val/r2_cd": float("nan"),
                "val/r2_cl": float("nan"),
                "val/abs_err_cd_avg": float("nan"),
                "val/abs_err_cl_avg": float("nan"),
                "val/cd_calc_avg": float("nan"),
                "val/cl_calc_avg": float("nan"),
                "val/cd_gt_avg": float("nan"),
                "val/cl_gt_avg": float("nan"),
            }
        )

    return torch.mean(nmse).item(), log_dict
