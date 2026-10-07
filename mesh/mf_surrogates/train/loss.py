import torch
import torch.nn as nn


def calc_mse(preds, gt, batch_index):
    se = (gt - preds)**2
    bs = int(batch_index.max().item()) + 1
    mse_per_graph = torch.zeros([bs, se.shape[-1]], device=se.device, dtype=se.dtype)
    batch_index = batch_index.view(-1, 1).expand(-1, se.shape[-1])
    mse_per_graph.scatter_reduce_(0, batch_index, se, reduce="mean")
    return mse_per_graph  # shape [n_graphs, n_channels]


def calc_rel_l2_loss(preds, gt, batch_index, kind="per_channel", y_channels=None, eps=1e-8):
    err_sq = (preds - gt) ** 2
    true_sq = gt ** 2

    if kind == "per_field":
        assert y_channels is not None
        results = []
        n_graphs = batch_index.max().item() + 1
        for  _slice in y_channels.values():
            # select channels for this field
            field_squared_diff = err_sq[:, _slice]
            field_squared_gt = true_sq[:, _slice]
            # squared vector norm per node
            squared_node_error = field_squared_diff.sum(dim=1)
            squared_node_gt = field_squared_gt.sum(dim=1)
            # scatter sum over graph nodes to get graph-level sums
            squared_graph_error = torch.zeros(n_graphs, device=preds.device, dtype=preds.dtype)
            squared_graph_gt = torch.zeros(n_graphs, device=preds.device, dtype=preds.dtype)
            squared_graph_error.scatter_add_(0, batch_index, squared_node_error)
            squared_graph_gt.scatter_add_(0, batch_index, squared_node_gt)
            nrmse = torch.sqrt(squared_graph_error / (squared_graph_gt + eps))  # shape [n_graphs]
            results.append(nrmse)
        rel_l2 = torch.stack(results, dim=1)  # shape [n_graphs, n_fields]
    
    elif kind == "per_channel":
        expanded_batch_idx = batch_index.unsqueeze(1).expand_as(preds)
        batch_size = int(batch_index.max().item()) + 1
        num_channels = preds.shape[1]

        batch_diff_sum = torch.zeros([batch_size, num_channels], device=preds.device, dtype=preds.dtype)
        batch_diff_sum.scatter_add_(0, expanded_batch_idx, err_sq)
        batch_true_sum = torch.zeros([batch_size, num_channels], device=preds.device, dtype=preds.dtype)
        batch_true_sum.scatter_add_(0, expanded_batch_idx, true_sq)
        # rel L2 per graph [n_graphs, channels]
        rel_l2 = torch.sqrt(batch_diff_sum / (batch_true_sum + eps))
    
    else:
        raise NotImplementedError(f"Unknkown kind: '{kind}'! Relative L2 loss only implemented for 'per_channel' or 'per_field'.")

    return rel_l2


class MSELoss(nn.Module):
    def __init__(self, dataset, normalized, **kwargs):
        super().__init__()
        self.dataset = dataset
        self.normalized = normalized
        _ = kwargs
        if self.normalized:
            self.name = "nmse"
        else:
            self.name = "mse"

    def forward(self, preds, sample):
        gt = sample.y
        if not self.normalized:
            preds = self.dataset.denormalize(preds)
            gt = self.dataset.denormalize(gt)
        return torch.mean(calc_mse(preds, gt, sample.batch_index)), {}


class MSELossSplit(nn.Module):
    def __init__(self, dataset, normalized, surface_loss_weight: float = 1.0, eps: float = 1e-8, **kwargs):
        super().__init__()
        _ = kwargs
        self.dataset = dataset
        self.normalized = normalized
        self.surface_loss_weight = surface_loss_weight
        self.eps = eps
        if self.normalized:
            self.name = "nmse_split"
        else:
            self.name = "mse_split"

    def _masked_mse_per_graph(self, se, batch_index, mask):
        bs = int(batch_index.max().item()) + 1
        n_channels = se.shape[-1]
        device = se.device
        dtype = se.dtype

        mask_f = mask.to(dtype=dtype)
        expanded_batch = batch_index.view(-1, 1).expand(-1, n_channels)

        se_sum = torch.zeros([bs, n_channels], device=device, dtype=dtype)
        se_sum.scatter_add_(0, expanded_batch, se * mask_f.view(-1, 1))

        n_sel = torch.zeros([bs], device=device, dtype=dtype)
        n_sel.scatter_add_(0, batch_index, mask_f)
        valid = n_sel > 0

        mse_per_graph = torch.full([bs, n_channels], float("nan"), device=device, dtype=dtype)
        if valid.any():
            mse_per_graph[valid] = se_sum[valid] / n_sel[valid].view(-1, 1)
        return mse_per_graph

    def forward(self, preds, sample):
        gt = sample.y
        if not self.normalized:
            preds = self.dataset.denormalize(preds)
            gt = self.dataset.denormalize(gt)

        se = (gt - preds) ** 2

        if not hasattr(self.dataset, "x_channels") or "on_surface" not in self.dataset.x_channels:
            # Fallback to standard mse if no surface marker is available.
            loss = torch.mean(calc_mse(preds, gt, sample.batch_index))
            return loss, {}

        on_surface_slice = self.dataset.x_channels["on_surface"]
        on_surface_vals = sample.x[:, on_surface_slice]
        if on_surface_vals.ndim == 2 and on_surface_vals.shape[1] > 1:
            on_surface_vals = on_surface_vals.mean(dim=1)
        else:
            on_surface_vals = on_surface_vals.reshape(-1)

        on_surf = on_surface_vals > 0.5
        in_vol = ~on_surf

        mse_per_graph_surf = self._masked_mse_per_graph(se, sample.batch_index, on_surf)
        mse_per_graph_vol = self._masked_mse_per_graph(se, sample.batch_index, in_vol)

        mse_surf = torch.nanmean(mse_per_graph_surf)
        mse_vol = torch.nanmean(mse_per_graph_vol)

        # If a subset is empty for all graphs, avoid propagating NaN into the objective.
        if torch.isnan(mse_surf):
            mse_surf = torch.zeros_like(mse_vol)
        if torch.isnan(mse_vol):
            mse_vol = torch.zeros_like(mse_surf)

        loss = mse_vol + self.surface_loss_weight * mse_surf
        components = {
            "mse_surf_vol": (mse_vol + self.surface_loss_weight * mse_surf).detach(),
            "mse_surf": mse_surf.detach(),
            "mse_vol": mse_vol.detach(),
        }
        return loss, components


class RelativeL2Loss(nn.Module):
    def __init__(self, dataset, normalized, eps: float = 1e-8, kind="per_channel", **kwargs):
        super().__init__()
        _ = kwargs
        self.dataset = dataset
        self.normalized = normalized
        self.eps = eps
        self.kind = kind
        if self.normalized:
            self.name = f"nrel_l2_error_{kind}"
        else:
            self.name = f"rel_l2_error_{kind}"

    def forward(self, preds, sample):
        gt = sample.y
        if not self.normalized:
            preds = self.dataset.denormalize(preds)
            gt = self.dataset.denormalize(gt)
        return torch.mean(calc_rel_l2_loss(preds, gt, sample.batch_index, kind=self.kind, y_channels=self.dataset.y_channels, eps=self.eps)), {}


class RelativeL2LossAirfoil(nn.Module):
    def __init__(self, dataset, normalized, eps: float = 1e-8, kind="per_channel", **kwargs):
        super().__init__()
        _ = kwargs
        self.dataset = dataset
        self.normalized = normalized
        self.eps = eps
        self.kind = kind
        if self.normalized:
            self.name = f"nrel_l2_error_{kind}"
        else:
            self.name = f"rel_l2_error_{kind}"

    def forward(self, preds, sample):
        gt = sample.y
        if not self.normalized:
            preds = self.dataset.denormalize(preds)
            gt = self.dataset.denormalize(gt)

        # masks
        on_surface_idx = self.dataset.x_channels["on_surface"]
        surface_mask = (sample.x[:, on_surface_idx] > 0.5).squeeze()
        volume_mask = ~surface_mask

        u_idx = self.dataset.y_channels["u"]
        p_idx = self.dataset.y_channels["p"]
        vol_channels = torch.cat([u_idx, p_idx])
        rel_l2_volume = calc_rel_l2_loss(
            preds=preds[volume_mask][:, vol_channels], 
            gt=gt[volume_mask][:, vol_channels], 
            batch_index=sample.batch_index[volume_mask], 
            kind=self.kind, 
            eps=self.eps
        )

        p_idx = self.dataset.y_channels["p"]
        wss_idx = self.dataset.y_channels["wss"]
        surf_channels = torch.cat([p_idx, wss_idx])
        rel_l2_surface = calc_rel_l2_loss(
            preds=preds[surface_mask][:, surf_channels], 
            gt=gt[surface_mask][:, surf_channels], 
            batch_index=sample.batch_index[surface_mask], 
            kind=self.kind,
            eps=self.eps
        )

        rel_l2 = torch.mean(rel_l2_volume) + torch.mean(rel_l2_surface)

        return torch.mean(rel_l2), {}
