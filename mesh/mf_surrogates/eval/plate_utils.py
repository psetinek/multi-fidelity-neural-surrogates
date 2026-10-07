import torch

from mf_surrogates.train.loss import calc_mse, calc_rel_l2_loss
from mf_surrogates.eval.utils import add_channelwise_field_splits, add_field_metric_splits


@torch.no_grad()
def eval_epoch(model, valset, valloader, device, forced_conditioning=None):
    model.to(device)
    model.eval()
    nmse, nrel_l2_loss_fields, nrel_l2_loss_channels, mse, rel_l2_loss_fields, rel_l2_loss_channels = [], [], [], [], [], []
    for sample in valloader:
        sample = sample.to(device)
        if forced_conditioning is not None:
            cond = forced_conditioning * torch.ones_like(sample.cond)
            cond = cond.to(device)
        else:
            cond = sample.cond
        preds = model(
            x=sample.x,
            mesh_coords=sample.coords,
            cond=cond,
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

    nmse = torch.mean(torch.cat(nmse, dim=0), dim=0)
    nrel_l2_loss_fields = torch.mean(torch.cat(nrel_l2_loss_fields, dim=0), dim=0)
    nrel_l2_loss_channels = torch.mean(torch.cat(nrel_l2_loss_channels, dim=0), dim=0)
    mse = torch.mean(torch.cat(mse, dim=0), dim=0)
    rel_l2_loss_fields = torch.mean(torch.cat(rel_l2_loss_fields, dim=0), dim=0)
    rel_l2_loss_channels = torch.mean(torch.cat(rel_l2_loss_channels, dim=0), dim=0)

    # log averages
    log_dict = {
        "val/nmse_avg": torch.mean(nmse).item(),
        "val/nrel_l2_loss_fields_avg": torch.mean(nrel_l2_loss_fields).item(),
        "val/nrel_l2_loss_channels_avg": torch.mean(nrel_l2_loss_channels).item(),
        "val/mse_avg": torch.mean(mse).item(),
        "val/rel_l2_loss_fields_avg": torch.mean(rel_l2_loss_fields).item(),
        "val/rel_l2_loss_channels_avg": torch.mean(rel_l2_loss_channels).item(),
    }
    # logs per field/channel
    add_channelwise_field_splits(log_dict, "val/nmse", nmse, valset.y_channels)
    add_channelwise_field_splits(log_dict, "val/nrel_l2_loss_channels", nrel_l2_loss_channels, valset.y_channels)
    add_field_metric_splits(log_dict, "val/nrel_l2_loss_fields", nrel_l2_loss_fields, valset.y_channels)
    add_channelwise_field_splits(log_dict, "val/mse", mse, valset.y_channels)
    add_channelwise_field_splits(log_dict, "val/rel_l2_loss_channels", rel_l2_loss_channels, valset.y_channels)
    add_field_metric_splits(log_dict, "val/rel_l2_loss_fields", rel_l2_loss_fields, valset.y_channels)

    return torch.mean(nmse).item(), log_dict
