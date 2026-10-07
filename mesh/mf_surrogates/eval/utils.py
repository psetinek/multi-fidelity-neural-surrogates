import torch


def add_channelwise_field_splits(log_dict, prefix, metric, y_channels):
    # per-channel metric (n_channels,) -> one entry per field, averaged over the field's channels
    for field_name, _slice in y_channels.items():
        log_dict[f"{prefix}_{field_name}"] = torch.nanmean(metric[_slice]).item()


def add_field_metric_splits(log_dict, prefix, metric, y_channels):
    # per-field metric (n_fields,) -> one entry per field
    for i, field_name in enumerate(y_channels.keys()):
        log_dict[f"{prefix}_{field_name}"] = metric[i].item()
