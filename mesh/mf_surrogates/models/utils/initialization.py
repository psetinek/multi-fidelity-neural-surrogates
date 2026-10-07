import math
import torch


# need this to reproduce initialization across different machines
def deterministic_trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """deterministic (no threading) implementation of: https://github.com/pytorch/pytorch/blob/3596e13d45d8a69ae83dd2f260b2fa8b8376f182/torch/nn/init.py#L25"""
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        low = norm_cdf((a - mean) / std)
        high = norm_cdf((b - mean) / std)

        u = torch.rand(tensor.shape, dtype=torch.float64, device="cpu")  # uses global seed, only works deterministically with float64...
        u = low + (high - low) * u
        x = mean + std * math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
        tensor.copy_(x.clamp_(a, b).to(dtype=tensor.dtype, device=tensor.device))
        return tensor
