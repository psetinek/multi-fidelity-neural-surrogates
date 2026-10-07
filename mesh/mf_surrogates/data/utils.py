import numpy as np


def get_exp_pmf(fidelities, lambd):
        fids = np.array(fidelities)
        if lambd is None or np.isclose(lambd, 0):
            return np.ones_like(fids) / len(fids)
        # softmax discrete logits to get probabilities
        exponents = lambd * fids
        exponents -= np.max(exponents)
        unnormalized = np.exp(exponents)
        return unnormalized / np.sum(unnormalized)