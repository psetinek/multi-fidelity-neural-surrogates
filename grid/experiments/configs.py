"""ExperimentConfig — the shared dataclass driving one run — plus
`build_model` for architecture dispatch.

`kw_only=True` lets us
intersperse required and defaulted fields freely. Adds `log_every` for
training-progress cadence and `folder_name()` for stable on-disk paths.
"""

from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from architectures import FiLMFNO, FiLMUNet


@dataclass(kw_only=True)
class ExperimentConfig:
    axis: str  # "truncation" | "discretization"
    dim: int  # 1, 2, 3
    fidelity_values: tuple
    lambd: float
    budget: float
    seed: int
    architecture: str  # "unet" | "fno"

    target_resolution: int = 64

    # training
    num_training_steps: int = 20_000
    num_warmup_steps: int = 2_000
    batch_size: int = 20
    learning_rate: float = 1e-4
    log_every: int = 100

    # test
    num_test_samples: int = 100
    test_seed: int = 42

    # RHS / domain
    num_modes: int = 15
    domain_extent: float = 1.0

    # architecture (consumed by the selected arch only)
    base_channels: int = 64
    num_levels: int = 4  # UNet
    # UNet FiLM conditioning width. `None` couples it to the trunk as
    # `4 * base_channels`, which reproduces the historical fixed 256 at the
    # default base_channels=64. Coupling matters for the architecture-size
    # ablation: held at 256, the conditioning network is 30% of parameters at
    # base_channels=16 but 0.3% at 128, so the pathway that encodes fidelity
    # would vary 100x in relative capacity across the sweep.
    cond_dim: int | None = None
    num_modes_fno: int = 12  # FNO
    fno_num_blocks: int = 4  # FNO
    fno_hidden_channels: int = 64  # FNO
    fno_pad_ratio: float = 1.5  # FNO

    # whitening
    num_whitening_samples: int = 8_000

    # dataset strategy
    cache_dataset: bool = False  # True = pre-materialise into RAM for faster steps
    cache_location: str = "gpu"  # "gpu" | "cpu" — where cached dataset lives; "cpu"
    # spills into host RAM, JIT transfers per batch.

    # Trained-weights persistence. False (default) → discard model.eqx
    # immediately after writing result.csv. Critical for big sweeps
    # (1D/2D × multi-seed easily writes ~80 GB of weights that nothing
    # downstream consumes — analysis reads only result.csv). Set True
    # only if you genuinely need to reload trained weights post-hoc
    # (e.g., to score on new test sets without re-training).
    save_model_weights: bool = False

    # Discretization prolongation: resample native-N samples to target_shape via
    # the DST-I (Dirichlet sine) basis instead of the default linear interpolation
    # (`prolongate`). DST resampling is spectrally exact for the sine modes, so it
    # removes interpolation artifacts, leaving pure resolution error as the
    # low-fidelity signal. Ablation only — changes the meaning of "low fidelity".
    dst_interpolation: bool = False

    # Also report prediction error against a super-fidelity reference solution
    # (in addition to the within-menu P_h(f; ξ_max) target). Truncation: the
    # DST-exact discrete solve at target_resolution (`solve_direct`), i.e. the
    # K→∞ limit the MG menu top approximates. Discretization: a DST solve on an
    # ultra-fine grid (see `multifidelity.evaluation`). Answers "error w.r.t. the
    # reference solution" without changing the training target.
    eval_super_reference: bool = False

    # Per-step training-loss trace. False (default) → no loss_history.csv;
    # runs write only result.csv + whitening.npz. Used for the architecture-size
    # convergence check (20k vs 40k steps); enable when a run's convergence
    # behaviour is itself the object of study.
    save_loss_history: bool = False

    def folder_name(self) -> str:
        return (
            f"{self.architecture}_dim_{self.dim}"
            f"_lambd_{self.lambd}_budget_{self.budget}_seed_{self.seed}"
        )


# Register with Hydra's ConfigStore so `@hydra.main`-decorated entry points
# get type-checked overrides against the dataclass schema.
ConfigStore.instance().store(name="experiment_config_schema", node=ExperimentConfig)


def resolve_cond_dim(cfg: ExperimentConfig) -> int:
    """UNet conditioning width, resolving the `None` → `4 * base_channels` rule.

    Shared by `build_model` and the runners' result-CSV writers so the
    coupling rule lives in exactly one place. Meaningless for the FNO
    branch, which carries its own conditioning sizing.
    """
    if cfg.cond_dim is not None:
        return int(cfg.cond_dim)
    return 4 * cfg.base_channels


def build_model(cfg: ExperimentConfig, *, key):
    """Dispatch to the model constructor named by `cfg.architecture`."""
    xi_max = float(max(cfg.fidelity_values))
    if cfg.architecture == "unet":
        return FiLMUNet(
            num_spatial_dims=cfg.dim,
            t_scale=xi_max,
            num_levels=cfg.num_levels,
            base_channels=cfg.base_channels,
            cond_dim=resolve_cond_dim(cfg),
            key=key,
        )
    if cfg.architecture == "fno":
        return FiLMFNO(
            num_spatial_dims=cfg.dim,
            t_scale=xi_max,
            num_blocks=cfg.fno_num_blocks,
            hidden_channels=cfg.fno_hidden_channels,
            num_modes=cfg.num_modes_fno,
            pad_ratio=cfg.fno_pad_ratio,
            key=key,
        )
    raise ValueError(f"Unknown architecture {cfg.architecture!r}")
