import os.path as osp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import json
import pandas as pd
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import time

from .utils import get_exp_pmf


@dataclass(kw_only=True)
class BaseSample:
    coords: torch.Tensor
    connectivity: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    cond: Optional[torch.Tensor] = None
    batch_index: Optional[torch.Tensor] = None
    connectivity_batch_index: Optional[torch.Tensor] = None
    pad_mask: Optional[torch.Tensor] = None
    # for evaluation & metadata
    metadata: Optional[dict] = None

    def to(self, device, **kwargs):
        self.coords = self.coords.to(device, **kwargs)
        self.connectivity = self.connectivity.to(device, **kwargs)
        self.x = self.x.to(device, **kwargs)
        self.y = self.y.to(device, **kwargs)
        if self.cond is not None:
            self.cond = self.cond.to(device, **kwargs)
        if self.batch_index is not None:
            self.batch_index = self.batch_index.to(device, **kwargs)
        if self.connectivity_batch_index is not None:
            self.connectivity_batch_index = self.connectivity_batch_index.to(device, **kwargs)
        if self.pad_mask is not None:
            self.pad_mask = self.pad_mask.to(device, **kwargs)
        return self


class BaseMFDataset(Dataset, ABC):
    def __init__(
        self,
        data_path,
        metadata_path=None,
        split="train",
        val_fraction=0.1,
        test_fraction=0.1,
        budget=None,
        linearization_col="n_nodes",
        lambd=None,
        lambda_bounds=(-15, 15),
        n_samples_target=2000,
        hf_only=False,
        lf_only=False,
        q = None,
        cost_column = "simulation_time",
        composition_rng=None,
        n_nodes_subsampling=None,
        dtype=torch.float32,
        **kwargs,
    ):
        _ = kwargs
        self.data_path = data_path
        self.metadata_path = metadata_path if metadata_path is not None else osp.join(data_path, "metadata.csv")
        self.split = split
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction

        self.linearization_col = linearization_col
        self.lambd = lambd
        self.n_samples_target = n_samples_target
        self.lambda_bounds = lambda_bounds
        self.hf_only = hf_only
        self.lf_only = lf_only
        self.q = q
        self.cost_column = cost_column
        self.composition_rng = composition_rng
        self.n_nodes_subsampling = n_nodes_subsampling
        self.dtype = dtype

        self.metadata_df = pd.read_csv(self.metadata_path, dtype={"sample_id": str, "fidelity_raw": float, "simulation_time": float, "fidelity_id": str} if 'fidelity_id' in pd.read_csv(self.metadata_path, nrows=1).columns else {"sample_id": str, "fidelity_raw": float, "simulation_time": float})
        self.budget = budget if budget is not None else 1e9

        # load sample_ids
        sample_ids = sorted(self.metadata_df["sample_id"].unique().tolist())
        self.composition_rng.shuffle(sample_ids)
        n_samples = len(sample_ids)
        print(f"Total samples available: {n_samples}. Budget: {self.budget}. Target samples: {self.n_samples_target}.")
        train_frac = 1 - val_fraction - test_fraction
        # linearize fidelities
        self._linearize_fidelities()
        # O(1) per-sample lookup for loading.
        self._fidelity_lookup = dict(zip(
            zip(self.metadata_df["sample_id"], self.metadata_df["fidelity_id"]),
            self.metadata_df["fidelity_linearized"],
        ))
        if self.hf_only:
            self.fidelities_linearized = [self.fidelities_linearized[-1]]
        if self.lf_only:
            self.fidelities_linearized = [self.fidelities_linearized[0]]
        if self.split == "train":
            self.sample_ids = sample_ids[:int(n_samples * train_frac)]
        elif self.split == "val":
            # hf only
            self.budget = 1e9
            self.lambd = 1e9
            self.fidelities_linearized = [self.fidelities_linearized[-1]]
            self.sample_ids = sample_ids[int(n_samples * train_frac): int(n_samples * (train_frac + val_fraction))]
        elif self.split == "test":
            # hf only
            self.budget = 1e9
            self.lambd = 1e9
            self.fidelities_linearized = [self.fidelities_linearized[-1]]
            self.sample_ids = sample_ids[int(n_samples * (train_frac + val_fraction)):]
        else:
            raise ValueError("Invalid split!")

        # calculate lambda if not provided and n_samples_target is provided
        if self.lambd is None and self.n_samples_target is not None and self.lambda_bounds is not None and self.split == "train":
            print(f"Calculating lambda for target of {self.n_samples_target} samples...")
            self.lambd, n_samples, cost = self._refine_lambda_empirically(sample_tolerance=0.005, budget_tolerance=0.005, max_iters=30)
            print(f"Final refined lambda: {self.lambd:.4f} with total cost: {cost:.2f}/{self.budget:.2f} and expected samples: {n_samples}/{self.n_samples_target}")

        # generate data plan with pdf sampling
        self.data_plan, n_samples, cost = self._create_sampling_plan(lambd=self.lambd, greedy_fill=True)

        # Load Data (Subclasses will define exactly what lists they need, so we call an abstract method)
        self._init_data_lists()
        self.cost = 0
        for sample_id, fidelity_id, selected_fidelity, selected_cost in tqdm(self.data_plan, desc=f"Loading {self.split} data"):
            self._load_and_append_sample(sample_id, fidelity_id)
            self.cost += selected_cost

        if len(self.data) > 0:
            self.x_channels, self.y_channels = self.channel_splitter()
            self.normalized = False
            print(f"[{self.split}] Used {self.cost:.2f} / {self.budget:.2f}. Samples: {len(self.data)}")
        else:
            print(f"[{self.split}] WARNING: No samples loaded!")

    def _linearize_fidelities(self):
        if self.linearization_col == "n_nodes":
            # linearization for power-law convergence
            mean_nodes = self.metadata_df.groupby("meshing_fidelity")["n_nodes"].mean().sort_values()
            # theoretical error scaling: n_nodes^-q
            E_theory = mean_nodes.values ** (-self.q)
            linearized_fidelities = (E_theory[0] - E_theory) / (E_theory[0] - E_theory[-1])
            fidelity_dict = dict(zip(mean_nodes.index, linearized_fidelities))
            self.metadata_df["fidelity_linearized"] = self.metadata_df["meshing_fidelity"].map(fidelity_dict)
            self.fidelities_linearized = sorted(self.metadata_df["fidelity_linearized"].unique())
        elif self.linearization_col == "solver_max_its":
            # linearization for exponential linear convergence
            solver_max_its = np.sort(self.metadata_df["solver_max_its"].unique())
            # linear theoretical error scaling: rho^k
            E_theory = self.q ** solver_max_its
            linearized_fidelities = (E_theory[0] - E_theory) / (E_theory[0] - E_theory[-1])
            fidelity_dict = dict(zip(solver_max_its, linearized_fidelities))
            self.metadata_df["fidelity_linearized"] = self.metadata_df["solver_max_its"].map(fidelity_dict)
            self.fidelities_linearized = sorted(self.metadata_df["fidelity_linearized"].unique())
        else:
            raise ValueError(f"Unsupported linearization column: {self.linearization_col}.")


    def _refine_lambda_empirically(self, sample_tolerance=0.02, budget_tolerance=0.02, max_iters=30):
        """Bisection on lambda within lambda_bounds until the plan hits n_samples_target
        within sample_tolerance and spends the budget within budget_tolerance."""
        original_rng_state = self.composition_rng.bit_generator.state

        low_lambd = self.lambda_bounds[0]
        high_lambd = self.lambda_bounds[1]
        
        best_lambd = (low_lambd + high_lambd) / 2.0
        target_n_samples = self.n_samples_target
        
        def simulate_sampling(test_l):
            self.composition_rng.bit_generator.state = original_rng_state
            _, n_samples, cost = self._create_sampling_plan(lambd=test_l, greedy_fill=False)
            return n_samples, cost

        n_samples_low, _ = simulate_sampling(low_lambd)
        n_samples_high, _ = simulate_sampling(high_lambd)
        is_increasing = n_samples_high > n_samples_low

        current_cost = 0.0
        for i in range(max_iters):
            mid_lambd = (low_lambd + high_lambd) / 2.0
            n_samples, current_cost = simulate_sampling(mid_lambd)

            sample_error = abs(n_samples - target_n_samples) / target_n_samples
            budget_error = abs(self.budget - current_cost) / self.budget
            
            print(f"Refinement Iteration {i+1}: lambda={mid_lambd:>7.4f} | Samples: {n_samples}/{target_n_samples} | Cost: {current_cost:.2f}/{self.budget:.2f}")

            # exit condition: We hit the sample target and we are close to maxing out the budget
            if sample_error <= sample_tolerance and budget_error <= budget_tolerance:
                best_lambd = mid_lambd
                print(f"Converged on BOTH targets in {i+1} iterations.")
                break
                
            # if we hit the sample target but have lots of budget left, force it to search higher lambdas
            if sample_error <= sample_tolerance and budget_error > budget_tolerance and current_cost < self.budget:
                print(f"  -> Hit sample target, but only spent {current_cost:.2f}. Forcing search to higher fidelities.")
                low_lambd = mid_lambd  # push lambda higher to spend more money
                    
            # standard bisection logic if we haven't hit the sample target
            elif n_samples > target_n_samples:
                if is_increasing:
                    high_lambd = mid_lambd
                else:
                    low_lambd = mid_lambd
            else:
                if is_increasing:
                    low_lambd = mid_lambd
                else:
                    high_lambd = mid_lambd
                    
            best_lambd = mid_lambd

        self.composition_rng.bit_generator.state = original_rng_state
        return best_lambd, n_samples, current_cost


    def _create_sampling_plan(self, lambd, greedy_fill=True):
        plan = []
        current_cost = 0.0
        
        # calc probabilities
        probs = get_exp_pmf(self.fidelities_linearized, lambd)
        
        # cache the per-sample cost and fidelity-id lookups once (reused across the bisection iterations)
        if not hasattr(self, '_cost_lookup'):
            split_metadata = self.metadata_df[self.metadata_df['sample_id'].isin(self.sample_ids)] 
            self._min_sample_cost = split_metadata[self.cost_column].min()
            self._has_fid_col = 'fidelity_id' in split_metadata.columns
            
            self._cost_lookup = {}
            self._fid_lookup = {}
            
            for sid, group in split_metadata.groupby('sample_id'):
                self._cost_lookup[sid] = dict(zip(group['fidelity_linearized'], group[self.cost_column]))
                if self._has_fid_col:
                    self._fid_lookup[sid] = dict(zip(group['fidelity_linearized'], group['fidelity_id']))

        # draw the target fidelity of every sample of the split at once
        chosen_indices = self.composition_rng.choice(
            len(self.fidelities_linearized), 
            size=len(self.sample_ids), 
            p=probs
        )

        # greedy filling sampling loop
        for loop_idx, sample_id in enumerate(self.sample_ids):
            # if we are close to budget, stop
            if self.budget - current_cost < self._min_sample_cost:
                break

            # O(1) dict lookup
            sample_costs = self._cost_lookup.get(sample_id, {})
            if not sample_costs:
                continue

            # pre-drawn random target
            chosen_idx = chosen_indices[loop_idx]
            chosen_fidelity = self.fidelities_linearized[chosen_idx]
            cost = sample_costs.get(chosen_fidelity, float('inf'))

            # greedy logic
            selected_fidelity = None
            selected_cost = 0.0

            # case A: chosen fidelity fits budget
            if current_cost + cost <= self.budget:
                selected_fidelity = chosen_fidelity
                selected_cost = cost
            
            # case B: chosen fidelity is too expensive --> downgrade on fidelity
            elif greedy_fill:
                # find highest possible fidelity that fits (greedy fill)
                for idx in range(chosen_idx - 1, -1, -1):
                    candidate_fidelity = self.fidelities_linearized[idx]
                    candidate_cost = sample_costs.get(candidate_fidelity, float('inf'))
                    if current_cost + candidate_cost <= self.budget:
                        selected_fidelity = candidate_fidelity
                        selected_cost = candidate_cost
                        break
            
            # add to plan if something fits
            if selected_fidelity is not None:
                if self._has_fid_col:
                    fidelity_id = str(self._fid_lookup[sample_id][selected_fidelity])
                else:
                    fidelity_id = f"{selected_fidelity:.1f}"

                plan.append((sample_id, fidelity_id, selected_fidelity, selected_cost))
                current_cost += selected_cost

        return plan, len(plan), current_cost

    def _init_data_lists(self):
            self.data, self.cond, self.tri, self.metadata = [], [], [], []

    def _read_h5_with_retry(self, sample_path, reader, max_attempts=5, base_delay=1.0):
            for attempt in range(1, max_attempts + 1):
                try:
                    with h5py.File(sample_path, "r", swmr=True) as h5f:
                        return reader(h5f)
                except FileNotFoundError:
                    raise
                except OSError as e:
                    if attempt == max_attempts:
                        raise
                    delay = base_delay * 2 ** (attempt - 1)
                    print(
                        f"Transient I/O error reading {sample_path} "
                        f"(attempt {attempt}/{max_attempts}): {e}. Retrying in {delay:.0f}s...",
                        flush=True,
                    )
                    time.sleep(delay)

    def _load_and_append_sample(self, sample_id, fidelity_str):
            """Opens the HDF5 file and appends the core fields."""
            sample_path = osp.join(self.data_path, sample_id) + "/" + fidelity_str + ".h5"

            def read_fields(h5f):
                channels = {k: torch.tensor(v[:], dtype=torch.long) for k, v in h5f["channels"].items()}
                data = torch.tensor(h5f["data"]["fields"][:], dtype=self.dtype)
                tri = torch.tensor(h5f["mesh"]["connectivity"][:], dtype=torch.long)
                metadata = json.loads(h5f.attrs["metadata_json"])
                return channels, data, tri, metadata

            channels, data, tri, metadata = self._read_h5_with_retry(sample_path, read_fields)

            # append only after the whole read succeeded, so a retried attempt cannot double-append.
            self._channels = channels
            fidelity_linearized = torch.tensor([self._fidelity_lookup[(sample_id, fidelity_str)]], dtype=self.dtype)
            self.cond.append(fidelity_linearized)
            self.data.append(data)
            self.tri.append(tri)
            self.metadata.append(metadata)

    def channel_splitter(self):
        def split_channels(channel_names):
            new_channels = {}
            curr_idx = 0
            for name in channel_names:
                _slice = self._channels[name]
                new_slice = torch.arange(curr_idx, curr_idx + len(_slice), dtype=torch.long)
                new_channels[name] = new_slice
                curr_idx += len(_slice)
            return new_channels
        return split_channels(self.x_channel_names), split_channels(self.y_channel_names)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fields = self.data[idx]
        cond = self.cond[idx]
        coords = fields[:, self._channels["coords"]]
        x = torch.cat(
            [fields[:, self._channels[x_channel_name]] for x_channel_name in self.x_channel_names],
            dim=-1,
        )
        y = torch.cat(
            [fields[:, self._channels[y_channel_name]] for y_channel_name in self.y_channel_names],
            dim=-1,
        )
        # subsample if needed
        if self.n_nodes_subsampling and self.n_nodes_subsampling < coords.shape[0]:
            keep_indices = np.sort(np.random.choice(coords.shape[0], self.n_nodes_subsampling, replace=False))
            coords, x, y = coords[keep_indices, :], x[keep_indices, :], y[keep_indices, :]
        # additionals
        connectivity = self.tri[idx]
        metadata = self.metadata[idx]
        return BaseSample(coords=coords, connectivity=connectivity, x=x, y=y, cond=cond, metadata=metadata)

    @staticmethod
    def collate(batch):
        n_nodes = torch.tensor([s.coords.shape[0] for s in batch])
        coords = torch.cat([s.coords for s in batch], dim=0)
        x = torch.cat([s.x for s in batch], dim=0)
        y = torch.cat([s.y for s in batch], dim=0)
        batch_index = torch.repeat_interleave(torch.arange(n_nodes.shape[0]), n_nodes)
        n_cells = torch.tensor([s.connectivity.shape[0] for s in batch])
        connectivity = torch.cat([s.connectivity for s in batch], dim=0)
        connectivity_batch_index = torch.repeat_interleave(torch.arange(n_cells.shape[0]), n_cells)
        if batch[0].cond is not None:
            cond = torch.stack([s.cond for s in batch], dim=0)  # (B,1)
        else:
            cond = None
        return BaseSample(coords=coords, x=x, y=y, cond=cond, batch_index=batch_index, connectivity=connectivity, connectivity_batch_index=connectivity_batch_index)

    @property
    def ckpt_metadata(self):
        fidelities_linearized = [float(f) for f in self.fidelities_linearized]
        fidelity_mix = [
            (str(sample_id), str(fidelity_id), float(selected_fidelity), float(selected_cost))
            for sample_id, fidelity_id, selected_fidelity, selected_cost in self.data_plan
        ]
        return {
            "fidelities_linearized": fidelities_linearized,
            "budget_used": float(self.cost),
            "fidelity_mix": fidelity_mix,
            "lambd": None if self.lambd is None else float(self.lambd),
        }

    @property
    @abstractmethod
    def x_channel_names(self):
        pass

    @property
    @abstractmethod
    def y_channel_names(self):
        pass

    @abstractmethod
    def _compute_normalization_stats(self):
        pass

    @abstractmethod
    def normalize(self):
        pass

    @abstractmethod
    def denormalize(self):
        pass

    @abstractmethod
    def denormalize_coords(self):
        pass

    @abstractmethod
    def denormalize_x(self):
        pass
