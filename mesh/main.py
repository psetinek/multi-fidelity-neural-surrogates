import os
import traceback
import sys
from datetime import datetime
import petname

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.multiprocessing as mp

from mf_surrogates.run import run
from mf_surrogates.utils import set_seed, find_free_port


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig) -> None:
    # set cublas workspace and seed libraries for reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    set_seed(cfg.seed)

    print("#" * 88, "\nStarting with configs:")
    print(OmegaConf.to_yaml(cfg))
    print("#" * 88, "\n")

    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size = 1

    try:
        if cfg.logging.run_id is None:
            date_and_time = datetime.today().strftime("%Y%m%d_%H%M%S")
            random_petname = petname.generate(2, separator="_")
            cfg.logging.run_id = f"{random_petname}_{date_and_time}"
        os.makedirs(os.path.join(cfg.output_path, cfg.logging.run_id), exist_ok=True)

        if cfg.use_ddp and world_size > 1:
            if "SLURM_NODELIST" not in os.environ:
                os.environ["MASTER_ADDR"] = "localhost"
            else:
                # only works for single node so far, adapt above for multinode
                os.environ["MASTER_ADDR"] = os.environ["SLURM_NODELIST"]
            os.environ["MASTER_PORT"] = str(find_free_port())
            if "NCCL_SOCKET_IFNAME" in os.environ:
                # unset nccl comm interface
                del os.environ["NCCL_SOCKET_IFNAME"]
            mp.spawn(run, args=(cfg, world_size), nprocs=world_size)
        else:
            rank = 0
            run(rank, cfg, world_size=1)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
