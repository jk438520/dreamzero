from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
import torch 
import os

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group("gloo", rank=0, world_size=1)

policy = GrootSimPolicy(
    embodiment_tag=EmbodimentTag.OXE_DROID,
    model_path="checkpoints/dreamzero_droid_vfh_finetune_10_traj",
    device="cpu",
)