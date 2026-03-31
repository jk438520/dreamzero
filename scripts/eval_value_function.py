import torch._dynamo
torch._dynamo.config.disable = True

import argparse
import os
import glob
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import pyarrow.parquet as pq
from collections import deque

from tianshou.data import Batch
from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy


def save_value_plots(all_preds, all_gts, output_dir, episode_idx):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(all_gts, label="GT Task Progress", alpha=0.8, linewidth=2)
    plt.plot(all_preds, label="Predicted Task Progress", alpha=0.8, linestyle="--", linewidth=2)
    plt.title(f"Episode {episode_idx} Task Progress")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"value_episode_{episode_idx}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    mse = np.mean((np.array(all_preds) - np.array(all_gts)) ** 2)
    print(f"Saved episode {episode_idx} plot to {plot_path}. MSE: {mse:.4f}")


def evaluate(args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl", rank=0, world_size=1)

    print(f"Loading model from {args.model_path} ...")
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.OXE_DROID,
        model_path=args.model_path,
        device=args.device,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    parquet_files = sorted(glob.glob(os.path.join(args.dataset_path, "data", "**", "episode_*.parquet"), recursive=True))
    videos_root = os.path.join(args.dataset_path, "videos")

    num_eval_eps = args.num_episodes if args.num_episodes > 0 else len(parquet_files)
    for ep_idx in range(min(num_eval_eps, len(parquet_files))):
        print(f"\n--- Evaluating Episode {ep_idx} ---")
        
        table = pq.read_table(parquet_files[ep_idx])
        num_rows = table.num_rows
        ep_name = os.path.basename(parquet_files[ep_idx]).replace('.parquet', '')
        
        mp4_files = glob.glob(os.path.join(videos_root, "**", f"{ep_name}.mp4"), recursive=True)
        caps = {
            os.path.basename(os.path.dirname(p)).replace("observation.images.", "video."): cv2.VideoCapture(p)
            for p in mp4_files
        }
        
        gt_progress_col = np.linspace(0, 1, num_rows)
        if "observation.state" in table.column_names:
            state_col = table.column("observation.state").to_pylist()
        else:
            state_col = None
        
        ep_preds, ep_gts = [], []
        policy.trained_model.action_head.reset()
        
        state_horizon = policy._state_horizon if policy._state_horizon is not None else 1
        frames_per_chunk = 4
        
        frame_buffers = {cam: [] for cam in caps.keys()}
        state_history = deque(maxlen=state_horizon)
        
        for step_idx in range(num_rows):
            obs_dict = {}
            for cam_name, cap in caps.items():
                _, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_buffers[cam_name].append(frame)
                
                num_frames = 1 if step_idx == 0 else frames_per_chunk
                frames_to_use = frame_buffers[cam_name][-num_frames:]
                
                while len(frames_to_use) < num_frames:
                    frames_to_use.insert(0, frame_buffers[cam_name][0])
                    
                obs_dict[cam_name] = np.expand_dims(np.stack(frames_to_use, axis=0), 0).astype(np.uint8)
                
            if state_col is not None:
                state_history.append(state_col[step_idx])
                while step_idx == 0 and len(state_history) < state_horizon:
                    state_history.append(state_col[step_idx])
                    
                obs_dict["state"] = np.expand_dims(np.stack(state_history, axis=0), 0).astype(np.float32)
            
            batch = policy.apply(Batch(obs=obs_dict))
            
            with torch.inference_mode():
                b_in, a_in = policy.trained_model.prepare_input(batch.normalized_obs)
                b_out = policy.trained_model.backbone(b_in)
                outputs = policy.trained_model.action_head.lazy_joint_video_action(b_out, a_in, latent_video=None)
                
            v_pred = outputs.data["value_pred"].cpu().float().numpy()
            ep_preds.append(float(v_pred[0, 0, 0] if len(v_pred.shape) == 3 else v_pred[0, 0]))
            ep_gts.append(float(gt_progress_col[step_idx]))
                
        for cap in caps.values():
            cap.release()
            
        save_value_plots(ep_preds, ep_gts, args.output_dir, ep_idx)

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_path", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_episodes", type=int, default=-1, help="Number of episodes. -1 for all.")
    p.add_argument("--output_dir", default="results_value_eval")
    evaluate(p.parse_args())

if __name__ == "__main__":
    main()
