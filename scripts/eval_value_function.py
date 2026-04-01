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


def _build_frame_progress(table):
    """Build per-row task progress consistent with training: frame_index / episode_length."""
    num_rows = table.num_rows
    if "frame_index" in table.column_names:
        frame_index = np.asarray(table.column("frame_index").to_pylist(), dtype=np.float32)
    else:
        frame_index = np.arange(num_rows, dtype=np.float32)
    max_length = max(num_rows, 1)
    return frame_index / float(max_length)


def _future_progress_targets(progress, step_idx, action_horizon):
    """Construct per-action-step future progress targets with end padding."""
    idx = np.arange(step_idx, step_idx + action_horizon)
    idx = np.clip(idx, 0, len(progress) - 1)
    return progress[idx].reshape(-1, 1).astype(np.float32)


def _project_to_value_tokens(raw_progress, num_value_tokens):
    """Project action-horizon progress targets to value-token targets like training code."""
    raw_t = torch.from_numpy(raw_progress.T).unsqueeze(0)  # [1, 1, action_horizon]
    if num_value_tokens <= 1:
        # Mirror interpolation branch behavior for single value token.
        token_t = torch.nn.functional.interpolate(
            raw_t, size=1, mode="linear", align_corners=True
        )
    else:
        token_t = torch.nn.functional.interpolate(
            raw_t, size=num_value_tokens, mode="linear", align_corners=True
        )
    return token_t.squeeze(0).T.numpy().astype(np.float32)  # [num_value_tokens, 1]


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
        
        gt_progress_col = _build_frame_progress(table)
        if "observation.state" in table.column_names:
            state_col = table.column("observation.state").to_pylist()
        else:
            state_col = None
        if "annotation.language.language_instruction" in table.column_names:
            language_col = table.column("annotation.language.language_instruction").to_pylist()
        else:
            language_col = None
        
        ep_preds, ep_gts = [], []
        # Reset AR state at episode start (mimics socket_test behavior)
        policy.trained_model.action_head.reset()
        
        state_horizon = policy._state_horizon if policy._state_horizon is not None else 1
        frames_per_chunk = 4
        
        frame_buffers = {cam: [] for cam in caps.keys()}
        state_history = deque(maxlen=state_horizon)
        call_count = 0
        
        for step_idx in range(num_rows):
            obs_dict = {}
            for cam_name, cap in caps.items():
                _, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_buffers[cam_name].append(frame)
                
                # Frame accumulation pattern matching socket_test:
                # First call: 1 frame, subsequent: 4 frames per chunk
                num_frames = 1 if call_count == 0 else frames_per_chunk
                frames_to_use = frame_buffers[cam_name][-num_frames:]
                
                while len(frames_to_use) < num_frames:
                    frames_to_use.insert(0, frame_buffers[cam_name][0])
                    
                obs_dict[cam_name] = np.expand_dims(np.stack(frames_to_use, axis=0), 0).astype(np.uint8)
                
            if state_col is not None:
                state_history.append(state_col[step_idx])
                while step_idx == 0 and len(state_history) < state_horizon:
                    state_history.append(state_col[step_idx])

                state_stack = np.stack(state_history, axis=0).astype(np.float32)
                # Match training schema: split state keys before transform concat.
                if state_stack.shape[-1] >= 8:
                    joint = state_stack[:, :7]
                    gripper = state_stack[:, 7:8]
                else:
                    joint = np.zeros((state_stack.shape[0], 7), dtype=np.float32)
                    gripper = np.zeros((state_stack.shape[0], 1), dtype=np.float32)
                    joint[:, : min(7, state_stack.shape[-1])] = state_stack[:, : min(7, state_stack.shape[-1])]
                obs_dict["state.joint_position"] = np.expand_dims(joint, 0)
                obs_dict["state.gripper_position"] = np.expand_dims(gripper, 0)

            # Match training language key for OXE DROID transforms.
            if language_col is not None and len(language_col) > step_idx:
                obs_dict["annotation.language.language_instruction"] = str(language_col[step_idx])
            else:
                obs_dict["annotation.language.language_instruction"] = ""
            
            # Use the same inference path as deployment and read propagated value_pred.
            with torch.inference_mode():
                result_batch, _ = policy.lazy_joint_forward_causal(Batch(obs=obs_dict))

            if hasattr(result_batch, "value_pred"):
                v_pred = result_batch.value_pred.cpu().float().numpy()
            else:
                print("Warning: result_batch.value_pred is missing; falling back to 0.")
                v_pred = np.zeros((1, 1), dtype=np.float32)

            if v_pred.ndim == 1:
                v_pred = v_pred.reshape(-1, 1)

            num_value_tokens = int(v_pred.shape[0])
            action_horizon = int(getattr(policy.trained_model.action_head, "action_horizon", 24))
            raw_progress = _future_progress_targets(gt_progress_col, step_idx, action_horizon)
            gt_tokens = _project_to_value_tokens(raw_progress, num_value_tokens)

            # Sequence-level metric: compare all predicted value tokens against aligned targets.
            token_mse = float(np.mean((v_pred[:, :1] - gt_tokens[:, :1]) ** 2))

            # Keep one scalar per step for plotting (mean over value tokens).
            ep_preds.append(float(np.mean(v_pred[:, 0])))
            ep_gts.append(float(np.mean(gt_tokens[:, 0])))
            print(f"step={step_idx:04d} token_mse={token_mse:.6f} pred_shape={v_pred.shape}")
            call_count += 1
                
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
