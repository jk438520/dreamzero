import os
import shutil
import json

def create_subset(source_dir, target_dir, size):
    print(f"Creating subset in {target_dir}...")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    
    os.makedirs(target_dir)
    os.makedirs(os.path.join(target_dir, 'meta'), exist_ok=True)
    
    # 1. Copy and filter meta/episodes.jsonl
    episodes_file = os.path.join(source_dir, 'meta', 'episodes.jsonl')
    if not os.path.exists(episodes_file):
        print(f"Error: {episodes_file} not found. Ensure this is a LeRobot dataset.")
        return
    
    total_frames = 0
    with open(episodes_file, 'r') as f_in, open(os.path.join(target_dir, 'meta', 'episodes.jsonl'), 'w') as f_out:
        for i, line in enumerate(f_in):
            if i >= size:
                break
            f_out.write(line)
            data = json.loads(line)
            total_frames += data.get('length', 0)
            
    # 2. Process and copy other meta files
    for meta_file in os.listdir(os.path.join(source_dir, 'meta')):
        if meta_file == 'episodes.jsonl':
            continue
            
        src_path = os.path.join(source_dir, 'meta', meta_file)
        dst_path = os.path.join(target_dir, 'meta', meta_file)
        
        if meta_file == 'info.json':
            with open(src_path, 'r') as f:
                info = json.load(f)
            info['total_episodes'] = size
            info['total_frames'] = total_frames
            info['total_chunks'] = 1
            info['total_videos'] = info.get('total_videos', 0) # Not changed usually
            with open(dst_path, 'w') as f:
                json.dump(info, f, indent=4)
        else:
            shutil.copy(src_path, dst_path)

    # 3. Copy chunk data correctly
    # we just need up to chunk-XXX where episodes are
    # For size < 1000, they are all in chunk-000
    os.makedirs(os.path.join(target_dir, 'data', 'chunk-000'), exist_ok=True)
    for i in range(size):
        ep_file = f"episode_{i:06d}.parquet"
        src_ep = os.path.join(source_dir, 'data', 'chunk-000', ep_file)
        if os.path.exists(src_ep):
            shutil.copy(src_ep, os.path.join(target_dir, 'data', 'chunk-000', ep_file))

    # 4. Copy required videos correctly
    src_vid_dir = os.path.join(source_dir, 'videos', 'chunk-000')
    if os.path.exists(src_vid_dir):
        for cam_dir in os.listdir(src_vid_dir):
            src_cam = os.path.join(src_vid_dir, cam_dir)
            if not os.path.isdir(src_cam):
                continue
            
            dst_cam = os.path.join(target_dir, 'videos', 'chunk-000', cam_dir)
            os.makedirs(dst_cam, exist_ok=True)
            for i in range(size):
                vid_file = f"episode_{i:06d}.mp4"
                src_vid = os.path.join(src_cam, vid_file)
                if os.path.exists(src_vid):
                    shutil.copy(src_vid, os.path.join(dst_cam, vid_file))

    print(f"Saved subset to {target_dir}")
    print("-" * 50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/droid_lerobot")
    args = parser.parse_args()
    
    source = args.source
    
    # 10 trajectories
    create_subset(source, source + "_10_traj", 10)
    
    # 100 trajectories
    create_subset(source, source + "_100_traj", 100)
