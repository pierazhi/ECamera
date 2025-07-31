import os
import numpy as np
from tqdm import tqdm

def convert_ems_pose_file_to_traj(pose_file, output_path):
    all_poses = []

    with open(pose_file, 'r') as f:
        for line in tqdm(f, desc=f"🔁 Converting {os.path.basename(pose_file)}"):
            vals = list(map(float, line.strip().split()))
            if len(vals) != 12:
                print(f"⚠️ Skipping malformed line: {line.strip()}")
                continue
            mat_4x3 = np.array(vals).reshape(4, 3)
            R = mat_4x3[:3, :]
            t = mat_4x3[3, :].reshape(3, 1)
            mat_4x4 = np.vstack([np.hstack([R, t]), [0, 0, 0, 1]])
            all_poses.append(mat_4x4.flatten())

    all_poses = np.array(all_poses)
    np.savetxt(output_path, all_poses, fmt="%.6f")
    print(f"✅ Saved {len(all_poses)} poses to {output_path}")

# === Run it ===
if __name__ == "__main__":
    pose_file = r"Z:\Datasets\E-M-S\training\Event\pose\479.txt"
    output_file = r"Z:\Datasets\inputs\synthetic\EMS_479\traj.txt"
    convert_ems_pose_file_to_traj(pose_file, output_file)
