import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.io import savemat



def convert_events_to_npy(npz_path: str, output_dir: str, filename: str = 'gray_events_data.npy'):
    """
    Convert ELOPE-style event data from (x, y, polarity, t in µs) to (x, y, t in s, polarity ∈ {-1, 1}).
    Saves as gray_events_data.npy and previews first 5 rows.
    """
    os.makedirs(output_dir, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)
    events = data['events']  # dtype: object, shape: (N,)

    # Unpack tuples into arrays
    x = np.array([e[0] for e in events], dtype=np.int16)
    y = np.array([e[1] for e in events], dtype=np.int16)
    p = np.array([1 if e[2] else -1 for e in events], dtype=np.int8)  # True → 1, False → -1
    t = np.array([e[3] for e in events], dtype=np.float32) * 1e-6     # µs → s

    # Combine into [x, y, t, p]
    events_converted = np.stack([x, y, t, p], axis=1).astype(np.float32)

    # Save
    out_path = os.path.join(output_dir, filename)
    np.save(out_path, events_converted)

    # Print first 5
    print("\nFirst 5 converted events [x, y, t (s), p]:")
    print(events_converted[:5])

    # Plot first 5
    plt.figure(figsize=(5, 5))
    colors = ['red' if pol > 0 else 'blue' for pol in events_converted[:5, 3]]
    for i in range(5):
        plt.scatter(events_converted[i, 0], events_converted[i, 1],
                    color=colors[i], label=f'Event {i+1} ({int(events_converted[i,3])})')
    plt.gca().invert_yaxis()
    plt.xlabel("x"), plt.ylabel("y")
    plt.title("First 5 Events (Red=ON, Blue=OFF)")
    plt.legend(), plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"\n✅ Saved converted events to: {out_path}")
    return events_converted

def convert_traj_to_txt(npz_path: str, output_dir: str, filename: str = 'traj.txt'):
    """
    Converts ELOPE trajectory data (position + Euler angles) to SE(3) poses and saves as traj.txt.

    Parameters:
        npz_path (str): Path to .npz file containing 'traj'.
        output_dir (str): Directory to save traj.txt in.
        filename (str): Output filename (default: 'traj.txt').

    Output:
        traj.txt: N x 16 matrix, each row is a 4x4 transformation matrix flattened row-wise.
    """
    os.makedirs(output_dir, exist_ok=True)
    data = np.load(npz_path, allow_pickle=True)

    traj = data['traj']  # shape (N, 12)
    positions = traj[:, :3]
    euler_angles = traj[:, 6:9]  # phi, theta, psi in radians

    poses = []
    for pos, angles in zip(positions, euler_angles):
        rot_mat = R.from_euler('xyz', angles).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot_mat
        T[:3, 3] = pos
        poses.append(T.reshape(-1))  # flatten 4x4 → (16,)

    poses = np.stack(poses)
    out_path = os.path.join(output_dir, filename)
    np.savetxt(out_path, poses, fmt='%.6f')

    print(f"✅ Saved {poses.shape[0]} SE(3) poses to: {out_path}")
    return poses

def save_pose_timestamps(npz_path: str, output_dir: str, filename: str = 'poses_ts.txt'):
    """
    Extracts pose timestamps from ELOPE .npz file and saves them as poses_ts.txt.

    Parameters:
        npz_path (str): Path to the .npz file containing 'timestamps'.
        output_dir (str): Directory to save poses_ts.txt.
        filename (str): Output filename (default: 'poses_ts.txt').

    Output:
        poses_ts.txt: Plain text file with one timestamp per line (float, in seconds).
    """
    os.makedirs(output_dir, exist_ok=True)
    data = np.load(npz_path, allow_pickle=True)

    timestamps = data['timestamps']  # shape (N,)
    out_path = os.path.join(output_dir, filename)

    np.savetxt(out_path, timestamps, fmt='%.6f')

    print(f"✅ Saved {len(timestamps)} pose timestamps to: {out_path}")
    return timestamps

def extract_elope_to_mat(npz_path, output_dir):
    """
    Extracts ground-truth poses and velocities from ELOPE .npz and saves to .mat.
    
    Output .mat file contains:
        - poses: (N, 4, 4) SE(3) camera poses
        - positions: (N, 3)
        - velocities: (N-1, 3)
        - timestamps: (N,)
    """
    os.makedirs(output_dir, exist_ok=True)
    data = np.load(npz_path, allow_pickle=True)

    traj = data['traj']          # (N, 12)
    timestamps = data['timestamps']  # (N,)
    positions = traj[:, :3]      # (N, 3)
    angles = traj[:, 6:9]        # (N, 3): [phi, theta, psi]

    N = traj.shape[0]
    poses = np.zeros((N, 4, 4), dtype=np.float32)

    for i in range(N):
        T = np.eye(4)
        R_mat = R.from_euler('xyz', angles[i]).as_matrix()
        T[:3, :3] = R_mat
        T[:3, 3] = positions[i]
        poses[i] = T

    # Estimate linear velocity
    delta_t = np.diff(timestamps).reshape(-1, 1)
    velocities = np.diff(positions, axis=0) / delta_t  # (N-1, 3)

    # Save to .mat
    out_file = os.path.join(output_dir, os.path.basename(npz_path).replace('.npz', '_groundtruth.mat'))
    savemat(out_file, {
        'poses': poses,               # SE(3) matrices (N, 4, 4)
        'positions': positions,       # (N, 3)
        'velocities': velocities,     # (N-1, 3)
        'timestamps': timestamps      # (N,)
    })

    print(f"✅ Saved ground truth to: {out_file}")

input = r"Z:\Datasets\elope_dataset\train\0027.npz"
output = r"Z:\Datasets\inputs\ELOPE\0027"
output_matlab = r"C:\Users\Pierazhi\Documents\MATLAB\Tesi\ELOPE"
convert_events_to_npy(input, output)
convert_traj_to_txt(input, output)
save_pose_timestamps(input, output)
extract_elope_to_mat(input, output_matlab)