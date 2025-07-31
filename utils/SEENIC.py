import numpy as np
import pandas as pd
import cv2
import os

def convert_SEENIC_us_to_IncEventGS_with_undistortion(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # === INPUT FILES ===
    poses_csv_path = os.path.join(input_folder, "cam-poses.csv")
    events_csv_path = os.path.join(input_folder, "events.csv")

    # === OUTPUT FILES ===
    timestamps_txt_path = os.path.join(output_folder, "poses_ts.txt")
    poses_txt_path = os.path.join(output_folder, "traj.txt")
    events_npy_path = os.path.join(output_folder, "gray_events_data.npy")

    # === ORIGINAL INTRINSICS & DISTORTION ===
    fx = 561.5818696156408
    fy = 560.6469609291944
    cx = 315.41043210173683
    cy = 248.89956648173637
    K_orig = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist = np.array([-0.40593716708733785, 0.2792463089413433,
                     0.0007263495196489163, -0.0003750949394150783,
                    -0.13889942810581364])

    # === Estimate undistorted intrinsics ===
    image_size = (640, 480)
    K_new, _ = cv2.getOptimalNewCameraMatrix(K_orig, dist, image_size, alpha=0.5)
    print("✅ Use the following undistorted intrinsics in your YAML:")
    print(f"fx: {K_new[0,0]:.6f}")
    print(f"fy: {K_new[1,1]:.6f}")
    print(f"cx: {K_new[0,2]:.6f}")
    print(f"cy: {K_new[1,2]:.6f}")

    # === 1. Load and convert poses (µs → s) ===
    poses_df = pd.read_csv(poses_csv_path, header=None, names=["t", "rx", "ry", "rz", "x", "y", "z"])
    poses_df["t"] *= 1e-6

    # === 2. Load and convert events (µs → s) ===
    events_df = pd.read_csv(events_csv_path, header=None, names=["t", "x", "y", "polarity"])
    events_df["t"] *= 1e-6

    # === 3. Align both to a common zero-start time ===
    t0_common = min(poses_df["t"].iloc[0], events_df["t"].iloc[0])
    poses_df["t"] -= t0_common
    events_df["t"] -= t0_common

    # === 4. Shift events to align with nearest pose ===
    first_event_time = events_df["t"].iloc[0]
    closest_pose_idx = poses_df["t"].sub(first_event_time).abs().idxmin()
    delta_shift = poses_df["t"].iloc[closest_pose_idx] - first_event_time
    events_df["t"] += delta_shift

    # === 4b. Save RAW (non-undistorted) event data ===
    raw_event_array = events_df[["x", "y", "t", "polarity"]].to_numpy(dtype=np.float32)
    raw_event_array[:, 3] = raw_event_array[:, 3] * 2 - 1  # polarity → {-1, +1}
    raw_npy_path = os.path.join(output_folder, "gray_events_data_raw.npy")
    np.save(raw_npy_path, raw_event_array)

    # === 5. Undistort event coordinates ===
    pts = np.stack([events_df["x"].values, events_df["y"].values], axis=-1).astype(np.float32).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(pts, K_orig, dist, P=K_new)
    events_df["x"] = undistorted[:, 0, 0]
    events_df["y"] = undistorted[:, 0, 1]

    # === 5. Undistort event coordinates ===
    pts = np.stack([events_df["x"].values, events_df["y"].values], axis=-1).astype(np.float32).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(pts, K_orig, dist, P=K_new)
    events_df["x"] = undistorted[:, 0, 0]
    events_df["y"] = undistorted[:, 0, 1]

    # === 6. Remove out-of-bounds events ===
    W, H = image_size
    valid = (
        (events_df["x"] >= 0) & (events_df["x"] < W) &
        (events_df["y"] >= 0) & (events_df["y"] < H)
    )
    events_df = events_df[valid]

    # === 7. Save poses_ts.txt ===
    poses_df["t"].to_csv(timestamps_txt_path, index=False, header=False)

    # === 8. Save traj.txt ===
    pose_matrices = []
    for _, row in poses_df.iterrows():
        rvec = np.array([row["rx"], row["ry"], row["rz"]], dtype=np.float64)
        tvec = np.array([row["x"], row["y"], row["z"]], dtype=np.float64)
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec
        pose_matrices.append(T.flatten())
    np.savetxt(poses_txt_path, pose_matrices)

    # === 9. Save gray_events_data.npy ===
    event_array = events_df[["x", "y", "t", "polarity"]].to_numpy(dtype=np.float32)
    event_array[:, 3] = event_array[:, 3] * 2 - 1  # polarity → {-1, +1}
    np.save(events_npy_path, event_array)

    # === 10. Report ===
    print("✅ Conversion complete (with undistortion)")
    print(f"- Poses saved: {len(poses_df)}")
    print(f"- Events saved: {len(event_array)}")
    print(f"- Pose timestamps:   {poses_df['t'].min():.6f}s → {poses_df['t'].max():.6f}s")
    print(f"- Event timestamps:  {event_array[:, 2].min():.6f}s → {event_array[:, 2].max():.6f}s")
    print(f"- Shift applied to events: {delta_shift:.6f}s")

# Example usage
input_folder = r"Z:\Datasets\inputs\SEENIC\test"
output_folder = r"Z:\Datasets\inputs\SEENIC\test"
convert_SEENIC_us_to_IncEventGS_with_undistortion(input_folder, output_folder)
