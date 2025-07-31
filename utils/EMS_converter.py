import os
import numpy as np
from glob import glob
from tqdm import tqdm

def merge_event_csvs_to_npy(folder_path, output_path):
    csv_files = sorted(
        glob(os.path.join(folder_path, "*.csv")),
        key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
    )

    all_events = []
    print(f"📂 Found {len(csv_files)} CSV files in: {folder_path}")

    for f in tqdm(csv_files, desc="🔄 Merging CSVs"):
        try:
            data = np.loadtxt(f, delimiter=",", skiprows=1)
            if data.ndim == 1:
                data = data[None, :]  # Ensure 2D
            t = data[:, 0] / 100000.0  # Convert ms → s
            x = data[:, 1]
            y = data[:, 2]
            p = (data[:, 3] * 2) - 1  # Normalize: 0 → -1, 1 → +1
            events = np.stack([x, y, t, p], axis=1)
            all_events.append(events)
        except Exception as e:
            print(f"⚠️ Skipping {f} due to error: {e}")

    all_events = np.concatenate(all_events, axis=0)
    print(f"💾 Saving {all_events.shape[0]} events to {output_path}")
    np.save(output_path, all_events)

# === Usage ===
if __name__ == "__main__":
    input_folder = r"Z:\Datasets\E-M-S\training\Event\event_stream\479"
    output_file = r"Z:\Datasets\inputs\synthetic\EMS_479\event_threshold_0.1\gray_events_data.npy"
    merge_event_csvs_to_npy(input_folder, output_file)
