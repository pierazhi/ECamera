import numpy as np
import argparse
import cv2
import os
import matplotlib.pyplot as plt

def infer_resolution(events):
    x_max = int(events[:, 0].max()) + 1
    y_max = int(events[:, 1].max()) + 1
    return (int(y_max), int(x_max))  # H, W

def render_events(events, H, W, mode="accumulate"):
    if mode == "color":
        image = np.zeros((H, W, 3), dtype=np.uint8)
        for x, y, _, p in events:
            ix, iy = int(x), int(y)
            if p > 0:
                image[iy, ix, 0] = 255
            elif p < 0:
                image[iy, ix, 2] = 255
        return image

    image = np.zeros((H, W), dtype=np.float32)

    if mode == "accumulate":
        for x, y, *_ in events:
            image[int(y), int(x)] += 1
        image = image / (image.max() + 1e-9) * 255

    elif mode == "polarity":
        temp = np.zeros((H, W), dtype=np.int32)
        for x, y, _, p in events:
            ix, iy = int(x), int(y)
            temp[iy, ix] = 1 if p > 0 else -1 if p < 0 else 0
        image = ((temp + 1) / 2.0 * 255).astype(np.uint8)
        return image

    elif mode == "time_surface":
        for x, y, t, _ in events:
            image[int(y), int(x)] = max(image[int(y), int(x)], t)
        t_min = image[image > 0].min() if (image > 0).any() else 0
        t_max = image.max()
        image = (image - t_min) / (t_max - t_min + 1e-9) * 255

    elif mode == "grayscale":
        for x, y, _, p in events:
            image[int(y), int(x)] += p
        image = np.clip((image - image.min()) / (image.max() - image.min() + 1e-9) * 255, 0, 255)

    return image.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Real-time Event Stream Viewer")
    parser.add_argument("file", type=str, help="Path to .npy file with events")
    parser.add_argument("--start_idx", type=int)
    parser.add_argument("--end_idx", type=int)
    parser.add_argument("--play_mode", type=str, choices=["range", "all"], default="range")
    parser.add_argument("--mode", type=str,
                        choices=["accumulate", "polarity", "time_surface", "grayscale", "color"],
                        default="accumulate")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--histogram", action="store_true")
    parser.add_argument("--save_video", type=str, default=None, help="Path to output video (e.g., out.avi or out.mp4)")
    parser.add_argument("--simple", action="store_true", help="Play events without requiring poses_ts.txt")
    parser.add_argument("--interval_analysis", action="store_true", help="Play events without requiring poses_ts.txt")
    parser.add_argument("--compare", action="store_true", help="Compare undistorted vs raw events")
    parser.add_argument("--raw_file", type=str, default=None, help="Optional path to raw events .npy file")

    args = parser.parse_args()

    events = np.load(args.file)
    events_raw = None
    if args.compare:
        if args.raw_file is None:
            raise ValueError("You must specify --raw_file when using --compare")
        if not os.path.exists(args.raw_file):
            raise FileNotFoundError(f"Raw events file not found: {args.raw_file}")
        events_raw = np.load(args.raw_file)

    if args.histogram:
        timestamps = events[:, 2]
        parent_dir = os.path.dirname(os.path.dirname(args.file))
        poses_ts_path = os.path.join(parent_dir, "poses_ts.txt")

        if os.path.exists(poses_ts_path):
            poses_ts = np.loadtxt(poses_ts_path)
            num_bins = len(poses_ts)

            # Make bin edges: one between each pose, last bin edge goes slightly beyond
            if len(poses_ts) >= 2:
                delta_last = poses_ts[-1] - poses_ts[-2]
            else:
                delta_last = 1e-3
            bin_edges = np.append(poses_ts, poses_ts[-1] + delta_last)

            # Histogram with bin edges = pose timestamps
            counts, _ = np.histogram(timestamps, bins=bin_edges)

            min_idx = int(np.argmin(counts))
            max_idx = int(np.argmax(counts))

            print(f"📉 Lowest count: {counts[min_idx]} at pose index {min_idx} (~{poses_ts[min_idx]:.6f}s)")
            print(f"📈 Highest count: {counts[max_idx]} at pose index {max_idx} (~{poses_ts[max_idx]:.6f}s)")

            # Plot
            plt.figure(figsize=(12, 4))
            plt.bar(range(num_bins), counts, align='center')
            plt.axvspan(min_idx - 0.5, min_idx + 0.5, color='red', alpha=0.3, label='Min Events Bin')
            plt.axvspan(max_idx - 0.5, max_idx + 0.5, color='green', alpha=0.3, label='Max Events Bin')
            plt.xlabel("Pose Index")
            plt.ylabel("Number of Events")
            plt.title("Event Count Per Pose")
            plt.xticks(np.linspace(0, num_bins - 1, min(num_bins, 10), dtype=int))
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            print("⚠️ poses_ts.txt not found, cannot do pose-based binning.")


    if events.shape[1] != 4:
        raise ValueError("Expected shape (N, 4): x, y, t, polarity")

    if args.simple:
        t_start = events[:, 2].min()
        t_end = events[:, 2].max()
        print(f"🎞️ Simple mode: playing full event range from {t_start:.6f}s to {t_end:.6f}s")
    else:
        parent_dir = os.path.dirname(os.path.dirname(args.file))
        poses_ts_path = os.path.join(parent_dir, "poses_ts.txt")
        if not os.path.exists(poses_ts_path):
            raise FileNotFoundError(f"poses_ts.txt not found in {parent_dir}")
        pose_ts = np.loadtxt(poses_ts_path)

        if args.play_mode == "range":
            if args.start_idx is None or args.end_idx is None:
                raise ValueError("Must provide --start_idx and --end_idx")
            t_start = pose_ts[args.start_idx]
            t_end = pose_ts[args.end_idx]
            print(f"🎞️ Playing range from pose {args.start_idx} to {args.end_idx}")
        else:
            t_start = pose_ts[0]
            t_end = pose_ts[-1]
            print(f"🎞️ Playing full timestamp range {t_start:.6f}s to {t_end:.6f}s")

    H, W = infer_resolution(events)

    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID') if args.save_video.endswith('.avi') else cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.save_video, fourcc, int(1 / args.dt), (W, H))
        print(f"💾 Saving video to {args.save_video}")

    t = t_start
    dt = args.dt
    frame_id = 0
    paused = False
    window_title = f"Event Stream - {args.mode}"

    while t < t_end:
        render_frame = True
        key = cv2.waitKey(1)

        if key == 27:  # ESC
            print("🛑 Exiting.")
            break
        elif key == 32:  # SPACE
            paused = not paused
            print("⏸️ Paused" if paused else "▶️ Resumed")
        elif key in [45, 95]:  # '-' or '_'
            dt = max(0.001, dt * 0.8)
            print(f"🐢 Slower: dt = {dt:.4f}s")
        elif key in [43, 61]:  # '+' or '='
            dt = min(0.5, dt * 1.25)
            print(f"⚡ Faster: dt = {dt:.4f}s")
        elif key in [ord('a'), ord('A')]:
            t = max(t_start, t - dt)
            frame_id = max(0, frame_id - 1)
            print(f"⬅️ Back: t = {t:.4f}s")
        elif key in [ord('d'), ord('D')]:
            t = min(t_end, t + dt)
            frame_id += 1
            print(f"➡️ Forward: t = {t:.4f}s")
        elif key != -1:
            render_frame = False

        if render_frame and (not paused or key in [ord('a'), ord('A'), ord('d'), ord('D')]):
            # Time window
            mask = (events[:, 2] >= t) & (events[:, 2] < t + dt)
            ev_chunk = events[mask]

            # Prepare overlay text
            font_scale = H / 750
            font_thickness = max(1, int(H / 500))
            text = f"t = {t:.2f}s  dt = {dt:.3f}s  events = {ev_chunk.shape[0]}  frame = {frame_id}"

            # Undistorted view
            img = render_events(ev_chunk, H, W, args.mode)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.putText(img, text, (5, int(25 * font_scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            cv2.imshow(window_title + " (Undistorted)", img)

            # Raw view (if enabled)
            if args.compare:
                mask_raw = (events_raw[:, 2] >= t) & (events_raw[:, 2] < t + dt)
                ev_chunk_raw = events_raw[mask_raw]
                img_raw = render_events(ev_chunk_raw, H, W, args.mode)
                if len(img_raw.shape) == 2:
                    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
                cv2.putText(img_raw, text, (5, int(25 * font_scale)), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                cv2.imshow(window_title + " (Raw)", img_raw)


            if video_writer is not None:
                video_writer.write(img)

            if not paused and key not in [ord('a'), ord('A'), ord('d'), ord('D')]:
                t = min(t + dt, t_end)
                frame_id += 1

    if video_writer is not None:
        video_writer.release()
        print(f"✅ Video saved to {args.save_video}")
        
    if args.interval_analysis:
        timestamps = np.sort(events[:, 2])
        unique_ts = np.unique(timestamps)
        intervals = np.diff(unique_ts)

        print("📊 Timestamp Interval Analysis")
        print(f"Total unique timestamps: {len(unique_ts)}")
        print(f"Average interval: {np.mean(intervals):.6f} s")
        print(f"Min interval:     {np.min(intervals):.6f} s")
        print(f"Max interval:     {np.max(intervals):.6f} s")
        print(f"Std deviation:    {np.std(intervals):.6f} s")

        plt.figure(figsize=(10, 4))
        plt.hist(intervals, bins=50, edgecolor='black')
        plt.title("Histogram of Time Intervals Between Unique Timestamps")
        plt.xlabel("Interval (s)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
