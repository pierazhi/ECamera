import subprocess
import yaml
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
from itertools import product
import torch
import pyiqa

# === CONFIGURATION ===

# sweep_param_1 = "pose_lr"
# sweep_values_1 = [1e-03, 1e-04, 1.0e-05]
 
# sweep_param_2 = "mapping.bounding_size"
# sweep_values_2 = [1, 5, 10]

# sweep_param_1 = "initialization.max_n_winsize"
# sweep_values_1 = [80000, 120000, 160000, 200000]

# sweep_param_2 = "initialization.min_n_winsize"
# sweep_values_2 = [40000, 80000, 120000, 160000]

# sweep_param_1 = "initialization.max_n_winsize"
# sweep_values_1 = [10000, 30000, 50000, 70000]

# sweep_param_2 = "initialization.min_n_winsize"
# sweep_values_2 = [5000, 10000, 15000, 20000]

sweep_param_1 = "initialization.training_batch_size"
sweep_values_1 = [1, 2, 3, 4, 5]

sweep_param_2 = None
sweep_values_2 = [5000, 10000, 15000, 20000]


base_config_path = r"C:\Users\Pierazhi\Desktop\IncEventGS\configs\SimuEvent\replica_seenic_approach_slow_ambient.yaml"
image_every = 100
font_size = 20
temp_yaml_path = "temp_sweep_config.yaml"
device = 'cuda'

# === UTILS ===
def set_nested(cfg, path, val):
    keys = path.split('.')
    for k in keys[:-1]:
        cfg = cfg.setdefault(k, {})
    cfg[keys[-1]] = val

def get_nested(cfg, path):
    keys = path.split('.')
    for k in keys:
        cfg = cfg[k]
    return cfg

def load_img(path):
    return Image.open(path).convert("RGB") if os.path.exists(path) else Image.new("RGB", (1, 1), (80, 80, 80))

def pad_to_size(img, size, fill=(255,255,255)):
    return ImageOps.pad(img, size, color=fill)

def create_mosaic(grid, labels, col_titles, filename, scores=None):
    if not grid: return
    rows, cols = len(grid), len(grid[0])
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    col_widths = [max(grid[r][c].width for r in range(rows)) for c in range(cols)]
    row_heights = [max(grid[r][c].height for c in range(cols)) for r in range(rows)]
    label_width = max(font.getbbox(label)[2] for label in labels) + 40
    header_height = max(font.getbbox(t)[3] - font.getbbox(t)[1] for t in col_titles) + 20

    total_width = sum(col_widths) + label_width
    total_height = sum(row_heights) + header_height

    mosaic = Image.new("RGB", (total_width, total_height), (255,255,255))
    draw = ImageDraw.Draw(mosaic)

    x_offset = label_width
    for i, title in enumerate(col_titles):
        tw = font.getbbox(title)[2]
        tx = x_offset + (col_widths[i] - tw)//2
        draw.text((tx, 5), title, fill="red", font=font)
        x_offset += col_widths[i]

    y_offset = header_height
    for r in range(rows):
        row_h = row_heights[r]
        x_offset = label_width
        for c in range(cols):
            img = pad_to_size(grid[r][c], (col_widths[c], row_h))
            mosaic.paste(img, (x_offset, y_offset))
            x_offset += col_widths[c]

        label = labels[r]
        score_text = f" | {scores[r]}" if scores else ""
        draw.text((10, y_offset + 10), label + score_text, fill="red", font=font)
        y_offset += row_h

    mosaic.save(filename)
    print(f"✅ Saved: {filename}")

def evaluate_images(gt_folder, est_folder):
    gt_paths = sorted([os.path.join(gt_folder, f) for f in os.listdir(gt_folder) if f.endswith(('.png', '.jpg'))])
    est_paths = sorted([os.path.join(est_folder, f) for f in os.listdir(est_folder) if f.endswith(('.png', '.jpg'))])
    assert len(est_paths) <= len(gt_paths), "More estimated images than ground truth!"
    gt_paths = gt_paths[:len(est_paths)]

    lpips_metric = pyiqa.create_metric('lpips',  device=device)
    psnr_metric = pyiqa.create_metric('psnr', test_y_channel=True, device=device)
    ssim_metric = pyiqa.create_metric('ssim', test_y_channel=True, device=device)

    lpips_list, psnr_list, ssim_list = [], [], []

    for gt, est in zip(gt_paths, est_paths):
        g = torch.tensor(np.array(Image.open(gt).convert('L'))).unsqueeze(0).unsqueeze(0).float() / 255.
        e = torch.tensor(np.array(Image.open(est).convert('L'))).unsqueeze(0).unsqueeze(0).float() / 255.
        if g.shape != e.shape:
            e = torch.nn.functional.interpolate(e, size=g.shape[2:], mode='bilinear', align_corners=False)
        g, e = g.to(device), e.to(device)
        lpips_list.append(lpips_metric(e, g).item())
        psnr_list.append(psnr_metric(e, g).item())
        ssim_list.append(ssim_metric(e, g).item())

    return {
    "lpips": np.mean(lpips_list),
    "psnr": np.mean(psnr_list),
    "ssim": np.mean(ssim_list),
    "summary": f"LPIPS:{np.mean(lpips_list):.3f} PSNR:{np.mean(psnr_list):.2f} SSIM:{np.mean(ssim_list):.3f}"
}

def run_sweep():
    sweep_combos = [(v1, None) for v1 in sweep_values_1] if not sweep_param_2 else list(product(sweep_values_1, sweep_values_2))
    for v1, v2 in sweep_combos:
        with open(base_config_path) as f:
            cfg = yaml.safe_load(f)

        set_nested(cfg, sweep_param_1, v1)
        label = f"{sweep_param_1.split('.')[-1]}={v1}"
        run_name = f"{sweep_param_1.replace('.', '_')}_{v1}"

        if sweep_param_2 and v2 is not None:
            set_nested(cfg, sweep_param_2, v2)
            label += f" | {sweep_param_2.split('.')[-1]}={v2}"
            run_name += f"__{sweep_param_2.replace('.', '_')}_{v2}"

        base_output = get_nested(cfg, "data.output")
        run_output = os.path.join(base_output, f"sweep_param_{sweep_param_1.replace('.', '_')}")
        cfg["data"]["exp_name"] = run_name
        cfg["data"]["output"] = run_output

        run_dir = os.path.join(run_output, run_name)
        os.makedirs(run_dir, exist_ok=True)

        with open(temp_yaml_path, 'w') as f:
            yaml.dump(cfg, f)

        print(f"\n▶ Running: {run_name}")
        subprocess.run(["python", "main.py", "--config", temp_yaml_path])

# def analyze_results():
#     sweep_combos = [(v1, None) for v1 in sweep_values_1] if not sweep_param_2 else list(product(sweep_values_1, sweep_values_2))
#     depth_grid, depth_pose_grid, labels, scores = [], [], [], []

#     for v1, v2 in sweep_combos:
#         run_name = f"{sweep_param_1.replace('.', '_')}_{v1}"
#         label = f"{sweep_param_1.split('.')[-1]}={v1}"
#         if sweep_param_2 and v2 is not None:
#             run_name += f"__{sweep_param_2.replace('.', '_')}_{v2}"
#             label += f" | {sweep_param_2.split('.')[-1]}={v2}"

#         run_output = os.path.join(get_nested(yaml.safe_load(open(base_config_path)), "data.output"),
#                                   f"sweep_param_{sweep_param_1.replace('.', '_')}")
#         run_dir = os.path.join(run_output, run_name)
#         init_dir = os.path.join(run_dir, "initialization")

#         try:
#             max_iter = int(yaml.safe_load(open(base_config_path))["num_opti_steps_for_init"]) - 1
#         except KeyError:
#             print("❌ Missing num_opti_steps_for_init")
#             continue

#         iter_list = [max_iter - 2*image_every, max_iter - image_every, max_iter]
#         depth_imgs = [load_img(os.path.join(init_dir, f"iter_{i}_vis.jpg")) for i in iter_list]
#         depth_grid.append(depth_imgs)

#         final_depth = load_img(os.path.join(init_dir, f"iter_{max_iter}_vis.jpg"))
#         final_pose = load_img(os.path.join(init_dir, f"init_f{max_iter}_pose.png"))
#         depth_pose_grid.append([final_depth, final_pose])
#         labels.append(label)

#         gt_dir = os.path.join(run_dir, "img_eval", "gt")
#         est_dir = os.path.join(run_dir, "img_eval", "est")
#         try:
#             score = evaluate_images(gt_dir, est_dir)
#         except Exception as e:
#             score = "N/A"
#             print(f"⚠️ Eval error for {run_name}: {e}")
#         scores.append(score)

#     mosaic_dir = os.path.join(get_nested(yaml.safe_load(open(base_config_path)), "data.output"),
#                               f"sweep_param_{sweep_param_1.replace('.', '_')}")
#     os.makedirs(mosaic_dir, exist_ok=True)
#     col_titles_depth = [f"iter {i}" for i in iter_list]
#     col_titles_pose = ["depth", "pose"]

#     create_mosaic(depth_grid, labels, col_titles_depth, os.path.join(mosaic_dir, "mosaic_depth.jpg"))
#     create_mosaic(depth_pose_grid, labels, col_titles_pose, os.path.join(mosaic_dir, "mosaic_depth_pose.jpg"), scores=scores)

def analyze_results():
    best_lpips = float('inf')
    best_psnr = float('-inf')
    best_lpips_label = ""
    best_psnr_label = ""
    sweep_combos = [(v1, None) for v1 in sweep_values_1] if not sweep_param_2 else list(product(sweep_values_1, sweep_values_2))
    depth_grid, depth_pose_grid, labels, scores = [], [], [], []

    for v1, v2 in sweep_combos:
        run_name = f"{sweep_param_1.replace('.', '_')}_{v1}"
        label = f"{sweep_param_1.split('.')[-1]}={v1}"
        if sweep_param_2 and v2 is not None:
            run_name += f"__{sweep_param_2.replace('.', '_')}_{v2}"
            label += f" | {sweep_param_2.split('.')[-1]}={v2}"

        run_output = os.path.join(get_nested(yaml.safe_load(open(base_config_path)), "data.output"),
                                  f"sweep_param_{sweep_param_1.replace('.', '_')}")
        run_dir = os.path.join(run_output, run_name)
        init_dir = os.path.join(run_dir, "initialization")

        try:
            max_iter = int(yaml.safe_load(open(base_config_path))["num_opti_steps_for_init"]) - 1
        except KeyError:
            print("❌ Missing num_opti_steps_for_init")
            continue

        iter_list = [max_iter - 2*image_every, max_iter - image_every, max_iter]
        depth_imgs = [load_img(os.path.join(init_dir, f"iter_{i}_vis.jpg")) for i in iter_list]
        depth_grid.append(depth_imgs)

        final_depth = load_img(os.path.join(init_dir, f"iter_{max_iter}_vis.jpg"))
        final_pose = load_img(os.path.join(init_dir, f"init_f{max_iter}_pose.png"))
        depth_pose_grid.append([final_depth, final_pose])
        labels.append(label)

        gt_dir = os.path.join(run_dir, "img_eval", "gt")
        est_dir = os.path.join(run_dir, "img_eval", "est")
        try:
            result = evaluate_images(gt_dir, est_dir)
            score = result["summary"]  # ✅ Extract summary string
        except Exception as e:
            score = "N/A"
            print(f"⚠️ Eval error for {run_name}: {e}")
        scores.append(score)

    mosaic_dir = os.path.join(get_nested(yaml.safe_load(open(base_config_path)), "data.output"),
                              f"sweep_param_{sweep_param_1.replace('.', '_')}")
    os.makedirs(mosaic_dir, exist_ok=True)
    col_titles_depth = [f"iter {i}" for i in iter_list]
    col_titles_pose = ["depth", "pose"]

    create_mosaic(depth_grid, labels, col_titles_depth, os.path.join(mosaic_dir, "mosaic_depth.jpg"))
    create_mosaic(depth_pose_grid, labels, col_titles_pose, os.path.join(mosaic_dir, "mosaic_depth_pose.jpg"), scores=scores)

    # === GLOBAL BA FINAL MOSAIC ===
    try:    
        eval_start_idx = get_nested(yaml.safe_load(open(base_config_path)), "eval_start_idx")
        eval_end_idx = get_nested(yaml.safe_load(open(base_config_path)), "eval_end_idx")
        index = eval_end_idx - eval_start_idx + 1
        global_BA_iter = get_nested(yaml.safe_load(open(base_config_path)), "num_opti_steps_for_global_BA")
        global_BA_iter = global_BA_iter -1
        print(index)
    except KeyError:
        print("❌ Missing eval_end_idx in config")
        return

    global_ba_img_grid, labels_ba = [], []

    for v1, v2 in sweep_combos:
        run_name = f"{sweep_param_1.replace('.', '_')}_{v1}"
        label = f"{sweep_param_1.split('.')[-1]}={v1}"
        if sweep_param_2 and v2 is not None:
            run_name += f"__{sweep_param_2.replace('.', '_')}_{v2}"
            label += f" | {sweep_param_2.split('.')[-1]}={v2}"

        run_output = os.path.join(
            get_nested(yaml.safe_load(open(base_config_path)), "data.output"),
            f"sweep_param_{sweep_param_1.replace('.', '_')}"
        )
        global_ba_dir = os.path.join(run_output, run_name, "global_BA")

        img_path = os.path.join(global_ba_dir, f"BA_f0{index}_{global_BA_iter}_img.jpg")
        pose_path = os.path.join(global_ba_dir, f"BA_f0{index}_whole_pose.png")

        try:
            gt_dir = os.path.join(run_output, run_name, "img_eval", "gt")
            est_dir = os.path.join(run_output, run_name, "img_eval", "est")
            scores = evaluate_images(gt_dir, est_dir)
            score_ba = scores["summary"]

            if scores["lpips"] < best_lpips:
                best_lpips = scores["lpips"]
                best_lpips_label = label

            if scores["psnr"] > best_psnr:
                best_psnr = scores["psnr"]
                best_psnr_label = label

        except Exception as e:
            score_ba = "N/A"
            print(f"⚠️ Global BA eval error for {run_name}: {e}")

        final_img = load_img(img_path)
        final_pose = load_img(pose_path)

        global_ba_img_grid.append([final_img, final_pose])
        labels_ba.append(f"{label} | {score_ba}")
        
    if global_ba_img_grid:
        col_titles_ba = ["Final Img", "Final Pose"]
        create_mosaic(
            global_ba_img_grid,
            labels_ba,
            col_titles_ba,
            os.path.join(mosaic_dir, "mosaic_global_BA.jpg")
        )
    print("\n🎯 Best Metrics for Global BA:")
    print(f"🏆 Best LPIPS: {best_lpips:.3f} from {best_lpips_label}")
    print(f"🏆 Best PSNR:  {best_psnr:.2f} from {best_psnr_label}")

if __name__ == "__main__":
    run_sweep()
    analyze_results()
