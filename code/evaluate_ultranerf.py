"""
evaluate_ultranerf.py
---------------------
Job: score a trained Ultra-NeRF run on ALL test frames (the straight-line scans it
never saw) and on a sample of training frames, inside the valid area only.

Reads:   <run>/<ckpt>                     trained network (default ckpt_final.pt)
         <datadir>/images, poses.npy, split.json, valid_mask.png,
         fan_geometry.json, frames.csv    (written by prepare_fan_data.py)
Writes:  <run>/eval/per_frame.csv         one row per scored frame
         <run>/eval/summary.csv           mean per test scan, all test, training sample
         <run>/eval/summary.json          same, plus seed and checkpoint step
         <run>/eval/images/*.png          rendered | real | difference, 8 test frames

Metrics (all inside the valid-area mask):
  PSNR   picture quality in decibels from the pixel error; higher is better
  SSIM   structural similarity 0..1 (standard constants K1=0.01, K2=0.03,
         11x11 Gaussian window, sigma 1.5); higher is better
  LPIPS  how different two images look to a pretrained AlexNet; lower is better
         (needs `pip install lpips`; skipped with a warning if not installed)

Randomness: Ultra-NeRF's renderer draws random boundaries and scatterers, so every
frame is rendered once with a torch.Generator re-seeded to --seed + frame index.
The same command always gives the same numbers.
"""

import argparse
import csv
import json
import os
import sys

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultranerf import (Embedder, UltraNeRFMLP, gaussian_kernel,  # noqa: E402
                       load_us_data, render_frame)


# ----------------------------------------------------------------- metrics
def _gauss_window(size=11, sigma=1.5, device="cpu"):
    r = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-0.5 * (r / sigma) ** 2)
    g = g / g.sum()
    return torch.outer(g, g)[None, None]                      # [1, 1, 11, 11]


def ssim_map(x, y, win):
    """Per-pixel SSIM for two [H, W] images in 0..1. Returns [H, W]."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    x, y = x[None, None], y[None, None]
    pad = win.shape[-1] // 2
    mu_x = F.conv2d(x, win, padding=pad)
    mu_y = F.conv2d(y, win, padding=pad)
    sxx = F.conv2d(x * x, win, padding=pad) - mu_x ** 2
    syy = F.conv2d(y * y, win, padding=pad) - mu_y ** 2
    sxy = F.conv2d(x * y, win, padding=pad) - mu_x * mu_y
    m = ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2))
    return m[0, 0]


def score(pred, real, mask, win, lpips_net):
    """pred, real [H, W] in 0..1, mask [H, W] bool. THE KEY STEP: score inside the mask only."""
    mse = ((pred - real) ** 2)[mask].mean()
    psnr = (10 * torch.log10(1.0 / mse)).item()
    ssim = ssim_map(pred, real, win)[mask].mean().item()
    lp = float("nan")
    if lpips_net is not None:
        # outside the mask both images are set to 0, so LPIPS ignores it equally
        p = torch.where(mask, pred, torch.zeros_like(pred))
        r = torch.where(mask, real, torch.zeros_like(real))
        to3 = lambda a: (a * 2 - 1)[None, None].expand(1, 3, *a.shape)   # [1,3,H,W] in -1..1
        lp = lpips_net(to3(p), to3(r)).item()
    return psnr, ssim, lp


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Score a trained Ultra-NeRF run")
    ap.add_argument("--run", required=True, help="run folder, e.g. runs\\tusrec_051LH")
    ap.add_argument("--ckpt", default="ckpt_final.pt")
    ap.add_argument("--datadir", required=True, help="e.g. data_unfolded\\051_LH")
    ap.add_argument("--n_train", type=int, default=100, help="training frames to score (evenly spaced)")
    ap.add_argument("--n_images", type=int, default=8, help="test frames saved as pictures")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    device = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    out_dir = os.path.join(a.run, "eval")
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    # ---- network, exactly as trained
    ck = torch.load(os.path.join(a.run, a.ckpt), map_location=device)
    targs = ck["args"]
    embedder = Embedder(targs["multires"]).to(device)
    model = UltraNeRFMLP(targs["netdepth"], targs["netwidth"], embedder.out_dim,
                         targs["output_ch"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"Loaded {a.ckpt} (step {ck['step']}) from {a.run}")

    # ---- data and fan geometry
    images, poses = load_us_data(a.datadir)                    # [N, H, W], poses in metres
    N, H, W = images.shape
    geo = json.load(open(os.path.join(a.datadir, "fan_geometry.json")))
    angles = torch.deg2rad(torch.linspace(geo["angle_min_deg"], geo["angle_max_deg"], W, device=device))
    fan = {"angles": angles, "r_inner": geo["r_inner_mm"] * 0.001}
    far = geo["depth_mm"] * 0.001
    mask = torch.from_numpy(imageio.imread(os.path.join(a.datadir, "valid_mask.png")) > 127).to(device)
    split = json.load(open(os.path.join(a.datadir, "split.json")))
    rows = {int(r["index"]): r for r in csv.DictReader(open(os.path.join(a.datadir, "frames.csv")))}

    test_idx = list(split["test"])
    tr = np.array(split["train"])
    train_idx = tr[np.linspace(0, len(tr) - 1, min(a.n_train, len(tr))).round().astype(int)].tolist()

    # 2 picture frames per test scan, at 1/3 and 2/3 along the scan
    by_scan = {}
    for i in split["test"]:
        by_scan.setdefault(rows[i]["scan"], []).append(i)
    pic_idx = set()
    for s, idx in by_scan.items():
        for f in (1 / 3, 2 / 3):
            pic_idx.add(idx[int(f * (len(idx) - 1))])
    pic_idx = set(sorted(pic_idx)[:a.n_images])

    # ---- LPIPS (optional)
    lpips_net = None
    try:
        import lpips
        lpips_net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    except Exception as e:                                     # not installed / no download
        print(f"WARNING: LPIPS skipped ({type(e).__name__}: {e}). "
              "Install it with `pip install lpips`; the first run also needs internet to "
              "download AlexNet weights (~230 MB, once).")

    kernel = gaussian_kernel(device=device)
    win = _gauss_window(device=device)
    poses_t = torch.from_numpy(poses).to(device)

    # ---- render and score
    results = []
    jobs = [("test", i) for i in test_idx] + [("train", i) for i in train_idx]
    with torch.no_grad():
        for role, i in tqdm(jobs, desc="render + score"):
            g = torch.Generator(device=device).manual_seed(a.seed + i)
            out = render_frame(model, embedder, poses_t[i], W, 0.0, 0.0, far,
                               targs["n_samples"], kernel, device, chunk=65536,
                               generator=g, fan=fan)
            pred = out["intensity_map"].T.clamp(0, 1)            # [W, H] -> [H, W] = depth x lateral
            real = torch.from_numpy(images[i]).to(device)
            psnr, ssim, lp = score(pred, real, mask, win, lpips_net)
            results.append(dict(index=i, role=role, scan=rows[i]["scan"],
                                original_frame=int(rows[i]["original_frame"]),
                                psnr_db=round(psnr, 3), ssim=round(ssim, 4),
                                lpips=round(lp, 4) if lp == lp else ""))
            if role == "test" and i in pic_idx:
                p, r = pred.cpu().numpy(), images[i]
                m = mask.cpu().numpy()
                diff = np.abs(p - r) * m
                gap = np.ones((H, 6), np.float32)
                img = np.concatenate([p * m, gap, r * m, gap, diff], axis=1)
                imageio.imwrite(os.path.join(out_dir, "images",
                                             f"{rows[i]['scan']}_f{rows[i]['original_frame']}.png"),
                                (255 * np.clip(img, 0, 1)).astype(np.uint8))

    # ---- tables
    with open(os.path.join(out_dir, "per_frame.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    def summarise(name, sel):
        v = lambda k: np.array([r[k] for r in sel if r[k] != ""], float)
        out = {"group": name, "frames": len(sel)}
        for k in ["psnr_db", "ssim", "lpips"]:
            x = v(k)
            out[f"{k}_mean"] = round(float(x.mean()), 4) if len(x) else ""
            out[f"{k}_std"] = round(float(x.std()), 4) if len(x) else ""
        return out

    test_rows = [r for r in results if r["role"] == "test"]
    summary = [summarise(f"test: {s}", [r for r in test_rows if r["scan"] == s]) for s in sorted(by_scan)]
    summary.append(summarise("TEST (all L scans)", test_rows))
    summary.append(summarise(f"TRAIN sample ({len(train_idx)} C+S frames)",
                             [r for r in results if r["role"] == "train"]))
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    json.dump({"checkpoint": a.ckpt, "step": ck["step"], "seed": a.seed,
               "lpips": lpips_net is not None, "ssim_constants": "K1=0.01, K2=0.03, 11x11, sigma 1.5",
               "summary": summary}, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)

    print("\n{:<34} {:>6} {:>14} {:>14} {:>14}".format("group", "frames", "PSNR dB", "SSIM", "LPIPS"))
    for s in summary:
        fmt = lambda k: f"{s[k + '_mean']} ± {s[k + '_std']}" if s[k + "_mean"] != "" else "-"
        print("{:<34} {:>6} {:>14} {:>14} {:>14}".format(s["group"], s["frames"], fmt("psnr_db"),
                                                         fmt("ssim"), fmt("lpips")))
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
