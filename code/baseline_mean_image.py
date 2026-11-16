"""
baseline_mean_image.py
----------------------
Job: the "no-anatomy" baseline. Average all TRAINING frames into one image and use
that same image as the prediction for EVERY test frame, ignoring the probe pose.
It knows nothing about where the probe is, so whatever it scores is the free score
for matching overall brightness and depth fall-off. Any real method must beat it.

Reads:   <datadir>/images, split.json, valid_mask.png, frames.csv  (prepare_fan_data.py)
Writes:  <out>/mean_image.png
         <out>/eval/per_frame.csv, summary.csv, summary.json, images/*.png
         -- same format, mask and metric code as evaluate_ultranerf.py
No randomness: the result is exact and repeatable.
"""

import argparse
import csv
import json
import os
import sys

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_ultranerf import score, _gauss_window       # noqa: E402  same metrics


def main():
    ap = argparse.ArgumentParser(description="No-anatomy baseline: mean training image")
    ap.add_argument("--datadir", required=True, help="e.g. data_unfolded\\051_LH")
    ap.add_argument("--out", default="runs/baseline_mean_051LH")
    ap.add_argument("--n_train", type=int, default=100)
    ap.add_argument("--n_images", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    ed = os.path.join(a.out, "eval")
    os.makedirs(os.path.join(ed, "images"), exist_ok=True)

    split = json.load(open(os.path.join(a.datadir, "split.json")))
    rows = {int(r["index"]): r for r in csv.DictReader(open(os.path.join(a.datadir, "frames.csv")))}
    mask = torch.from_numpy(imageio.imread(os.path.join(a.datadir, "valid_mask.png")) > 127).to(dev)
    img = lambda i: torch.from_numpy(
        imageio.imread(os.path.join(a.datadir, "images", f"{i}.png")).astype(np.float32) / 255.0).to(dev)

    # ---- THE KEY STEP: one prediction for all frames = mean of the training frames
    total = None
    for i in tqdm(split["train"], desc="mean of training frames"):
        x = img(i)
        total = x if total is None else total + x
    mean_img = total / len(split["train"])                      # [512 depths, 256 beams]
    imageio.imwrite(os.path.join(a.out, "mean_image.png"),
                    (255 * mean_img.clamp(0, 1).cpu().numpy()).astype(np.uint8))

    # ---- score it exactly like the trained methods
    lpips_net = None
    try:
        import lpips
        lpips_net = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    except Exception as e:
        print(f"WARNING: LPIPS skipped ({type(e).__name__}: {e})")
    win = _gauss_window(device=dev)
    tr = np.array(split["train"])
    train_idx = tr[np.linspace(0, len(tr) - 1, min(a.n_train, len(tr))).round().astype(int)].tolist()
    by_scan = {}
    for i in split["test"]:
        by_scan.setdefault(rows[i]["scan"], []).append(i)
    pic = set()
    for idx in by_scan.values():
        for f in (1 / 3, 2 / 3):
            pic.add(idx[int(f * (len(idx) - 1))])
    pic = set(sorted(pic)[:a.n_images])

    res = []
    with torch.no_grad():
        for role, i in tqdm([("test", i) for i in split["test"]] + [("train", i) for i in train_idx],
                            desc="score"):
            real = img(i)
            psnr, ssim, lp = score(mean_img, real, mask, win, lpips_net)
            res.append(dict(index=i, role=role, scan=rows[i]["scan"],
                            original_frame=int(rows[i]["original_frame"]),
                            psnr_db=round(psnr, 3), ssim=round(ssim, 4),
                            lpips=round(lp, 4) if lp == lp else ""))
            if role == "test" and i in pic:
                p, r, m = mean_img.cpu().numpy(), real.cpu().numpy(), mask.cpu().numpy()
                gap = np.ones((p.shape[0], 6), np.float32)
                im = np.concatenate([p * m, gap, r * m, gap, np.abs(p - r) * m], 1)
                imageio.imwrite(os.path.join(ed, "images", f"{rows[i]['scan']}_f{rows[i]['original_frame']}.png"),
                                (255 * np.clip(im, 0, 1)).astype(np.uint8))

    with open(os.path.join(ed, "per_frame.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)

    def summ(name, sel):
        o = {"group": name, "frames": len(sel)}
        for k in ["psnr_db", "ssim", "lpips"]:
            x = np.array([r[k] for r in sel if r[k] != ""], float)
            o[f"{k}_mean"] = round(float(x.mean()), 4) if len(x) else ""
            o[f"{k}_std"] = round(float(x.std()), 4) if len(x) else ""
        return o

    t_rows = [r for r in res if r["role"] == "test"]
    summary = [summ(f"test: {s}", [r for r in t_rows if r["scan"] == s]) for s in sorted(by_scan)]
    summary.append(summ("TEST (all L scans)", t_rows))
    summary.append(summ(f"TRAIN sample ({len(train_idx)} C+S frames)", [r for r in res if r["role"] == "train"]))
    with open(os.path.join(ed, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    json.dump({"method": "No-anatomy baseline (mean training image)", "summary": summary},
              open(os.path.join(ed, "summary.json"), "w"), indent=2)
    print("\n{:<34} {:>6} {:>16} {:>16} {:>16}".format("group", "frames", "PSNR dB", "SSIM", "LPIPS"))
    for s in summary:
        fm = lambda k: f"{s[k + '_mean']} ± {s[k + '_std']}" if s[k + "_mean"] != "" else "-"
        print("{:<34} {:>6} {:>16} {:>16} {:>16}".format(s["group"], s["frames"], fm("psnr_db"),
                                                         fm("ssim"), fm("lpips")))
    print(f"\nWritten to {ed}")


if __name__ == "__main__":
    main()
