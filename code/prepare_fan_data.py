"""
prepare_fan_data.py
-------------------
Job: turn the fan-shaped (curvilinear-probe) TUS-REC frames of ONE arm into
rectangles where each column is one sound beam and each row is one depth, and
write them in the exact format ultranerf.py loads, with the 8 + 4 scan split.

Reads:   Dataset/frames/<subject>/<scan>.h5, Dataset/transfs/<subject>/<scan>.h5,
         Dataset/calib_matrix.csv
Writes:  data_unfolded/<subject>_<arm>/
             images/0.png, 1.png, ...   unfolded frames, 512 (depth) x 256 (beams)
             poses.npy                  [N, 4, 4] fan-apex frame -> world, in mm,
                                        world re-centred on the arm
             split.json                 {"train": [...], "test": [...]} frame indices
             frames.csv                 index -> scan name, original frame number, role
             fan_geometry.json          apex, radii, angles, centre offset
             valid_mask.png             512 x 256, white = real data, black = never
                                        recorded (fan wider than the image at depth)
             checks/                    preview images + checks.json

Coordinates used here (all in mm):
  image frame: x = to the right (columns), y = downwards (rows = depth), z = 0 plane
  fan frame:   same axes, but the origin is moved to the fan APEX (where beams meet).
               A beam at angle theta (0 = straight down, + = towards the right)
               passes through r * (sin theta, cos theta, 0).
No randomness is used, so there is no random seed.
"""

from pathlib import Path
import argparse
import csv
import json

import h5py
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

# ----------------------------------------------------------------- settings
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Dataset"
SUBJECT, ARM = "051", "LH"
TRAIN_SHAPES, TEST_SHAPES = ["C", "S"], ["L"]
FRAME_STEP = 2            # keep every 2nd frame
N_BEAMS = 256             # columns of the unfolded image (beam angles)
N_DEPTH = 512             # rows of the unfolded image (depth samples)
MASK_EVERY = 10           # frames used to find the fan outline
MASK_THRESHOLD = 10       # grey level above which a pixel counts as "inside the fan"
EDGE_INSET_DEG = 0.5      # stay this far inside the fan's side edges
OUT = ROOT / "data_unfolded" / f"{SUBJECT}_{ARM}"


def read_calibration(path):
    rows = list(csv.reader(open(path)))
    nums = [[float(v) for v in r] for r in rows if r and r[0][0] in "-0123456789"]
    return np.array(nums[0:4]), np.array(nums[4:8])     # S, T_tool_img


def scan_names():
    return [f"{ARM}_{o}_{s}_{d}" for s in TRAIN_SHAPES + TEST_SHAPES
            for o in ["Par", "Per"] for d in ["DtP", "PtD"]]


# ----------------------------------------------------------------- 1. fan geometry
def fit_fan(mask, sx, sy):
    """Find the apex, inner/outer radius and side angles of the fan, in mm."""
    rows = np.where(mask.any(1))[0]
    top, bot = rows.min(), rows.max()
    ys, left, right = [], [], []
    for r in range(top + 5, int(top + 0.6 * (bot - top)), 2):   # straight part of the sides
        xs = np.where(mask[r])[0]
        if len(xs) > 50:
            ys.append(r * sy); left.append(xs.min() * sx); right.append(xs.max() * sx)
    ys, left, right = map(np.array, (ys, left, right))
    aL, bL = np.polyfit(ys, left, 1)            # left edge:  x = aL * y + bL
    aR, bR = np.polyfit(ys, right, 1)           # right edge: x = aR * y + bR
    ay = (bR - bL) / (aL - aR)                  # where the two edges meet
    ax = aL * ay + bL
    th_min = np.arctan(aL) + np.radians(EDGE_INSET_DEG)
    th_max = np.arctan(aR) - np.radians(EDGE_INSET_DEG)

    # inner (probe face) and outer (deepest) radius: walk out along many beams
    r = np.arange(0, 250, 0.1)
    inner, outer = [], []
    for th in np.linspace(th_min + 0.05, th_max - 0.05, 41):
        cols = ((ax + r * np.sin(th)) / sx).round().astype(int)
        rws = ((ay + r * np.cos(th)) / sy).round().astype(int)
        ok = (cols >= 0) & (cols < mask.shape[1]) & (rws >= 0) & (rws < mask.shape[0])
        hit = np.zeros_like(r, bool)
        hit[ok] = mask[rws[ok], cols[ok]]
        if hit.any():
            inner.append(r[hit].min()); outer.append(r[hit].max())
    return dict(apex_x_mm=float(ax), apex_y_mm=float(ay),
                r_inner_mm=float(np.median(inner)), r_outer_mm=float(np.median(outer)),
                r_inner_spread_mm=float(np.ptp(inner)),
                angle_min_deg=float(np.degrees(th_min)),
                angle_max_deg=float(np.degrees(th_max)))


# ----------------------------------------------------------------- 2. unfolding
def unfold_lookup(g, sx, sy, H, W):
    """THE KEY STEP: for every (depth, beam) cell, which image position to read.

    Returns integer pixel corners and bilinear weights, computed once and reused
    for every frame (the geometry is the same for all frames)."""
    th = np.radians(np.linspace(g["angle_min_deg"], g["angle_max_deg"], N_BEAMS))
    r = np.linspace(g["r_inner_mm"], g["r_outer_mm"], N_DEPTH)
    R, TH = np.meshgrid(r, th, indexing="ij")               # [N_DEPTH, N_BEAMS]
    col = (g["apex_x_mm"] + R * np.sin(TH)) / sx            # fractional pixel column
    row = (g["apex_y_mm"] + R * np.cos(TH)) / sy            # fractional pixel row
    # Cells whose image position lies outside the 640 x 480 frame were never
    # recorded (the fan is wider than the image at depth). Mark them invalid.
    in_image = (col >= 0) & (col <= W - 1) & (row >= 0) & (row <= H - 1)
    c0 = np.clip(np.floor(col).astype(int), 0, W - 2)
    r0 = np.clip(np.floor(row).astype(int), 0, H - 2)
    fc, fr = np.clip(col - c0, 0, 1), np.clip(row - r0, 0, 1)
    w = [(1 - fr) * (1 - fc), (1 - fr) * fc, fr * (1 - fc), fr * fc]
    return r0, c0, w, in_image


def unfold(img, lut):
    r0, c0, w, valid = lut
    img = img.astype(np.float32)
    out = (img[r0, c0] * w[0] + img[r0, c0 + 1] * w[1]
           + img[r0 + 1, c0] * w[2] + img[r0 + 1, c0 + 1] * w[3])    # [N_DEPTH, N_BEAMS]
    return np.where(valid, out, 0.0)


def fold_back(unf, g, sx, sy, H, W):
    """Inverse mapping (for the round-trip check only): fan image from unfolded one."""
    rr, cc = np.mgrid[0:H, 0:W]
    x, y = cc * sx - g["apex_x_mm"], rr * sy - g["apex_y_mm"]
    r, th = np.hypot(x, y), np.arctan2(x, y)
    i = (r - g["r_inner_mm"]) / (g["r_outer_mm"] - g["r_inner_mm"]) * (N_DEPTH - 1)
    j = (np.degrees(th) - g["angle_min_deg"]) / (g["angle_max_deg"] - g["angle_min_deg"]) * (N_BEAMS - 1)
    inside = (i >= 0) & (i <= N_DEPTH - 1) & (j >= 0) & (j <= N_BEAMS - 1)
    i0 = np.clip(np.floor(i).astype(int), 0, N_DEPTH - 2)
    j0 = np.clip(np.floor(j).astype(int), 0, N_BEAMS - 2)
    fi, fj = i - i0, j - j0
    out = (unf[i0, j0] * (1 - fi) * (1 - fj) + unf[i0, j0 + 1] * (1 - fi) * fj
           + unf[i0 + 1, j0] * fi * (1 - fj) + unf[i0 + 1, j0 + 1] * fi * fj)
    return np.where(inside, out, 0), inside


def psnr(a, b, m):
    mse = np.mean((a[m] / 255.0 - b[m] / 255.0) ** 2)
    return float(10 * np.log10(1.0 / mse))


# ----------------------------------------------------------------- main
def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None,
                    help="output folder (default: data_unfolded/<subject>_<arm> next to Dataset)")
    ap.add_argument("--frame_step", type=int, default=FRAME_STEP,
                    help="keep every Nth frame (default 2; use e.g. 20 for a quick smoke test)")
    a = ap.parse_args()
    if a.out:
        OUT = Path(a.out)
    step = a.frame_step
    S, T_tool_img = read_calibration(DATA / "calib_matrix.csv")
    sx, sy = S[0, 0], S[1, 1]
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "checks").mkdir(exist_ok=True)
    names = scan_names()

    # --- 1. fan outline from all 12 scans, then fit
    mx, per_scan_fit = None, []
    for n in tqdm(names, desc="fan outline"):
        with h5py.File(DATA / "frames" / SUBJECT / f"{n}.h5", "r") as f:
            m = f["frames"][::MASK_EVERY].max(0)
        per_scan_fit.append(fit_fan(m > MASK_THRESHOLD, sx, sy))
        mx = m if mx is None else np.maximum(mx, m)
    H, W = mx.shape
    g = fit_fan(mx > MASK_THRESHOLD, sx, sy)
    g["fit_spread_between_scans"] = {k: float(np.ptp([p[k] for p in per_scan_fit]))
                                     for k in ["apex_x_mm", "apex_y_mm", "r_inner_mm",
                                               "angle_min_deg", "angle_max_deg"]}
    g.update(n_beams=N_BEAMS, n_depth=N_DEPTH,
             depth_mm=g["r_outer_mm"] - g["r_inner_mm"],
             pixel_mm=[float(sx), float(sy)])
    lut = unfold_lookup(g, sx, sy, H, W)
    # valid area = recorded (inside the image) AND inside the fan outline
    valid = lut[3] & (unfold(mx, lut) > MASK_THRESHOLD)
    imageio.imwrite(OUT / "valid_mask.png", (valid * 255).astype(np.uint8))
    g["valid_pct"] = round(100 * float(valid.mean()), 2)

    # pose of the fan frame: shift the image-frame origin to the apex
    T_img_fan = np.eye(4)
    T_img_fan[0, 3], T_img_fan[1, 3] = g["apex_x_mm"], g["apex_y_mm"]

    # --- 2. unfold every kept frame, build its pose
    poses, rows_csv, split = [], [], {"train": [], "test": []}
    idx = 0
    for n in tqdm(names, desc="unfold scans"):
        role = "train" if n.split("_")[2] in TRAIN_SHAPES else "test"
        with h5py.File(DATA / "frames" / SUBJECT / f"{n}.h5", "r") as f:
            frames = f["frames"][::step]
        with h5py.File(DATA / "transfs" / SUBJECT / f"{n}.h5", "r") as f:
            T = f["tforms"][::step].astype(float)
        for k in range(len(frames)):
            u = unfold(frames[k], lut)
            imageio.imwrite(OUT / "images" / f"{idx}.png", np.clip(u, 0, 255).astype(np.uint8))
            poses.append(T[k] @ T_tool_img @ T_img_fan)      # fan frame -> camera, mm
            rows_csv.append([idx, n, k * step, role])
            split[role].append(idx)
            idx += 1
    poses = np.array(poses)

    # re-centre the world on the arm: mean position of the middle of the centre beam
    mid = np.array([0, (g["r_inner_mm"] + g["r_outer_mm"]) / 2, 0, 1])
    centre = (poses @ mid)[:, :3].mean(0)
    poses[:, :3, 3] -= centre
    g["world_centre_subtracted_mm"] = centre.tolist()

    np.save(OUT / "poses.npy", poses.astype(np.float32))
    json.dump(split, open(OUT / "split.json", "w"))
    json.dump(g, open(OUT / "fan_geometry.json", "w"), indent=2)
    with open(OUT / "frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "scan", "original_frame", "role"])
        w.writerows(rows_csv)

    # --- 3. checks
    checks = {"frames_train": len(split["train"]), "frames_test": len(split["test"]),
              "calib_rotation_row_norms": np.linalg.norm(T_tool_img[:3, :3], axis=1).round(5).tolist(),
              "calib_rotation_det": float(np.linalg.det(T_tool_img[:3, :3]))}
    # (a) round trip on a few frames, inside the fan only
    with h5py.File(DATA / "frames" / SUBJECT / f"{names[0]}.h5", "r") as f:
        test_frames = f["frames"][::100]
    rt = []
    for fr in test_frames:
        back, inside = fold_back(unfold(fr, lut), g, sx, sy, H, W)
        rt.append(psnr(back, fr.astype(np.float32), inside))
    checks["round_trip_psnr_db"] = [round(v, 2) for v in rt]
    # (b) 3D position: one pixel placed two ways
    with h5py.File(DATA / "transfs" / SUBJECT / f"{names[0]}.h5", "r") as f:
        T0 = f["tforms"][0].astype(float)
    th, r = np.radians(10.0), 70.0
    p_fan = np.array([r * np.sin(th), r * np.cos(th), 0, 1])              # via new pose
    via_new = (T0 @ T_tool_img @ T_img_fan @ p_fan)[:3]
    px = np.array([g["apex_x_mm"] + r * np.sin(th), g["apex_y_mm"] + r * np.cos(th), 0, 1])
    via_old = (T0 @ T_tool_img @ px)[:3]                                 # alignment-check chain
    checks["position_check_diff_mm"] = float(np.linalg.norm(via_new - via_old))
    # (c) unfolded image corners: should not be black
    u0 = np.array(imageio.imread(OUT / "images" / "0.png"), float)
    checks["unfolded_corner_means"] = [round(float(u0[:20, :20].mean()), 1),
                                       round(float(u0[:20, -20:].mean()), 1),
                                       round(float(u0[-20:, :20].mean()), 1),
                                       round(float(u0[-20:, -20:].mean()), 1)]
    checks["unfolded_black_pixel_pct"] = round(100 * float((u0 < 1).mean()), 2)
    checks["valid_area_pct"] = g["valid_pct"]
    json.dump(checks, open(OUT / "checks" / "checks.json", "w"), indent=2)

    # preview: original fan | unfolded rectangle, for 3 frames
    for j, fr in enumerate(test_frames[:3]):
        u = unfold(fr, lut)
        u_big = np.repeat(np.repeat(u, 1, 0), 2, 1)[:, :, None]          # 512 x 512 for display
        pad = np.zeros((512, 640 + 16 + 512), np.uint8)
        pad[:480, :640] = fr
        pad[:, 656:] = np.clip(u_big[..., 0], 0, 255).astype(np.uint8)
        imageio.imwrite(OUT / "checks" / f"preview_{j}.png", pad)

    print(json.dumps({k: g[k] for k in ["apex_x_mm", "apex_y_mm", "r_inner_mm", "r_outer_mm",
                                         "depth_mm", "angle_min_deg", "angle_max_deg",
                                         "fit_spread_between_scans"]}, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
