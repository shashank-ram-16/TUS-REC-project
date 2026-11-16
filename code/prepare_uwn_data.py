"""
prepare_uwn_data.py
-------------------
Job: convert the 12 scans of one TUS-REC arm into the input format of the official
Ultra-Wide-NeRF code (Valera, Wysocki et al., ASMUS @ MICCAI 2026), with the fan
narrowed to +/-35 deg so that it fits inside the recorded 640 x 480 image.

Reads:   Dataset/frames/<subject>/<scan>.h5               original fan frames
         data_unfolded/<subject>_<arm>/fan_geometry.json   apex and radii (Step 2)
         data_unfolded/<subject>_<arm>/poses.npy, frames.csv  SAME frames and fan-frame
                                                          poses as the other methods
Writes:  data_uwn/<subject>_<arm>/volume0 ... volume11/
             images.npy   [D, H, W] uint8, each frame cropped to the fan's bounding box
             poses.npy    [D, 4, 4], local fan frame -> world (see "units" below)
         data_uwn/<subject>_<arm>/uwn_geometry.json   numbers for the config file
         data_uwn/<subject>_<arm>/checks/             round-trip check + preview

Volume numbers follow the 8 + 4 split: volume0-7 = C and S scans (train),
volume8-11 = L scans (test).

How their code places a fan (sampling.py, define_rays_numpy / image_to_pixel):
  - beam angle a runs from -A/2 to +A/2; beam direction (-sin a, cos a) in the local frame
  - the OUTER radius is 1 (normalised); the inner radius is r_n = r_inner / r_outer
  - local y is shifted so that the highest ray start is y = 0
  - the image is stretched so the fan's bounding box fills it exactly
  So a local point (x, y) is, in our fan frame (apex origin, mm, x right, y down):
        X = R * x,        Y = R * (y + r_n * cos(A/2))

Units (OUR CHOICE, to keep their code unmodified): their ray start points are in
normalised units (R = 1) while depth ("far") and pose translations are in metres.
We therefore make 1 world unit = R (the outer radius, 126.7 mm): poses get
rotation unchanged and translation = 1000 * t_mm / R (their loader multiplies by
0.001), and the config uses convex_big_radius = 1000, convex_scaling = 1, so that
far = (1000 - 1000 r_n) * 1 * 0.001 = 1 - r_n.
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

ROOT = Path(__file__).resolve().parents[1]


def bilinear(img, rows, cols):
    """Sample img [H, W] at fractional (rows, cols). Outside -> 0."""
    H, W = img.shape
    ok = (rows >= 0) & (rows <= H - 1) & (cols >= 0) & (cols <= W - 1)
    r0 = np.clip(np.floor(rows).astype(int), 0, H - 2)
    c0 = np.clip(np.floor(cols).astype(int), 0, W - 2)
    fr, fc = np.clip(rows - r0, 0, 1), np.clip(cols - c0, 0, 1)
    img = img.astype(np.float32)
    v = (img[r0, c0] * (1 - fr) * (1 - fc) + img[r0, c0 + 1] * (1 - fr) * fc
         + img[r0 + 1, c0] * fr * (1 - fc) + img[r0 + 1, c0 + 1] * fr * fc)
    return np.where(ok, v, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="051")
    ap.add_argument("--arm", default="LH")
    ap.add_argument("--half_angle_deg", type=float, default=35.0)
    ap.add_argument("--pixel_mm", type=float, default=0.25, help="pixel size of the cropped images")
    ap.add_argument("--frame_step", type=int, default=1,
                    help="extra thinning on top of data_unfolded's every-2nd frame (1 = same frames)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    src = ROOT / "data_unfolded" / f"{a.subject}_{a.arm}"
    out = Path(a.out) if a.out else ROOT / "data_uwn" / f"{a.subject}_{a.arm}"
    (out / "checks").mkdir(parents=True, exist_ok=True)
    g = json.load(open(src / "fan_geometry.json"))
    sx, sy = g["pixel_mm"]
    ax, ay = g["apex_x_mm"], g["apex_y_mm"]
    r_in, R = g["r_inner_mm"], g["r_outer_mm"]
    A2 = np.radians(a.half_angle_deg)
    r_n = r_in / R

    # ---- 1. the fan's bounding box in the original image (mm), and the crop grid
    x0, x1 = ax - R * np.sin(A2), ax + R * np.sin(A2)
    y0, y1 = ay + r_in * np.cos(A2), ay + R
    W = int(round((x1 - x0) / a.pixel_mm))
    H = int(round((y1 - y0) / a.pixel_mm))
    cols_mm = x0 + (np.arange(W) + 0.5) / W * (x1 - x0)          # pixel centres, as their
    rows_mm = y0 + (np.arange(H) + 0.5) / H * (y1 - y0)          # image_to_pixel floors them
    RR, CC = np.meshgrid(rows_mm / sy, cols_mm / sx, indexing="ij")  # fractional original pixels
    if x0 < 0 or x1 > 640 * sx or y0 < 0 or y1 > 480 * sy:
        raise ValueError(f"+/-{a.half_angle_deg} deg fan does not fit in the image: "
                         f"x {x0:.1f}-{x1:.1f} mm, y {y0:.1f}-{y1:.1f} mm")
    print(f"Fan +/-{a.half_angle_deg} deg, r {r_in:.1f}-{R:.1f} mm -> crop "
          f"x {x0:.2f}-{x1:.2f} mm, y {y0:.2f}-{y1:.2f} mm -> {W} x {H} px")

    # ---- 2. THE KEY STEP: their local frame (R = 1) -> our fan frame (mm) -> world, in R units
    M = np.eye(4)
    M[:3, :3] *= R
    M[1, 3] = R * r_n * np.cos(A2)                               # local y = 0 sits below the apex
    poses_fan = np.load(src / "poses.npy").astype(np.float64)    # fan frame -> world, mm
    rows = list(csv.DictReader(open(src / "frames.csv")))

    scans = []
    for r in rows:
        if r["scan"] not in scans:
            scans.append(r["scan"])
    order = ([s for s in scans if s.split("_")[2] in ("C", "S")]
             + [s for s in scans if s.split("_")[2] == "L"])     # volume0-7 train, 8-11 test

    vol_info = []
    for v, scan in enumerate(tqdm(order, desc="volumes")):
        sel = [r for r in rows if r["scan"] == scan][::a.frame_step]
        idx = [int(r["index"]) for r in sel]
        frame_ids = [int(r["original_frame"]) for r in sel]
        with h5py.File(ROOT / "Dataset" / "frames" / a.subject / f"{scan}.h5", "r") as f:
            frames = f["frames"][sorted(set(frame_ids))]
        lut = {fid: k for k, fid in enumerate(sorted(set(frame_ids)))}
        imgs = np.stack([bilinear(frames[lut[fid]], RR, CC) for fid in frame_ids])
        P = poses_fan[idx] @ M                                   # local (R units) -> world (mm)
        P[:, :3, :3] /= R                                        # world in R units: rotation stays pure
        P[:, :3, 3] = P[:, :3, 3] / R * 1000.0                   # their loader multiplies by 0.001
        d = out / f"volume{v}"
        d.mkdir(exist_ok=True)
        np.save(d / "images.npy", np.clip(imgs, 0, 255).astype(np.uint8))
        np.save(d / "poses.npy", P.astype(np.float32))
        vol_info.append(dict(volume=v, scan=scan, role="train" if v < 8 else "test",
                             frames=len(idx), first_index=idx[0], last_index=idx[-1]))

    # ---- 3. checks
    # (a) their sampling of OUR crop must land on the same tissue as the original image
    n_rays, n_s = 256, 512
    a_ang = np.linspace(-A2, A2, n_rays)
    t = np.linspace(0, 1 - r_n, n_s)
    dx, dy = -np.sin(a_ang)[:, None], np.cos(a_ang)[:, None]
    px = (r_n + t[None]) * dx                                     # local x
    py = (r_n + t[None]) * dy - r_n * np.cos(A2)                 # local y, top at 0
    nx = (px - px.min()) / (px.max() - px.min())                 # their image_to_pixel
    ny = (py - py.min()) / (py.max() - py.min())
    pc = np.minimum((nx * W).astype(int), W - 1)
    pr = np.minimum((ny * H).astype(int), H - 1)
    test_img = imgs[len(imgs) // 2]                              # a test-scan frame
    theirs = test_img[pr, pc].astype(np.float32)
    phys_x = ax + R * px                                          # same points, physically
    phys_y = ay + R * (py + r_n * np.cos(A2))
    with h5py.File(ROOT / "Dataset" / "frames" / a.subject / f"{order[-1]}.h5", "r") as f:
        orig = f["frames"][frame_ids[len(frame_ids) // 2]]
    ours = bilinear(orig, phys_y / sy, phys_x / sx)
    mse = np.mean((theirs / 255 - ours / 255) ** 2)
    checks = {"crop_px": [W, H], "half_angle_deg": a.half_angle_deg,
              "r_n": r_n, "unit_mm": R,
              "roundtrip_psnr_db": round(float(10 * np.log10(1 / mse)), 2),
              "pose_rotation_det_range": [float(np.linalg.det(P[:, :3, :3]).min()),
                                          float(np.linalg.det(P[:, :3, :3]).max())],
              "volumes": vol_info}
    json.dump(checks, open(out / "checks" / "checks.json", "w"), indent=2)
    imageio.imwrite(out / "checks" / "preview_crop.png", np.clip(test_img, 0, 255).astype(np.uint8))

    uwn = {"convex_fan_angle": 2 * a.half_angle_deg, "convex_big_radius": 1000.0,
           "convex_small_radius": round(1000.0 * r_n, 3), "convex_scaling": 1.0,
           "unit_mm": R, "image_size_WxH": [W, H],
           "train_volumes": ",".join(str(v) for v in range(8)),
           "test_volumes": ",".join(str(v) for v in range(8, len(order)))}
    json.dump(uwn, open(out / "uwn_geometry.json", "w"), indent=2)
    print(json.dumps({k: checks[k] for k in ["crop_px", "roundtrip_psnr_db", "pose_rotation_det_range"]}))
    print(json.dumps(uwn, indent=2))
    print("Frames per volume:", [v["frames"] for v in vol_info])


if __name__ == "__main__":
    main()
