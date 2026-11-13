"""
check_scan_alignment.py
-----------------------
Job: check whether all 12 scans of one forearm sit in the SAME 3D coordinate
frame (tracker camera space), so that Ultra-NeRF can be trained on the 8 C+S
scans and tested on the 4 L scans of that arm.

Reads only:  Dataset/transfs/<subject>/<scan>.h5   (tforms, N x 4 x 4)
             Dataset/frames/<subject>/<scan>.h5    (a few frames, for the fan mask)
             Dataset/calib_matrix.csv
Writes:      results/alignment/<subject>_<arm>_paths3d.html   (interactive 3D figure)
             results/alignment/alignment_per_scan.csv          (one row per scan)
             results/alignment/alignment_per_arm.csv           (one row per arm)

No randomness is used, so there is no random seed.
"""

from pathlib import Path
import csv

import h5py
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm

# ----------------------------------------------------------------- settings
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Dataset"
OUT = ROOT / "results" / "alignment"
SUBJECTS = ["050", "051", "052"]
ARMS = ["LH", "RH"]
TRAIN_SHAPES = ["C", "S"]          # training scans
TEST_SHAPES = ["L"]                # test scans (never seen in training)
VOXEL_MM = 2.0                     # grid size for the coverage measure
PIXEL_STEP = 8                     # sample every 8th pixel of the fan (~1.8 mm)
MASK_EVERY = 20                    # frames used to build the fan mask
OUTLINE_EVERY = 30                 # draw every 30th frame outline in the figure


def read_calibration(path):
    """Return S (pixel -> mm scaling) and T_tool_img (image mm -> tracker tool)."""
    rows = list(csv.reader(open(path)))
    nums = [[float(v) for v in r] for r in rows if r and r[0][0] in "-0123456789"]
    S = np.array(nums[0:4])
    T_tool_img = np.array(nums[4:8])
    return S, T_tool_img


def scan_names(arm):
    return [f"{arm}_{o}_{s}_{d}" for s in TRAIN_SHAPES + TEST_SHAPES
            for o in ["Par", "Per"] for d in ["DtP", "PtD"]]


def fan_mask(subject, arm):
    """Pixels that ever show tissue (inside the curved fan), from a few frames."""
    mx = None
    for name in scan_names(arm)[:4]:
        with h5py.File(DATA / "frames" / subject / f"{name}.h5", "r") as f:
            fr = f["frames"][::MASK_EVERY].max(axis=0)
        mx = fr if mx is None else np.maximum(mx, fr)
    return mx > 10                                     # (480, 640) bool


def image_points(mask, S):
    """Fan pixels as homogeneous image coordinates in mm: (4, P)."""
    rows, cols = np.nonzero(mask[::PIXEL_STEP, ::PIXEL_STEP])
    rows, cols = rows * PIXEL_STEP, cols * PIXEL_STEP
    p = np.stack([cols, rows, np.zeros_like(cols), np.ones_like(cols)]).astype(float)
    return S @ p


def to_camera(tforms, T_tool_img, pts_img_mm):
    """THE KEY STEP: image mm -> tool -> camera, for every frame. Returns (N, P, 3)."""
    T_cam_img = tforms @ T_tool_img                    # (N, 4, 4)
    return np.einsum("nij,jp->npi", T_cam_img, pts_img_mm)[..., :3]


def voxels(points):
    return set(map(tuple, np.floor(points.reshape(-1, 3) / VOXEL_MM).astype(int)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S, T_tool_img = read_calibration(DATA / "calib_matrix.csv")
    H, W = 480, 640
    corners_px = np.array([[0, 0, 0, 1], [W, 0, 0, 1], [W, H, 0, 1],
                           [0, H, 0, 1], [0, 0, 0, 1]], float).T
    corners_mm = S @ corners_px
    centre_mm = S @ np.array([[W / 2], [H / 2], [0], [1]])

    per_scan, per_arm = [], []
    jobs = [(s, a) for s in SUBJECTS for a in ARMS]
    for subject, arm in tqdm(jobs, desc="arms"):
        pts_img = image_points(fan_mask(subject, arm), S)
        fig = go.Figure()
        centroids, train_vox, test_vox = {}, set(), {}

        for name in scan_names(arm):
            with h5py.File(DATA / "transfs" / subject / f"{name}.h5", "r") as f:
                T = f["tforms"][:].astype(float)
            shape = name.split("_")[2]
            role = "train" if shape in TRAIN_SHAPES else "test"

            centres = to_camera(T, T_tool_img, centre_mm)[:, 0, :]     # (N, 3)
            steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
            det = np.linalg.det(T[:, :3, :3])
            centroids[name] = centres.mean(0)

            fan_pts = to_camera(T, T_tool_img, pts_img)                 # (N, P, 3)
            vox = voxels(fan_pts)
            if role == "train":
                train_vox |= vox
            else:
                test_vox[name] = vox

            per_scan.append(dict(
                subject=subject, arm=arm, scan=name, role=role, frames=len(T),
                path_length_mm=round(steps.sum(), 1),
                mean_step_mm=round(steps.mean(), 3),
                max_step_mm=round(steps.max(), 2),
                rotation_det_min=round(det.min(), 4),
                rotation_det_max=round(det.max(), 4)))

            dash = "solid" if role == "train" else "dash"
            fig.add_trace(go.Scatter3d(
                x=centres[:, 0], y=centres[:, 1], z=centres[:, 2], mode="lines",
                name=f"{name} ({role})", legendgroup=name,
                line=dict(width=5 if role == "test" else 3, dash=dash)))
            outl = to_camera(T[::OUTLINE_EVERY], T_tool_img, corners_mm)
            xs, ys, zs = [], [], []
            for o in outl:
                xs += list(o[:, 0]) + [None]
                ys += list(o[:, 1]) + [None]
                zs += list(o[:, 2]) + [None]
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                       legendgroup=name, showlegend=False,
                                       line=dict(width=1), opacity=0.35))

        # how far apart the 12 scans' centre points are
        C = np.array(list(centroids.values()))
        pair = np.linalg.norm(C[:, None] - C[None], axis=-1)
        spread_to_mean = np.linalg.norm(C - C.mean(0), axis=1)
        # how much of each test scan's volume the training scans also cover
        cov = {n: 100 * len(v & train_vox) / len(v) for n, v in test_vox.items()}
        for row in per_scan:
            if row["subject"] == subject and row["arm"] == arm:
                row["coverage_by_train_pct"] = round(cov.get(row["scan"], float("nan")), 1)

        per_arm.append(dict(
            subject=subject, arm=arm,
            centroid_spread_max_pair_mm=round(pair.max(), 1),
            centroid_spread_mean_to_centre_mm=round(spread_to_mean.mean(), 1),
            test_coverage_mean_pct=round(np.mean(list(cov.values())), 1),
            test_coverage_min_pct=round(min(cov.values()), 1),
            frames_train=sum(r["frames"] for r in per_scan if r["subject"] == subject
                             and r["arm"] == arm and r["role"] == "train"),
            frames_test=sum(r["frames"] for r in per_scan if r["subject"] == subject
                            and r["arm"] == arm and r["role"] == "test")))

        fig.update_layout(
            title=f"Subject {subject}, {arm}: frame-centre paths (solid = train C/S, "
                  f"dashed = test L), grey boxes = every {OUTLINE_EVERY}th frame",
            scene=dict(aspectmode="data", xaxis_title="x (mm)",
                       yaxis_title="y (mm)", zaxis_title="z (mm)"),
            height=800, legend=dict(font=dict(size=10)))
        fig.write_html(OUT / f"{subject}_{arm}_paths3d.html", include_plotlyjs="cdn")

    for fname, rows in [("alignment_per_scan.csv", per_scan),
                        ("alignment_per_arm.csv", per_arm)]:
        with open(OUT / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print("Done. Results in", OUT)


if __name__ == "__main__":
    main()
