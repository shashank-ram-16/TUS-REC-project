"""
ultragauss_reimpl.py
--------------------
Job: OUR REIMPLEMENTATION of UltraGauss (Eid, Namburete, Henriques, ICLR 2026),
written from the paper because the official code is not released yet. Trains a
cloud of 3D Gaussians ("blobs") on the 8 training scans and scores the 787 test
frames exactly like evaluate_ultranerf.py. Report its numbers as
"UltraGauss (our reimplementation)", never as UltraGauss.

Reads (all written by prepare_fan_data.py, so frames, poses, split and mask are
IDENTICAL to the Ultra-NeRF run):
    <datadir>/images/*.png      unfolded targets, 512 depths x 256 beams
    <datadir>/poses.npy         fan-apex frame -> world, mm, re-centred
    <datadir>/split.json, valid_mask.png, fan_geometry.json, frames.csv
Writes:
    <basedir>/<expname>/val/*.png        progress: rendered | real (a test frame)
    <basedir>/<expname>/ckpt_final.pt    all Gaussian parameters
    <basedir>/<expname>/eval/...         same tables/images as evaluate_ultranerf.py

What follows the paper (section numbers of the ICLR 2026 version):
    Eq. 6-7   pixel = weighted mean of Gaussians touching it + uniform background
    Eq. 8     each Gaussian evaluated in 3D at the pixel lifted onto the probe plane
    Eq. 11    precision (inverse covariance) = L L^T, L lower-triangular, diag l^2+beta
    Eq. 13    95% cut-off (chi-square 7.815): Gaussians too far from the plane skipped
    Eq. 15    shadows: cumulative absorption down each beam in the unfolded fan
    Sec. 4.6  init: mu uniform in the scanned volume, c = 0.5, alpha = 0.731,
              L entries ~ U[4, 5); learning rate 0.05 except mu (decaying from 1.6e-4)
What is OUR choice (not given in the paper, or replaced because there are no CUDA kernels):
    - plain PyTorch "patch splatting": each Gaussian is evaluated on a square
      pixel patch sized from its 95% ellipsoid (largest patch 65 px = ~14 mm)
    - coordinate unit 20 mm; mu learning rate scaled by scene size (as Kerbl et al.)
    - fixed-budget densification: every 100 steps, useless Gaussians (opacity < 0.05,
      or not touched by any training frame in the last 100 steps) are moved next to the
      highest-gradient Gaussians, and both are shrunk by 1.6x (at most 2% per step)
    - the loss is Ultra-NeRF's (0.75 MS-SSIM + 0.25 L2, inside the valid mask)
"""

import argparse
import csv
import json
import math
import os
import sys
import time

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultranerf import compute_loss                        # noqa: E402  same loss as Ultra-NeRF
from prepare_fan_data import unfold_lookup                # noqa: E402  same fan unfolding
from evaluate_ultranerf import score, _gauss_window       # noqa: E402  same metrics

H_IMG, W_IMG = 480, 640            # original fan image (rows = depth, cols = lateral)
CHI2_95 = 7.815                    # chi-square, 3 degrees of freedom, 95% (paper Eq. 13)
PATCH_SIZES = (9, 17, 25, 33, 49, 65)
CHUNK_PIXELS = 2_000_000          # patch pixels per chunk (~8 MB per array); lower if GPU memory is short
L_DIAG_MAX = 200.0                 # cap on l^2 + beta: no blob thinner than ~1/200 unit = 0.1 mm
L_OFF_MAX = 200.0                  # cap on |off-diagonal L|. Without caps L grew to 84,750 and the
                                   # run went to NaN at step ~13,700 (blobs shrunk 1.6x again and again)
BETA, EPS = 1.0, 1e-6            # diag of L >= 1 -> no Gaussian wider than ~1 unit (20 mm),
                                   # and L is always invertible (beta = 1e-3 crashed at step 3177)


# ======================================================================== model
class GaussianCloud(torch.nn.Module):
    """N Gaussians. Per Gaussian: centre mu (3), L (6), colour c (1), opacity alpha (1) = 11 numbers."""

    def __init__(self, init_points, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        N = init_points.shape[0]
        self.mu = torch.nn.Parameter(init_points.clone())                       # [N, 3] in units
        self.L_raw = torch.nn.Parameter(4 + torch.rand(N, 6, generator=g))      # l11 l21 l22 l31 l32 l33
        self.c_raw = torch.nn.Parameter(torch.zeros(N))                         # sigmoid(0) = 0.5
        self.a_raw = torch.nn.Parameter(torch.ones(N))                          # sigmoid(1) = 0.731
        self.bg_a_raw = torch.nn.Parameter(torch.tensor(math.log(math.e ** 0.1 - 1)))  # softplus -> 0.1
        self.bg_c_raw = torch.nn.Parameter(torch.tensor(-3.0))                  # sigmoid -> 0.05

    def L(self):
        l = self.L_raw
        z = torch.zeros_like(l[:, 0])
        row1 = torch.stack([l[:, 0] ** 2 + BETA, z, z], -1)
        row2 = torch.stack([l[:, 1], l[:, 2] ** 2 + BETA, z], -1)
        row3 = torch.stack([l[:, 3], l[:, 4], l[:, 5] ** 2 + BETA], -1)
        return torch.stack([row1, row2, row3], 1)                               # [N, 3, 3]

    def precision(self):
        L = self.L()
        return L @ L.transpose(1, 2) + EPS * torch.eye(3, device=L.device)      # [N, 3, 3]  Eq. 11

    def colour(self):
        return torch.sigmoid(self.c_raw)

    def opacity(self):
        return torch.sigmoid(self.a_raw)


# ======================================================================== rendering
class Renderer:
    def __init__(self, geo, unit_mm, device):
        self.U = unit_mm
        self.sx, self.sy = geo["pixel_mm"]
        self.ax, self.ay = geo["apex_x_mm"], geo["apex_y_mm"]
        self.device = device
        r0, c0, w, valid = unfold_lookup(geo, self.sx, self.sy, H_IMG, W_IMG)
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=device)
        self.i00 = t(r0 * W_IMG + c0).long()
        self.i01, self.i10, self.i11 = self.i00 + 1, self.i00 + W_IMG, self.i00 + W_IMG + 1
        self.w = [t(x).float() for x in w]
        self.valid = t(valid).float()

    def unfold(self, flat):
        """[H_IMG*W_IMG] image -> [512 depths, 256 beams], same lookup as prepare_fan_data.py."""
        out = (flat[self.i00] * self.w[0] + flat[self.i01] * self.w[1]
               + flat[self.i10] * self.w[2] + flat[self.i11] * self.w[3])
        return out * self.valid

    def _patch(self, P, mu_b, Lb, a_b, c_b, col_b, row_b):
        """K Gaussians on their P x P pixel patches -> two full-image sums (num, den)."""
        U, dev = self.U, self.device
        off = torch.arange(P, device=dev) - P // 2
        cc = col_b.round().long()[:, None, None] + off[None, None, :]            # [K, 1, P]
        rr = row_b.round().long()[:, None, None] + off[None, :, None]            # [K, P, 1]
        inb = (rr >= 0) & (rr < H_IMG) & (cc >= 0) & (cc < W_IMG)                # [K, P, P]
        dx = (cc.float() * self.sx - self.ax) / U - mu_b[:, 0, None, None]
        dy = (rr.float() * self.sy - self.ay) / U - mu_b[:, 1, None, None]
        dz = -mu_b[:, 2, None, None]                                              # pixel lies at z = 0
        q = (Lb[:, 0, 0, None, None] * dx * dx + Lb[:, 1, 1, None, None] * dy * dy
             + 2 * Lb[:, 0, 1, None, None] * dx * dy
             + 2 * dz * (Lb[:, 0, 2, None, None] * dx + Lb[:, 1, 2, None, None] * dy)
             + Lb[:, 2, 2, None, None] * dz * dz)                                 # squared Mahalanobis
        wgt = a_b[:, None, None] * torch.exp(-0.5 * q) * (inb & (q <= CHI2_95))
        flat = (rr.clamp(0, H_IMG - 1) * W_IMG + cc.clamp(0, W_IMG - 1)).expand_as(wgt).reshape(-1)
        zeros = torch.zeros(H_IMG * W_IMG, device=dev)
        n_add = zeros.index_add(0, flat, (wgt * c_b[:, None, None]).reshape(-1))
        d_add = zeros.index_add(0, flat, wgt.reshape(-1))
        return n_add, d_add

    def render(self, cloud, pose_mm, stats=None):
        """One frame. pose_mm [4,4]: fan-apex frame -> world (mm). Returns [512, 256] image."""
        U, dev = self.U, self.device
        R, t = pose_mm[:3, :3], pose_mm[:3, 3] / U
        L = cloud.L()                                                           # [N, 3, 3]
        Lam = L @ L.transpose(1, 2) + EPS * torch.eye(3, device=dev)            # Eq. 11

        # --- 1. move Gaussians into this frame (probe plane = z 0)       paper Eq. 9
        mu_p = (cloud.mu - t) @ R                                                # [N, 3]
        Lam_p = R.T @ Lam @ R                                                    # [N, 3, 3]

        # --- 2. which Gaussians touch the plane (95% ellipsoid), and how big   Eq. 13-14
        with torch.no_grad():
            # Covariance from the triangular factor (stable): Sigma = L^-T L^-1
            eye = torch.eye(3, device=dev).expand_as(L)
            Linv = torch.linalg.solve_triangular(L, eye, upper=False)
            Sig = Linv.transpose(1, 2) @ Linv
            Sig_p = R.T @ Sig @ R
            keep = mu_p[:, 2] ** 2 <= CHI2_95 * Sig_p[:, 2, 2]
            keep &= torch.isfinite(Sig_p).all(dim=(1, 2)) & torch.isfinite(mu_p).all(dim=1)
            col0 = (mu_p[:, 0] * U + self.ax) / self.sx                          # centre, in pixels
            row0 = (mu_p[:, 1] * U + self.ay) / self.sy
            half = torch.sqrt(CHI2_95 * torch.maximum(Sig_p[:, 0, 0], Sig_p[:, 1, 1])) * U / min(self.sx, self.sy)
            keep &= (col0 + half >= 0) & (col0 - half < W_IMG) & (row0 + half >= 0) & (row0 - half < H_IMG)
            idx = keep.nonzero().squeeze(1)
            sizes = torch.as_tensor(PATCH_SIZES, device=dev)
            need = 2 * torch.ceil(half[idx]) + 1
            bucket = torch.searchsorted(sizes.float(), need).clamp(max=len(PATCH_SIZES) - 1)
        if stats is not None:
            stats["visible"] = idx

        alpha, colour = cloud.opacity(), cloud.colour()
        num = torch.zeros(H_IMG * W_IMG, device=dev)
        den = torch.zeros(H_IMG * W_IMG, device=dev)
        if stats is not None:
            stats["mean_patch"] = float(torch.as_tensor(PATCH_SIZES, device=dev)[bucket].float().mean()) \
                if idx.numel() else 0.0

        # --- 3. THE KEY STEP: evaluate each touching Gaussian in 3D on its pixel patch   Eq. 8
        # Work in chunks of at most CHUNK_PIXELS patch pixels. During training each chunk is
        # "gradient-checkpointed": its intermediate arrays are freed after the forward pass and
        # recomputed during backward, so GPU memory stays small however large the blobs grow.
        for b, P in enumerate(PATCH_SIZES):
            ib_all = idx[bucket == b]
            step = max(1, CHUNK_PIXELS // (P * P))
            for s in range(0, ib_all.numel(), step):
                ib = ib_all[s:s + step]
                args = (mu_p[ib], Lam_p[ib], alpha[ib], colour[ib], col0[ib], row0[ib])
                if torch.is_grad_enabled():
                    n_add, d_add = checkpoint(self._patch, P, *args, use_reentrant=False)
                else:
                    n_add, d_add = self._patch(P, *args)
                num = num + n_add
                den = den + d_add

        # --- 4. combine with the background                              Eq. 6-7
        a_bg = torch.nn.functional.softplus(cloud.bg_a_raw)
        c_bg = torch.sigmoid(cloud.bg_c_raw)
        c_img = (num + a_bg * c_bg) / (den + a_bg)
        a_img = den + a_bg

        # --- 5. shadows: cumulative absorption down each beam             Eq. 15, Fig. 3
        c_u, a_u = self.unfold(c_img), self.unfold(a_img)                        # [512, 256]
        T = torch.exp(-torch.cumsum(a_u, dim=0) / a_u.shape[0])
        return c_u * T


# ======================================================================== helpers
def load_data(datadir, device):
    files = sorted(os.listdir(os.path.join(datadir, "images")), key=lambda f: int(os.path.splitext(f)[0]))
    imgs = np.stack([imageio.imread(os.path.join(datadir, "images", f)) for f in tqdm(files, desc="load frames")])
    poses = np.load(os.path.join(datadir, "poses.npy")).astype(np.float32)              # mm
    geo = json.load(open(os.path.join(datadir, "fan_geometry.json")))
    split = json.load(open(os.path.join(datadir, "split.json")))
    mask = imageio.imread(os.path.join(datadir, "valid_mask.png")) > 127
    rows = {int(r["index"]): r for r in csv.DictReader(open(os.path.join(datadir, "frames.csv")))}
    return (torch.from_numpy(imgs).to(device), torch.from_numpy(poses).to(device),
            geo, split, torch.from_numpy(mask).to(device), rows)


def init_points(n, poses, train_idx, geo, mask, unit_mm, seed):
    """mu uniform inside the scanned volume: random valid pixels of random training frames."""
    rng = np.random.default_rng(seed)
    ii, jj = np.nonzero(mask.cpu().numpy())
    th = np.radians(np.linspace(geo["angle_min_deg"], geo["angle_max_deg"], 256))
    r = np.linspace(geo["r_inner_mm"], geo["r_outer_mm"], 512)
    f = rng.choice(train_idx, n)
    k = rng.integers(0, len(ii), n)
    rr, tt = r[ii[k]], th[jj[k]]
    p = np.stack([rr * np.sin(tt), rr * np.cos(tt), np.zeros(n), np.ones(n)], 1)        # fan frame, mm
    P = poses.cpu().numpy()[f]
    w = np.einsum("nij,nj->ni", P, p)[:, :3] / unit_mm
    return torch.from_numpy(w.astype(np.float32))


def save_pair(path, pred, real):
    gap = np.ones((pred.shape[0], 8), np.float32)
    img = np.concatenate([pred, gap, real], 1)
    imageio.imwrite(path, (255 * np.clip(img, 0, 1)).astype(np.uint8))


# ======================================================================== train
def clamp_L(cloud):
    """Keep every blob's shape inside safe limits (applied after each update)."""
    with torch.no_grad():
        dmax = math.sqrt(L_DIAG_MAX - BETA)
        cloud.L_raw[:, [0, 2, 5]] = cloud.L_raw[:, [0, 2, 5]].clamp(-dmax, dmax)
        cloud.L_raw[:, [1, 3, 4]] = cloud.L_raw[:, [1, 3, 4]].clamp(-L_OFF_MAX, L_OFF_MAX)


def respawn(cloud, opt, grad_acc, cnt, max_frac=0.02):
    """Fixed-budget densification (OUR choice): move near-transparent Gaussians next to
    the highest-gradient ones; shrink both by 1.6x. Returns how many were moved."""
    with torch.no_grad():
        op = cloud.opacity()
        # "useless" = nearly transparent, or not touched by any training frame in the last
        # 100 steps. (A 0.005 cut-off never fired in the first smoke run: lowest opacity 0.024.)
        useless = (op < 0.05) | (cnt == 0)
        dead = useless.nonzero().squeeze(1)
        dead = dead[torch.argsort(op[dead])]                   # most transparent first
        avg = grad_acc / cnt.clamp(min=1)
        avg[useless] = -1
        # a blob already near the smallest allowed size cannot be split again
        diag = cloud.L_raw[:, [0, 2, 5]] ** 2 + BETA
        avg[(diag * 1.6 > L_DIAG_MAX).any(dim=1)] = -1
        n = int(min(dead.numel(), max_frac * op.numel(), (avg > 0).sum().item()))
        if n == 0:
            return 0
        parents = torch.topk(avg, n).indices
        children = dead[:n]
        Lp = cloud.L()[parents]
        z = torch.randn(n, 3, 1, device=Lp.device)
        step = torch.linalg.solve_triangular(Lp.transpose(1, 2), z, upper=True)        # L^-T z
        child_mu = cloud.mu[parents] + step.squeeze(-1)                                 # paper Eq. 12
        l = cloud.L_raw[parents].clone()
        for d in (0, 2, 5):                                    # diag: (l^2 + beta) * 1.6
            l[:, d] = torch.sqrt(torch.clamp(1.6 * (l[:, d] ** 2 + BETA) - BETA, min=1e-6))
        for o in (1, 3, 4):                                    # off-diagonal: * 1.6
            l[:, o] = l[:, o] * 1.6
        cloud.L_raw[parents] = l
        cloud.L_raw[children] = l
        cloud.mu[children] = child_mu
        cloud.c_raw[children] = cloud.c_raw[parents]
        cloud.a_raw[children] = cloud.a_raw[parents]
        changed = torch.cat([parents, children])
        for group in opt.param_groups:                         # forget Adam's history for these
            for p in group["params"]:
                st = opt.state.get(p, {})
                if p.dim() >= 1 and p.shape[0] == op.numel() and "exp_avg" in st:
                    st["exp_avg"][changed] = 0
                    st["exp_avg_sq"][changed] = 0
        grad_acc.zero_()
        cnt.zero_()
        return n


def train(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    out = os.path.join(a.basedir, a.expname)
    os.makedirs(os.path.join(out, "val"), exist_ok=True)
    json.dump(vars(a), open(os.path.join(out, "args.json"), "w"), indent=2)

    imgs, poses, geo, split, mask, rows = load_data(a.datadir, dev)
    tr, te = np.array(split["train"]), np.array(split["test"])
    i_val = int(te[len(te) // 2])
    rend = Renderer(geo, a.unit_mm, dev)
    mask_t = mask.T.contiguous()

    pts = init_points(a.n_gauss, poses, tr, geo, mask, a.unit_mm, a.seed).to(dev)
    cloud = GaussianCloud(pts, a.seed).to(dev)
    extent = float((pts.max(0).values - pts.min(0).values).max())
    mu_lr0 = 1.6e-4 * extent
    print(f"{len(tr)} training frames, {len(te)} test frames. {a.n_gauss:,} Gaussians "
          f"({11 * a.n_gauss:,} numbers). Scene size {extent * a.unit_mm:.0f} mm; mu lr {mu_lr0:.2e}")

    opt = torch.optim.Adam([
        {"params": [cloud.mu], "lr": mu_lr0},
        {"params": [cloud.L_raw, cloud.c_raw, cloud.a_raw, cloud.bg_a_raw, cloud.bg_c_raw], "lr": 0.05}])
    grad_acc = torch.zeros(a.n_gauss, device=dev)
    cnt = torch.zeros(a.n_gauss, device=dev)

    start, moved_total = 0, 0
    if a.resume:
        ck = torch.load(a.resume, map_location=dev)
        cloud.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start = ck["step"] + 1
        moved_total = ck.get("moved_total", 0)
        print(f"Resumed from {a.resume} at step {start}")

    times, bad_steps = [], 0
    clamp_L(cloud)                               # also makes old checkpoints safe on --resume
    pbar = tqdm(range(start, a.n_iters), initial=start, total=a.n_iters, dynamic_ncols=True)
    for step in pbar:
        t0 = time.time()
        opt.param_groups[0]["lr"] = mu_lr0 * (0.01 ** (step / a.n_iters))      # 1.6e-4 -> 1.6e-6 (scaled)
        i = int(np.random.choice(tr))
        stats = {}
        pred = rend.render(cloud, poses[i], stats)
        target = imgs[i].float() / 255.0
        loss, parts = compute_loss(pred.T.contiguous(), target.T.contiguous(), "ssim", 0.75, 7, mask=mask_t)
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss) or stats["visible"].numel() == 0:
            bad_steps += 1                       # skip this step instead of corrupting the blobs
            if bad_steps >= 50:
                raise RuntimeError(f"50 bad steps in a row at step {step} (loss {loss.item()}, "
                                   f"visible {stats['visible'].numel()}). Resume from an earlier checkpoint.")
            continue
        bad_steps = 0
        loss.backward()
        with torch.no_grad():
            v = stats["visible"]
            if cloud.mu.grad is not None:
                grad_acc[v] += cloud.mu.grad[v].norm(dim=1)
                cnt[v] += 1
        opt.step()
        clamp_L(cloud)
        if a.densify_from <= step <= a.densify_until and step % 100 == 0:
            moved_total += respawn(cloud, opt, grad_acc, cnt)
            clamp_L(cloud)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)

        if step % a.i_print == 0:
            dt = float(np.mean(times[-a.i_print:]))
            mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else 0.0
            pbar.set_postfix(loss=f"{loss.item():.4f}", visible=int(stats["visible"].numel()),
                             patch=f"{stats['mean_patch']:.0f}px", gpu_gb=f"{mem:.1f}",
                             s_per_it=f"{dt:.3f}", eta_h=f"{(a.n_iters - step) * dt / 3600:.2f}")
        if step > 0 and step % a.i_weights == 0:
            torch.save({"step": step, "model": cloud.state_dict(), "optimizer": opt.state_dict(),
                        "moved_total": moved_total, "args": vars(a)},
                       os.path.join(out, f"ckpt_{step:06d}.pt"))
        if step % a.i_img == 0 or step == a.n_iters - 1:
            with torch.no_grad():
                vp = rend.render(cloud, poses[i_val])
                vl, _ = compute_loss(vp.T.contiguous(), (imgs[i_val].float() / 255).T.contiguous(),
                                     "ssim", 0.75, 7, mask=mask_t)
            save_pair(os.path.join(out, "val", f"{step:06d}.png"), vp.cpu().numpy(),
                      imgs[i_val].cpu().numpy() / 255.0)
            tqdm.write(f"[step {step}] train {loss.item():.5f}  holdout {vl.item():.5f}  "
                       f"moved so far {moved_total:,}  max|L| {cloud.L_raw.abs().max().item():.1f}")

    torch.save({"step": a.n_iters - 1, "model": cloud.state_dict(), "optimizer": opt.state_dict(),
                "args": vars(a)}, os.path.join(out, "ckpt_final.pt"))
    dt = float(np.mean(times)) if times else float("nan")
    json.dump({"iterations": len(times), "seconds_per_iteration": dt, "device": str(dev)},
              open(os.path.join(out, "timing.json"), "w"), indent=2)
    print(f"Done. {dt:.4f} s/iteration")
    return cloud, rend, imgs, poses, split, mask, rows, out, dev


# ======================================================================== evaluate
def evaluate(cloud, rend, imgs, poses, split, mask, rows, out, dev, n_train=100, n_images=8):
    """Same frames, mask and metric code as evaluate_ultranerf.py. Rendering is deterministic."""
    ed = os.path.join(out, "eval")
    os.makedirs(os.path.join(ed, "images"), exist_ok=True)
    lpips_net = None
    try:
        import lpips
        lpips_net = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    except Exception as e:
        print(f"WARNING: LPIPS skipped ({type(e).__name__}: {e})")
    win = _gauss_window(device=dev)
    tr = np.array(split["train"])
    train_idx = tr[np.linspace(0, len(tr) - 1, min(n_train, len(tr))).round().astype(int)].tolist()
    by_scan = {}
    for i in split["test"]:
        by_scan.setdefault(rows[i]["scan"], []).append(i)
    pic = set()
    for idx in by_scan.values():
        for f in (1 / 3, 2 / 3):
            pic.add(idx[int(f * (len(idx) - 1))])
    pic = set(sorted(pic)[:n_images])

    res = []
    with torch.no_grad():
        for role, i in tqdm([("test", i) for i in split["test"]] + [("train", i) for i in train_idx],
                            desc="render + score"):
            pred = rend.render(cloud, poses[i]).clamp(0, 1)
            real = imgs[i].float() / 255.0
            psnr, ssim, lp = score(pred, real, mask, win, lpips_net)
            res.append(dict(index=i, role=role, scan=rows[i]["scan"],
                            original_frame=int(rows[i]["original_frame"]),
                            psnr_db=round(psnr, 3), ssim=round(ssim, 4),
                            lpips=round(lp, 4) if lp == lp else ""))
            if role == "test" and i in pic:
                p, r, m = pred.cpu().numpy(), real.cpu().numpy(), mask.cpu().numpy()
                gap = np.ones((p.shape[0], 6), np.float32)
                img = np.concatenate([p * m, gap, r * m, gap, np.abs(p - r) * m], 1)
                imageio.imwrite(os.path.join(ed, "images", f"{rows[i]['scan']}_f{rows[i]['original_frame']}.png"),
                                (255 * np.clip(img, 0, 1)).astype(np.uint8))

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
    json.dump({"method": "UltraGauss (our reimplementation)", "summary": summary},
              open(os.path.join(ed, "summary.json"), "w"), indent=2)
    print("\n{:<34} {:>6} {:>16} {:>16} {:>16}".format("group", "frames", "PSNR dB", "SSIM", "LPIPS"))
    for s in summary:
        fm = lambda k: f"{s[k + '_mean']} ± {s[k + '_std']}" if s[k + "_mean"] != "" else "-"
        print("{:<34} {:>6} {:>16} {:>16} {:>16}".format(s["group"], s["frames"], fm("psnr_db"),
                                                         fm("ssim"), fm("lpips")))
    print(f"\nWritten to {ed}")


def main():
    p = argparse.ArgumentParser(description="UltraGauss, our reimplementation from the ICLR 2026 paper")
    p.add_argument("--datadir", required=True)
    p.add_argument("--basedir", default="./runs")
    p.add_argument("--expname", default="ultragauss_051LH")
    p.add_argument("--n_gauss", type=int, default=100_000)
    p.add_argument("--n_iters", type=int, default=30_000)
    p.add_argument("--unit_mm", type=float, default=20.0)
    p.add_argument("--densify_from", type=int, default=500)
    p.add_argument("--densify_until", type=int, default=15_000)
    p.add_argument("--i_print", type=int, default=100)
    p.add_argument("--i_img", type=int, default=2000)
    p.add_argument("--i_weights", type=int, default=2000, help="save a checkpoint every N steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", default=None)
    p.add_argument("--eval_only", default=None, help="path to a ckpt: skip training, just evaluate")
    p.add_argument("--n_train_eval", type=int, default=100)
    a = p.parse_args()

    if a.eval_only:
        dev = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
        imgs, poses, geo, split, mask, rows = load_data(a.datadir, dev)
        ck = torch.load(a.eval_only, map_location=dev)
        n = ck["model"]["mu"].shape[0]
        cloud = GaussianCloud(torch.zeros(n, 3), a.seed).to(dev)
        cloud.load_state_dict(ck["model"])
        rend = Renderer(geo, ck["args"]["unit_mm"], dev)
        evaluate(cloud, rend, imgs, poses, split, mask, rows,
                 os.path.dirname(os.path.abspath(a.eval_only)), dev, a.n_train_eval)
        return
    cloud, rend, imgs, poses, split, mask, rows, out, dev = train(a)
    evaluate(cloud, rend, imgs, poses, split, mask, rows, out, dev, a.n_train_eval)


if __name__ == "__main__":
    main()
