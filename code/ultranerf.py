"""
Ultra-NeRF in PyTorch -- one file, whole method.

Reimplementation of:
  Wysocki, Azampour, Eilers, Busam, Salehi, Navab.
  "Ultra-NeRF: Neural Radiance Fields for Ultrasound Imaging". MIDL 2023.
  Original TensorFlow 1.15 code: https://github.com/magdalena-wysocki/ultra-nerf

Why this file exists:
  The original code is TensorFlow 1.15 / Python 3.7. TF 1.15 GPU builds target
  CUDA 10.0, which does not support Ampere GPUs (RTX 30-series). This is a
  line-for-line faithful PyTorch port so the method runs on a modern GPU.

Correctness is not assumed. `check_pretrained.py` loads the authors' own
trained weights (model_200000.npy) into this exact network and renders with
this exact renderer. If the picture comes out right, the port is right.

WHAT THE METHOD DOES (one paragraph)
  A 2D ultrasound frame is not a photograph. Each image column is one
  transducer element firing a pulse straight down; the pixels down that column
  are echoes coming back at increasing time, so pixel row == depth. Ultra-NeRF
  learns a 3D field of *acoustic tissue properties* (attenuation, reflection,
  scattering) rather than colour and density. To render a frame, it fires
  simulated rays from the probe face, asks a small MLP for those properties at
  points along each ray, then runs an ultrasound image-formation model down the
  ray to produce the echo. The loss is the difference between that simulated
  frame and the real recorded frame.

Author-facing note: every tensor shape is written in the comments. Read them.
"""

import argparse
import json
import math
import os
import random
import time

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# pytorch_msssim gives MS-SSIM (multi-scale structural similarity -- how alike
# two images look in structure, checked at several zoom levels). The original
# uses tf.image.ssim_multiscale with the same window and constants.
try:
    from pytorch_msssim import ms_ssim
except ImportError:
    ms_ssim = None


# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------

def load_us_data(datadir):
    """Load one tracked ultrasound sweep.

    Expects:
      <datadir>/images/0.png, 1.png, ...   grayscale B-mode frames
      <datadir>/poses.npy                  [N, 4, 4] probe-to-world matrices

    Returns:
      images [N, H, W] float32 in 0..1
      poses  [N, 4, 4] float32, translation converted from millimetres to metres
    """
    poses = np.load(os.path.join(datadir, "poses.npy"))

    imgdir = os.path.join(datadir, "images")
    files = [f for f in os.listdir(imgdir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    # Sort numerically ("10.png" must come after "9.png", not after "1.png").
    files = sorted(files, key=lambda f: int(os.path.splitext(f)[0]))

    if poses.shape[0] != len(files):
        raise ValueError(
            f"{len(files)} images but {poses.shape[0]} poses in {datadir}. They must match."
        )

    imgs = np.stack([imageio.imread(os.path.join(imgdir, f)) for f in files]).astype(np.float32)
    if imgs.ndim == 4:  # a colour PNG of a grayscale image -- take one channel
        imgs = imgs[..., 0]
    imgs /= 255.0

    poses = poses.astype(np.float32).copy()
    poses[:, :3, 3] *= 0.001  # millimetres -> metres. The original does this too.

    return imgs, poses


# ----------------------------------------------------------------------------
# 2. RAY GEOMETRY -- linear probe
# ----------------------------------------------------------------------------

def get_rays_us_linear(W, sw, c2w, device):
    """Rays for a LINEAR probe (flat transducer face, parallel beams).

    One ray per image column. All rays are parallel -- they do NOT fan out from
    a single point like a camera. That is the geometric difference between an
    ultrasound probe and a pinhole camera.

    Args:
      W    number of image columns (= number of rays), e.g. 256
      sw   metres per column, e.g. 0.080 m probe width / 256 columns
      c2w  [3, 4] or [4, 4] probe pose: rotation R and translation t

    Returns:
      rays_o [W, 3] where each ray starts, spread along the probe face
      rays_d [W, 3] unit direction, identical for every ray (straight into tissue)
    """
    R = c2w[:3, :3].to(device)
    t = c2w[:3, 3].to(device)

    # Element positions along the probe face, centred on zero: -W/2 .. W/2-1
    x = torch.arange(-W / 2, W / 2, dtype=torch.float32, device=device) * sw
    zeros = torch.zeros_like(x)
    origin_base = torch.stack([x, zeros, zeros], dim=1)          # [W, 3] in probe frame

    rays_o = origin_base @ R.T + t                                # [W, 3] in world frame

    # The probe images along its own +y axis. Rotate that into world coords.
    dirs_base = torch.tensor([0.0, 1.0, 0.0], device=device)
    rays_d = (R @ dirs_base).expand_as(rays_o).contiguous()       # [W, 3]

    return rays_o, rays_d


def get_rays_us_curvilinear(angles, r_inner, c2w, device):
    """Rays for a CURVILINEAR probe (curved face, beams fan out). [TUS-REC]

    The pose c2w is the FAN frame: origin at the apex (where all beams would
    meet if extended upwards), +x to the right, +y straight down.
    Beam j points along (sin a_j, cos a_j, 0) and starts on the probe face,
    r_inner metres from the apex.

    Args:
      angles   [W] beam angles in radians (0 = straight down, + = right)
      r_inner  probe-face radius in metres (~0.0407 for the 4DC7-3/40)
      c2w      [4, 4] fan-frame pose

    Returns:
      rays_o [W, 3] start points on the curved face, world frame
      rays_d [W, 3] unit directions, a different one for every beam
    """
    R = c2w[:3, :3].to(device)
    t = c2w[:3, 3].to(device)
    dirs_base = torch.stack([torch.sin(angles), torch.cos(angles),
                             torch.zeros_like(angles)], dim=1)         # [W, 3] fan frame
    rays_o = (r_inner * dirs_base) @ R.T + t                          # [W, 3] world
    rays_d = dirs_base @ R.T                                          # [W, 3] world
    return rays_o, rays_d


# ----------------------------------------------------------------------------
# 3. POSITIONAL ENCODING
# ----------------------------------------------------------------------------

class Embedder(nn.Module):
    """Positional encoding: turn (x, y, z) into 63 numbers.

    A plain MLP fed raw coordinates can only learn smooth, blurry fields. Feeding
    it sin/cos at many frequencies lets it represent fine detail -- which
    ultrasound needs, because speckle (the grainy salt-and-pepper texture of
    ultrasound -- interference between scatterers, not real tissue structure) is
    high frequency.

    multires=10 -> 3 raw + 3 coords * 10 freqs * 2 functions (sin, cos) = 63.
    """

    def __init__(self, multires=10):
        super().__init__()
        self.multires = multires
        freqs = 2.0 ** torch.linspace(0.0, multires - 1, multires)   # 1, 2, 4, ... 512
        self.register_buffer("freqs", freqs, persistent=False)
        self.out_dim = 3 + 3 * multires * 2

    def forward(self, x):
        # x [P, 3] -> [P, 63]
        scaled = x[..., None, :] * self.freqs[:, None]               # [P, multires, 3]
        scaled = scaled.reshape(*x.shape[:-1], -1)                   # [P, multires*3]
        # Order matters for loading the authors' weights: for each frequency,
        # sin of all 3 coords, then cos of all 3 coords.
        parts = [x]
        for i in range(self.multires):
            block = scaled[..., i * 3:(i + 1) * 3]
            parts.append(torch.sin(block))
            parts.append(torch.cos(block))
        return torch.cat(parts, dim=-1)


# ----------------------------------------------------------------------------
# 4. THE NETWORK
# ----------------------------------------------------------------------------

class UltraNeRFMLP(nn.Module):
    """8 hidden layers of 128 units, skip connection at layer 5. 132,485 params.

    Smaller than a normal NeRF (which uses 256 units). Output is 5 numbers per
    3D point, not 4:
        0  attenuation coefficient   how fast sound energy dies here
        1  reflection coefficient    how much energy bounces at a boundary here
        2  border logit              is there a tissue boundary here
        3  scatterer density logit   how many tiny scatterers sit here
        4  scatter amplitude logit   how bright those scatterers are
    """

    def __init__(self, D=8, Wdim=128, input_ch=63, output_ch=5, skips=(4,)):
        super().__init__()
        self.D, self.skips = D, set(skips)
        layers = []
        in_ch = input_ch
        for i in range(D):
            layers.append(nn.Linear(in_ch, Wdim))
            in_ch = Wdim + input_ch if i in self.skips else Wdim
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(Wdim, output_ch)

    def forward(self, x):
        # x [P, 63] -> [P, 5]
        h = x
        for i, layer in enumerate(self.layers):
            h = F.relu(layer(h))
            if i in self.skips:
                h = torch.cat([x, h], dim=-1)   # re-inject the raw encoding
        return self.out(h)


# ----------------------------------------------------------------------------
# 5. THE ULTRASOUND RENDERER  <-- this is the heart of the paper
# ----------------------------------------------------------------------------

def gaussian_kernel(size=3, std_lateral=2.0, std_axial=1.0, device="cpu"):
    """The PSF (point spread function -- the blur the probe itself adds).

    A 7x7 Gaussian, wider across the beam (lateral) than along depth (axial).
    That asymmetry is physical: ultrasound resolves depth better than width.
    Matches the original: Normal(0, std*2) on the first axis, Normal(0, std) on
    the second, outer product, normalised to sum to 1.
    """
    r = torch.arange(-size, size + 1, dtype=torch.float32, device=device)
    gx = torch.exp(-0.5 * (r / std_lateral) ** 2) / (std_lateral * math.sqrt(2 * math.pi))
    gy = torch.exp(-0.5 * (r / std_axial) ** 2) / (std_axial * math.sqrt(2 * math.pi))
    k = torch.outer(gx, gy)
    return (k / k.sum())[None, None]                                # [1, 1, 7, 7]


def _conv_psf(x, kernel):
    """Blur a [n_rays, n_samples] map with the PSF. Axis 0 = lateral, axis 1 = depth."""
    pad = kernel.shape[-1] // 2
    return F.conv2d(x[None, None], kernel, padding=pad)[0, 0]


def render_ultrasound(raw, z_vals, kernel, generator=None):
    """Turn per-point tissue properties into one simulated ultrasound frame.

    Args:
      raw    [n_rays, n_samples, 5]  raw MLP output
      z_vals [n_rays, n_samples]     depth of each sample, in metres
      kernel [1, 1, 7, 7]            the PSF

    Returns dict; 'intensity_map' [n_rays, n_samples] is the simulated frame.

    The five steps, in order:
      A. Attenuation   -- energy left after travelling through tissue
      B. Reflection    -- energy lost bouncing off boundaries
      C. Backscatter   -- the speckle texture
      D. Combine       -- echo = backscatter + boundary reflection
    """
    # Distance between consecutive samples. Sampling is equidistant, so these
    # are all equal, but we compute them the way the original does.
    dists = torch.abs(z_vals[..., :-1] - z_vals[..., 1:])
    dists = torch.cat([dists, dists[:, -1:]], dim=-1)               # [n_rays, n_samples]

    # ---- A. ATTENUATION ----------------------------------------------------
    # Beer-Lambert: intensity decays as exp(-alpha * distance). alpha is forced
    # positive with abs() because tissue cannot amplify sound.
    attenuation_coeff = torch.abs(raw[..., 0])
    attenuation = torch.exp(-attenuation_coeff * dists)
    # Running product DOWN the ray, excluding the current sample: how much
    # energy is still left by the time the pulse arrives here.
    attenuation_transmission = torch.cumprod(
        torch.cat([torch.ones_like(attenuation[:, :1]), attenuation[:, :-1]], dim=1), dim=1
    )                                                                # [n_rays, n_samples]

    # ---- B. REFLECTION -----------------------------------------------------
    prob_border = torch.sigmoid(raw[..., 2])                         # 0..1
    # Sample a hard yes/no boundary. Sampling (not just using the probability)
    # is what produces crisp interfaces instead of a smeared average.
    # stop-gradient: the sample is treated as a constant during backprop.
    border_indicator = torch.bernoulli(prob_border, generator=generator).detach()

    reflection_coeff = torch.sigmoid(raw[..., 1])                    # 0..1
    reflection_transmission = 1.0 - reflection_coeff * border_indicator
    reflection_transmission = torch.cumprod(
        torch.cat([torch.ones_like(reflection_transmission[:, :1]),
                   reflection_transmission[:, :-1]], dim=1), dim=1
    )
    border_convolution = _conv_psf(border_indicator, kernel)

    # ---- C. BACKSCATTER (speckle) -----------------------------------------
    density_coeff = torch.sigmoid(raw[..., 3])
    scatterers_density = torch.bernoulli(density_coeff, generator=generator).detach()
    amplitude = torch.sigmoid(raw[..., 4])
    scatterers_map = scatterers_density * amplitude
    psf_scatter = _conv_psf(scatterers_map, kernel)

    # ---- D. COMBINE --------------------------------------------------------
    transmission = attenuation_transmission * reflection_transmission
    b = transmission * psf_scatter                                   # speckle term
    r = transmission * reflection_coeff * border_convolution         # specular term
    intensity_map = b + r

    return {
        "intensity_map": intensity_map,
        "attenuation_coeff": attenuation_coeff,
        "reflection_coeff": reflection_coeff,
        "attenuation_transmission": attenuation_transmission,
        "reflection_transmission": reflection_transmission,
        "scatterers_density": scatterers_density,
        "scatterers_density_coeff": density_coeff,
        "scatter_amplitude": amplitude,
        "b": b,
        "r": r,
        "transmission": transmission,
    }


def render_frame(model, embedder, pose, W, sw, near, far, n_samples, kernel,
                 device, chunk=0, generator=None, fan=None):
    """Render one full ultrasound frame from one probe pose.

    Returns dict of [W, n_samples] maps. Note the orientation: axis 0 is the
    image COLUMN (lateral) and axis 1 is DEPTH. The recorded frame is
    [depth, lateral], so it gets transposed before comparison.
    """
    if fan is None:
        rays_o, rays_d = get_rays_us_linear(W, sw, pose, device)      # [W,3], [W,3]
    else:  # curvilinear probe: fan-out beams (fan = {"angles": [W], "r_inner": m})
        rays_o, rays_d = get_rays_us_curvilinear(fan["angles"], fan["r_inner"], pose, device)

    # Equidistant sampling from near to far. NOT stratified/random like a normal
    # NeRF: in ultrasound, position along the ray is time-of-flight, which the
    # scanner already fixed. Sample n at depth (n/N) * probe_depth.
    t = torch.linspace(0.0, 1.0, n_samples, device=device)
    z_vals = near * (1.0 - t) + far * t                               # [n_samples]
    z_vals = z_vals.expand(W, n_samples)                              # [W, n_samples]

    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., None]  # [W, n_samples, 3]
    flat = pts.reshape(-1, 3)                                          # [W*n_samples, 3]

    if chunk and chunk > 0:
        raw = torch.cat([model(embedder(flat[i:i + chunk]))
                         for i in range(0, flat.shape[0], chunk)], dim=0)
    else:
        raw = model(embedder(flat))
    raw = raw.reshape(W, n_samples, -1)                                # [W, n_samples, 5]

    return render_ultrasound(raw, z_vals, kernel, generator=generator)


# ----------------------------------------------------------------------------
# 6. LOSS
# ----------------------------------------------------------------------------

def compute_loss(pred, target, mode, ssim_lambda, filter_size, mask=None):
    """pred, target: [n_rays, n_samples] in 0..1. Returns (total, parts dict).

    mask (optional, same shape, bool): True where the recorded frame has real
    data. Outside it, the target is replaced by the (frozen) prediction, so those
    pixels give zero error and the network is never taught "black tissue" there.
    """
    if mask is not None:
        target = torch.where(mask, target, pred.detach())
    l2 = F.mse_loss(pred, target)
    if mode == "l2":
        return l2, {"l2": l2.item()}

    # MS-SSIM wants [batch, channel, H, W].
    p = pred[None, None].clamp(0, 1)
    t = target[None, None].clamp(0, 1)
    # k2=0.1 is the paper's value, not the usual 0.03.
    msssim = ms_ssim(p, t, data_range=1.0, win_size=filter_size,
                     win_sigma=1.5, K=(0.01, 0.1))
    ssim_loss = 1.0 - msssim
    total = ssim_lambda * ssim_loss + (1.0 - ssim_lambda) * l2
    return total, {"ssim": ssim_loss.item(), "l2": l2.item()}


# ----------------------------------------------------------------------------
# 7. LOGGING HELPERS
# ----------------------------------------------------------------------------

def to8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def save_comparison(path, pred, target):
    """Side-by-side PNG: rendered frame | recorded frame. Both [depth, lateral]."""
    gap = np.ones((pred.shape[0], 8), dtype=np.float32)
    imageio.imwrite(path, to8b(np.concatenate([pred, gap, target], axis=1)))


def save_maps(outdir, out):
    """Save the intermediate acoustic maps. These are the interpretable part --
    the reason this method is interesting for a thesis, not just the final image."""
    os.makedirs(outdir, exist_ok=True)
    for k in ["attenuation_coeff", "reflection_coeff", "scatterers_density",
              "scatter_amplitude", "attenuation_transmission", "b", "r"]:
        m = out[k].detach().cpu().numpy().T                  # -> [depth, lateral]
        lo, hi = float(m.min()), float(m.max())
        m = (m - lo) / (hi - lo + 1e-8)
        imageio.imwrite(os.path.join(outdir, f"{k}.png"), to8b(m))


# ----------------------------------------------------------------------------
# 8. ARGUMENTS
# ----------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Ultra-NeRF (PyTorch port)")
    # data
    p.add_argument("--datadir", type=str, required=True,
                   help="folder holding images/ and poses.npy")
    p.add_argument("--basedir", type=str, default="./runs")
    p.add_argument("--expname", type=str, default="ultranerf")
    # probe geometry, in millimetres
    p.add_argument("--probe_depth", type=float, default=140.0)
    p.add_argument("--probe_width", type=float, default=80.0)
    p.add_argument("--probe_type", type=str, default="linear",
                   choices=["linear", "curvilinear"],
                   help="curvilinear reads fan_geometry.json (and valid_mask.png, "
                        "split.json if present) from --datadir; probe_depth/width are then ignored")
    # model
    p.add_argument("--netdepth", type=int, default=8)
    p.add_argument("--netwidth", type=int, default=128)
    p.add_argument("--multires", type=int, default=10, help="positional-encoding frequencies")
    p.add_argument("--output_ch", type=int, default=5)
    p.add_argument("--n_samples", type=int, default=512, help="samples per ray (depth rows)")
    # training
    p.add_argument("--n_iters", type=int, default=200000)
    p.add_argument("--lrate", type=float, default=1e-4)
    p.add_argument("--lrate_decay", type=int, default=250, help="in thousands of steps")
    p.add_argument("--loss", type=str, default="ssim", choices=["l2", "ssim"])
    p.add_argument("--ssim_lambda", type=float, default=0.75)
    p.add_argument("--ssim_filter_size", type=int, default=7)
    p.add_argument("--chunk", type=int, default=0,
                   help="points per forward pass; 0 = all at once. Lower it if you hit OOM.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init", type=str, default="torch", choices=["torch", "glorot"],
                   help="glorot = TensorFlow/Keras default: Glorot-uniform weights, zero biases")
    p.add_argument("--device", type=str, default="cuda")
    # logging
    p.add_argument("--i_print", type=int, default=200)
    p.add_argument("--i_img", type=int, default=2000)
    p.add_argument("--i_weights", type=int, default=5000)
    p.add_argument("--resume", type=str, default=None, help="path to a .pt checkpoint")
    return p


# ----------------------------------------------------------------------------
# 9. TRAIN
# ----------------------------------------------------------------------------

def train(args):
    # One seed, recorded with every result.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    if args.device == "cuda" and device.type == "cpu":
        print("WARNING: CUDA was requested but is not available. Falling back to CPU. "
              "Training on CPU is not practical for 200k iterations.")

    if args.loss == "ssim" and ms_ssim is None:
        raise ImportError("--loss ssim needs pytorch_msssim. Run: pip install pytorch_msssim")

    outdir = os.path.join(args.basedir, args.expname)
    os.makedirs(os.path.join(outdir, "val"), exist_ok=True)
    with open(os.path.join(outdir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---- data ----
    images, poses = load_us_data(args.datadir)
    N, H, W = images.shape
    print(f"Loaded {N} frames of {H}x{W} (depth x lateral) from {args.datadir}")

    scaling = 0.001                       # millimetres -> metres
    near = 0.0
    far = args.probe_depth * scaling      # 0.140 m
    sw = (args.probe_width * scaling) / float(W)   # metres per image column
    if args.n_samples != H:
        print(f"NOTE: n_samples={args.n_samples} but the image has {H} depth rows. "
              f"They should normally match.")

    # Curvilinear probe (TUS-REC): beam angles, face radius and depth come from
    # the fan geometry written by prepare_fan_data.py, not from the arguments.
    fan, mask_t = None, None
    if args.probe_type == "curvilinear":
        with open(os.path.join(args.datadir, "fan_geometry.json")) as f:
            geo = json.load(f)
        if geo["n_beams"] != W or geo["n_depth"] != H:
            raise ValueError(f"fan_geometry.json says {geo['n_depth']}x{geo['n_beams']} "
                             f"but images are {H}x{W}.")
        angles = torch.deg2rad(torch.linspace(geo["angle_min_deg"], geo["angle_max_deg"],
                                              W, device=device))
        fan = {"angles": angles, "r_inner": geo["r_inner_mm"] * scaling}
        far = geo["depth_mm"] * scaling
        print(f"Curvilinear probe: {W} beams from {geo['angle_min_deg']:.1f} to "
              f"{geo['angle_max_deg']:.1f} deg, face radius {geo['r_inner_mm']:.1f} mm, "
              f"depth {geo['depth_mm']:.1f} mm")
        mpath = os.path.join(args.datadir, "valid_mask.png")
        if os.path.exists(mpath):
            m = imageio.imread(mpath) > 127                              # [H, W]
            mask_t = torch.from_numpy(m.T.copy()).to(device)             # [W, H]
            print(f"Valid-area mask: {100 * m.mean():.1f}% of each frame is real data")

    # Frames are compared as [lateral, depth], so transpose once here.
    images_t = torch.from_numpy(images).to(device).permute(0, 2, 1).contiguous()  # [N, W, H]
    poses_t = torch.from_numpy(poses).to(device)

    split_path = os.path.join(args.datadir, "split.json")
    if os.path.exists(split_path):
        # Fixed split from prepare_fan_data.py: train = C+S scans, test = L scans.
        with open(split_path) as f:
            split = json.load(f)
        i_train = np.array(split["train"])
        i_test = np.array(split["test"])
        i_val = int(i_test[len(i_test) // 2])         # one test frame shown during training
        print(f"split.json: {len(i_train)} training frames, {len(i_test)} test frames "
              f"(never trained on). Progress images use test frame {i_val}.")
    else:
        # Hold out the frame nearest the average probe position, as the original does.
        centre = poses[:, :3, 3].mean(0)
        i_val = int(np.argmin(((poses[:, :3, 3] - centre) ** 2).sum(-1)))
        i_train = np.array([i for i in range(N) if i != i_val])
        print(f"Holdout (validation) frame index: {i_val}. Training on the other {len(i_train)}.")

    # ---- model ----
    embedder = Embedder(args.multires).to(device)
    model = UltraNeRFMLP(args.netdepth, args.netwidth, embedder.out_dim,
                         args.output_ch).to(device)
    # Starting weights. "torch" = PyTorch default (as before).
    # "glorot" = what the original TensorFlow/Keras code uses: Glorot-uniform
    # weights and zero biases. Only the starting point changes, not the network.
    if args.init == "glorot":
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)   # same as Keras 'glorot_uniform'
                nn.init.zeros_(m.bias)              # Keras starts biases at zero
    print(f"Weight initialization: {args.init}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.netdepth} x {args.netwidth}, input {embedder.out_dim}, "
          f"{n_params:,} parameters")

    kernel = gaussian_kernel(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)
    # Exponential decay to 0.1x every lrate_decay*1000 steps -- the original schedule.
    gamma = 0.1 ** (1.0 / (args.lrate_decay * 1000))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start = ck["step"] + 1
        print(f"Resumed from {args.resume} at step {start}")

    # ---- loop ----
    timings = []
    pbar = tqdm(range(start, args.n_iters), initial=start, total=args.n_iters,
                dynamic_ncols=True)
    for step in pbar:
        t0 = time.time()

        img_i = int(np.random.choice(i_train))
        target = images_t[img_i]                                # [W, H]
        pose = poses_t[img_i]

        out = render_frame(model, embedder, pose, W, sw, near, far,
                           args.n_samples, kernel, device, chunk=args.chunk, fan=fan)
        total, parts = compute_loss(out["intensity_map"], target,
                                    args.loss, args.ssim_lambda, args.ssim_filter_size,
                                    mask=mask_t)

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        scheduler.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.time() - t0)

        if step % args.i_print == 0:
            mean_dt = float(np.mean(timings[-args.i_print:]))
            remaining = (args.n_iters - step) * mean_dt / 3600.0
            pbar.set_postfix(loss=f"{total.item():.5f}",
                             s_per_it=f"{mean_dt:.3f}",
                             eta_h=f"{remaining:.1f}")

        if step % args.i_img == 0 or step == args.n_iters - 1:
            model.eval()
            with torch.no_grad():
                vout = render_frame(model, embedder, poses_t[i_val], W, sw, near, far,
                                    args.n_samples, kernel, device,
                                    chunk=args.chunk or 65536, fan=fan)
                vpred = vout["intensity_map"]
                vtot, vparts = compute_loss(vpred, images_t[i_val], args.loss,
                                            args.ssim_lambda, args.ssim_filter_size,
                                            mask=mask_t)
            model.train()
            save_comparison(os.path.join(outdir, "val", f"{step:06d}.png"),
                            vpred.cpu().numpy().T, images[i_val])
            save_maps(os.path.join(outdir, "val", f"{step:06d}_maps"), vout)
            tqdm.write(f"[step {step}] train {total.item():.5f}  "
                       f"holdout {vtot.item():.5f}  parts {vparts}")

        if step % args.i_weights == 0 and step > 0:
            torch.save({"step": step, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "args": vars(args)},
                       os.path.join(outdir, f"ckpt_{step:06d}.pt"))

    torch.save({"step": args.n_iters - 1, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "args": vars(args)},
               os.path.join(outdir, "ckpt_final.pt"))

    mean_dt = float(np.mean(timings))
    with open(os.path.join(outdir, "timing.json"), "w") as f:
        json.dump({"iterations": len(timings),
                   "seconds_per_iteration": mean_dt,
                   "hours_for_200k": 200000 * mean_dt / 3600.0,
                   "device": str(device)}, f, indent=2)
    print(f"\nDone. {mean_dt:.4f} s/iteration -> "
          f"{200000 * mean_dt / 3600.0:.2f} h for a full 200k run.")


if __name__ == "__main__":
    train(build_parser().parse_args())
