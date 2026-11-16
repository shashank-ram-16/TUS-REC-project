"""
patch_uwn.py
------------
Job: apply the two small fixes the official Ultra-Wide-NeRF code needs to run its
cone (mip) mode on 2D frames. The authors' code supports cones only in 3D mode
(full_volume_mode); their own 2D config has use_mip = False.

  1. render_rays_us_mip_combined.py: `num_rays` was only set in 3D mode -> None in 2D.
     Fix: use the batch size, rays_o.shape[0].
  2. nerf_utils.py, create_nerf(): the network input size for integrated positional
     encoding (multires * i_embed, no raw xyz) was only set in 3D mode -> 63 instead of 60.
     Fix: set it whenever use_mip is on.

Nothing else is changed. Safe to run twice (it reports "already patched").
Usage:  python code\\patch_uwn.py external\\ultra-wide-nerf
"""

import sys
from pathlib import Path

FIXES = [
    ("render_rays_us_mip_combined.py",
     """        if convex.config['full_volume_mode']:
            num_rays = rays_o.shape[0]
        # Convex mode: create z_vals for subset of rays""",
     """        # [TUS-REC fix] 2D + mip: num_rays was never set (None); use the batch size
        num_rays = rays_o.shape[0]
        # Convex mode: create z_vals for subset of rays"""),
    ("nerf_utils.py",
     """    if convex.config['use_mip'] and convex.config['full_volume_mode']:
        input_ch = args.multires * args.i_embed""",
     """    # [TUS-REC fix] integrated positional encoding has no raw-xyz term in 2D mode too
    if convex.config['use_mip']:
        input_ch = args.multires * args.i_embed"""),
]


def main():
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else "external/ultra-wide-nerf")
    for fname, old, new in FIXES:
        p = repo / fname
        t = p.read_text(encoding="utf-8")
        if new in t:
            print(f"{fname}: already patched")
        elif t.count(old) == 1:
            p.write_text(t.replace(old, new), encoding="utf-8")
            print(f"{fname}: patched")
        else:
            raise SystemExit(f"{fname}: expected code not found - the repository may have changed. Stop and check.")


if __name__ == "__main__":
    main()
