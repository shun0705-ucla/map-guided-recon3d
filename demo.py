import os
import argparse
from typing import Any, Dict,Tuple, Optional

import numpy as np
import torch
from safetensors.torch import load_file
from PIL import Image
import cv2
import matplotlib

from omegaconf import OmegaConf

from src.recon3d import MapGuidedRecon3D
from depth_anything_3.model.da3 import DepthAnything3Net
from src.fusion_encoder import FusionEncoder
from depth_anything_3.model.dualdpt import DualDPT
from depth_anything_3.model.cam_enc import CameraEnc
from depth_anything_3.model.cam_dec import CameraDec
from src.depth_tokenizer import DepthTokenizer


# ---------------------------------------------------------
# Utility: save depth (same spirit as your Xvader demo)
# ---------------------------------------------------------
def colorize_and_save_depth(depth: np.ndarray, out_png: str, out_npy: str):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    np.save(out_npy, depth)

    dmin, dmax = float(depth.min()), float(depth.max())
    depth_norm = (depth - dmin) / (dmax - dmin + 1e-6)

    cmap = matplotlib.colormaps.get_cmap("Spectral_r")
    depth_color = (cmap(depth_norm)[..., :3] * 255).astype(np.uint8)

    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    cv2.imwrite(out_png, depth_color[..., ::-1])  # RGB → BGR

def depth_rgb_to_colored_point_cloud(
    depth: np.ndarray,       # (H,W)
    K: np.ndarray,           # (3,3)
    rgb: np.ndarray,         # (H,W,3) uint8
    extrinsic: Optional[np.ndarray] = None,  # (3,4) or (4,4) cam->world or world->cam (see note)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unproject depth to a colored point cloud using pinhole intrinsics K.
    Returns points_flat (N,3) and colors_flat (N,3).
    """
    assert depth.ndim == 2
    H, W = depth.shape

    depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx = float(K[0, 2]); cy = float(K[1, 2])

    # pixel grid
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)  # (H,W)

    Z = depth
    X = (uu - cx) * Z / (fx + 1e-12)
    Y = (vv - cy) * Z / (fy + 1e-12)

    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)
    cols = rgb.reshape(-1, 3).astype(np.uint8)

    # Optional: apply extrinsic transform
    # NOTE: DA3 extrinsics convention may be cam->world or world->cam.
    # If your point cloud looks "wrong", we can swap/invert.
    if extrinsic is not None:
        ext = extrinsic.astype(np.float32)
        if ext.shape == (3, 4):
            R = ext[:, :3]
            t = ext[:, 3]
            pts = (pts @ R.T) + t[None, :]
        elif ext.shape == (4, 4):
            R = ext[:3, :3]
            t = ext[:3, 3]
            pts = (pts @ R.T) + t[None, :]
        else:
            raise ValueError(f"extrinsic must be (3,4) or (4,4), got {ext.shape}")

    return pts, cols


def save_pointcloud_pcd_xyzrgb(points: np.ndarray,
                               colors: np.ndarray,
                               out_path_pcd: str):
    """
    Save colored point cloud in ASCII PCD format using packed float 'rgb'.
      points: (N, 3) float32 (x,y,z)
      colors: (N, 3) uint8  (r,g,b)
    """
    import os
    os.makedirs(os.path.dirname(out_path_pcd), exist_ok=True)

    assert points.shape[0] == colors.shape[0], \
        "points and colors must have same length"
    N = points.shape[0]

    pts = points.astype(np.float32)
    cols = colors.astype(np.uint8)

    # Pack RGB into one float32 (PCL convention)
    rgb_uint32 = (
        (cols[:, 0].astype(np.uint32) << 16) |  # R
        (cols[:, 1].astype(np.uint32) << 8)  |  # G
        (cols[:, 2].astype(np.uint32))          # B
    )
    rgb_packed = rgb_uint32.view(np.float32)

    header = [
        "VERSION .7",
        "FIELDS x y z rgb",
        "SIZE 4 4 4 4",
        "TYPE F F F F",
        "COUNT 1 1 1 1",
        f"WIDTH {N}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {N}",
        "DATA ascii",
    ]

    with open(out_path_pcd, "w") as f:
        for line in header:
            f.write(line + "\n")
        for p, rgb in zip(pts, rgb_packed):
            f.write(
                f"{float(p[0])} {float(p[1])} {float(p[2])} {rgb}\n"
            )


def _cfg_to_kwargs(section_cfg) -> Dict[str, Any]:
    """
    Convert an OmegaConf section to plain kwargs dict, excluding the special __object__ block.
    """
    if section_cfg is None:
        return {}
    # Convert to plain container (keeps lists/dicts native Python types)
    d = OmegaConf.to_container(section_cfg, resolve=True)
    if not isinstance(d, dict):
        raise ValueError("Config section is not a mapping/dict.")
    d.pop("__object__", None)  # remove DA3 object spec
    return d


# ---------------------------------------------------------
# Main demo
# ---------------------------------------------------------
def run_demo(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("CUDA is required for this demo.")

    # dtype choice (keeps it simple + similar to your earlier demo style)
    dtype = torch.float16
    print(f"[INFO] Using device={device}, dtype={dtype}")

    # -----------------------------------------------------
    # 1) Load config (YAML → DictConfig)
    # -----------------------------------------------------
    cfg = OmegaConf.load(args.config)

    # -----------------------------------------------------
    # 2) Build model modules from config
    # -----------------------------------------------------
    # Expect sections: net, head, cam_enc, cam_dec like da3_large.yaml
    net_kwargs = _cfg_to_kwargs(cfg.get("net"))
    head_kwargs = _cfg_to_kwargs(cfg.get("head"))
    cam_enc_kwargs = _cfg_to_kwargs(cfg.get("cam_enc"))
    cam_dec_kwargs = _cfg_to_kwargs(cfg.get("cam_dec"))
    depth_tokenizer_kwargs = _cfg_to_kwargs(cfg.get("depth_tokenizer"))

    # Instantiate explicitly (plain Python, no factory)
    backbone = FusionEncoder(**net_kwargs)
    head = DualDPT(**head_kwargs)
    cam_enc = CameraEnc(**cam_enc_kwargs)
    cam_dec = CameraDec(**cam_dec_kwargs)
    depth_tokenizer = DepthTokenizer(**depth_tokenizer_kwargs)

    # Assemble your model (you used MapGuidedRecon3D as the top module)
    model = MapGuidedRecon3D(
        net=backbone,
        head=head,
        cam_enc=cam_enc,
        cam_dec=cam_dec,
        depth_tokenizer=depth_tokenizer,
    ).to(device)

    model.eval()

    # -----------------------------------------------------
    # 3) Load pretrained weights (optional)
    # -----------------------------------------------------
    if args.checkpoint is not None:
        print(f"[INFO] Loading checkpoint: {args.checkpoint}")

        if args.checkpoint.endswith(".safetensors"):
            state_dict = load_file(args.checkpoint)
        else:
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        # pick/strip prefixes
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print("[INFO] Missing keys:")
            for k in missing:
                print("  ", k)

    # -----------------------------------------------------
    # 4) Load and preprocess images & depth maps
    # -----------------------------------------------------
    images = []
    for img_path in args.image:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((args.resolution, args.resolution))
        img_np = np.asarray(img).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1)  # (3,H,W)
        images.append(img_t)

    rgb_torch = torch.stack(images, dim=0).unsqueeze(0).to(device)  # (1,S,3,H,W)
    # For coloring later: bring RGB back to uint8 and match depth resolution
    rgb_for_colors = (rgb_torch.detach().cpu().numpy() * 255.0).squeeze(0)  # (S, 3, H, W)
    rgb_for_colors = np.clip(rgb_for_colors, 0, 255).astype(np.uint8)
    rgb_for_colors = np.transpose(rgb_for_colors, (0, 2, 3, 1))   # (S, H, W, 3)

    depths = []
    for depth_path in args.depth:
        depth_np = np.load(depth_path)
        if depth_np.ndim != 2:
            raise ValueError(f"Depth must be (H,W), got {depth_np.shape}")
        # NaN / inf to 0.0
        depth_np = np.nan_to_num(
            depth_np,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)
        # resize（depthは NEAREST or BILINEAR、用途次第）
        depth_t = torch.from_numpy(depth_np)[None, None, ...]  # (1,1,H,W)
        depth_t = torch.nn.functional.interpolate(
            depth_t,
            size=(args.resolution, args.resolution),
            mode="nearest",  # metric depthなら nearest 推奨
        )
        depth_t = depth_t[0, 0]  # (H,W)

        depths.append(depth_t)

    depth_torch = (
        torch.stack(depths, dim=0)      # (S,H,W)
        .unsqueeze(0)                 # (1,S,H,W) <- this shape is expected by trainer.
        #.unsqueeze(2)               # (1,S,1,H,W)
        .to(device)
    )

    # -----------------------------------------------------
    # 5) Inference
    # -----------------------------------------------------
    print("[INFO] Running inference...")
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        outputs = model(rgb_torch, depth_torch)

    print("\n[DEBUG] Model output keys:")
    for k, v in outputs.items():
        if hasattr(v, "shape"):
            print(f"  {k:15s} -> shape={tuple(v.shape)}, dtype={v.dtype}")
        else:
            print(f"  {k:15s} -> type={type(v)}")
    print()

    # Your model returns dict-like output (same convention you used)
    depth = outputs["depth"].detach().float().cpu().numpy()
    conf = outputs["depth_conf"].detach().float().cpu().numpy()
    K = outputs["intrinsics"].detach().float().cpu().numpy()
    #T = outputs["extrinsics"].detach().float().cpu().numpy()
    print("depth.shape:", depth.shape)

    # -----------------------------------------------------
    # 6) Save results
    # -----------------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)

    B, S, H, W = depth.shape
    for b in range(B):
        for s in range(S):
            depth_i = depth[b, s]  # (H, W)
            conf_i = conf[b, s]    # (H, W)
            K_i = K[b, s]          # (3, 3)
            rgb_i = rgb_for_colors[s] # (H, W, 3)

            base = f"demo_b{b}_s{s}"
            out_png = os.path.join(args.outdir, f"{base}_depth.png")
            out_npy = os.path.join(args.outdir, f"{base}_depth.npy")
            out_points_npy = os.path.join(args.outdir, f"{base}_points.npy")
            out_points_pcd = os.path.join(args.outdir, f"{base}_points.pcd")

            # 1) depth + colored depth (unmasked, for visualization)
            colorize_and_save_depth(depth_i, out_png, out_npy)
            print(f"[INFO] Saved: {out_png}, {out_npy}")

            # 2) create colored point cloud using utils (all pixels initially)
            points_flat, colors_flat = depth_rgb_to_colored_point_cloud(
                depth_i, K_i, rgb_i, extrinsic=None
            )   # (N=H*W, 3)

            # 3) build a single boolean mask of length H*W
            N = depth_i.size  # H*W
            valid = np.ones(N, dtype=bool)
            
            # 3-1) sky mask (keep non-sky)
            #if args.mask_sky and sky_session is not None:
            #    sky_keep = get_sky_keep_mask(rgb_i, sky_session)   # (H, W) bool
            #    sky_keep_flat = sky_keep.reshape(-1)               # (N,)
            #    valid &= sky_keep_flat

            # 3-2) confidence threshold
            #if args.conf_threshold > 0.0:
            #    conf_flat = conf.reshape(-1)                       # (N,)
            #    valid &= (conf_flat >= args.conf_threshold)

            # 4) apply the combined mask
            points_flat = points_flat[valid]
            colors_flat = colors_flat[valid]

            # Save raw points (after masking)
            np.save(out_points_npy, points_flat)

            # Save colored PCD
            save_pointcloud_pcd_xyzrgb(points_flat, colors_flat, out_points_pcd)

            print(f"[INFO] Frame s={s}:")
            print(f"       depth  -> {out_npy}, {out_png}")
            print(f"       points -> {out_points_npy}, {out_points_pcd}")

    print(f"[INFO] Done. Results in {args.outdir}")


# ---------------------------------------------------------
# CLI (argparse, like your Xvader demo)
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("DA3 demo (config-driven explicit instantiation)")

    parser.add_argument("--image", type=str, nargs="+", required=True, help="Input image paths")
    parser.add_argument("--depth", type=str, nargs="+", default=None, help="Input depth map paths (optional)")
    parser.add_argument("--config", type=str, required=True, help="DA3 model YAML (e.g. da3_large.yaml)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path (.pt/.pth)")
    parser.add_argument("--outdir", type=str, default="output", help="Output directory")
    parser.add_argument("--resolution", type=int, default=518, help="Inference resolution")
    parser.add_argument(
        "--strict_ckpt",
        action="store_true",
        help="If set, enforce strict checkpoint loading (error on mismatch).",
    )

    args = parser.parse_args()
    run_demo(args)
