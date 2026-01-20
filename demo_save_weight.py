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

def save_init_weights_pth(model, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(sd, out_path)
    print(f"[INFO] Saved init weights to: {out_path}")


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
    def _shape(x):
        try:
            return tuple(x.shape)
        except Exception:
            return None

    def _summarize_keysets(model_sd, ckpt_sd, max_items=80):
        mkeys = list(model_sd.keys())
        ckeys = list(ckpt_sd.keys())
        mset, cset = set(mkeys), set(ckeys)

        only_in_model = sorted(mset - cset)
        only_in_ckpt  = sorted(cset - mset)
        in_both = sorted(mset & cset)

        shape_mismatch = []
        dtype_mismatch = []
        for k in in_both:
            ms = _shape(model_sd[k])
            cs = _shape(ckpt_sd[k])
            if (ms is not None) and (cs is not None) and (ms != cs):
                shape_mismatch.append((k, ms, cs))
            else:
                # shape is same; check dtype mismatch (optional)
                try:
                    if model_sd[k].dtype != ckpt_sd[k].dtype:
                        dtype_mismatch.append((k, str(model_sd[k].dtype), str(ckpt_sd[k].dtype)))
                except Exception:
                    pass

        print("\n[CKPT DEBUG] ===== state_dict comparison =====")
        print(f"[CKPT DEBUG] model tensors : {len(mkeys)}")
        print(f"[CKPT DEBUG] ckpt  tensors : {len(ckeys)}")
        print(f"[CKPT DEBUG] only_in_model : {len(only_in_model)}")
        print(f"[CKPT DEBUG] only_in_ckpt  : {len(only_in_ckpt)}")
        print(f"[CKPT DEBUG] in_both        : {len(in_both)}")
        print(f"[CKPT DEBUG] shape_mismatch : {len(shape_mismatch)}")
        print(f"[CKPT DEBUG] dtype_mismatch : {len(dtype_mismatch)}")

        def _print_list(title, items):
            if not items:
                return
            print(f"\n[CKPT DEBUG] {title} (showing up to {max_items}):")
            for k in items[:max_items]:
                print("  ", k)
            if len(items) > max_items:
                print(f"  ... ({len(items) - max_items} more)")

        _print_list("Keys only in MODEL", only_in_model)
        _print_list("Keys only in CKPT", only_in_ckpt)

        if shape_mismatch:
            print(f"\n[CKPT DEBUG] Shape mismatches (showing up to {max_items}):")
            for k, ms, cs in shape_mismatch[:max_items]:
                print(f"  {k}: model{ms} vs ckpt{cs}")
            if len(shape_mismatch) > max_items:
                print(f"  ... ({len(shape_mismatch) - max_items} more)")

        if dtype_mismatch:
            print(f"\n[CKPT DEBUG] DType mismatches (showing up to {max_items}):")
            for k, md, cd in dtype_mismatch[:max_items]:
                print(f"  {k}: model{md} vs ckpt{cd}")
            if len(dtype_mismatch) > max_items:
                print(f"  ... ({len(dtype_mismatch) - max_items} more)")

        print("[CKPT DEBUG] =================================\n")

        return only_in_model, only_in_ckpt, shape_mismatch, dtype_mismatch


    def _strip_known_prefixes(sd):
        # あなたの環境で起きがちな prefix をまとめて吸収（必要に応じて追加）
        prefixes = ["model.", "module.","network."]
        # まず、どれか prefix が多数を占めてるかを見て “一括” で剥がす
        for p in prefixes:
            hit = sum(1 for k in sd.keys() if k.startswith(p))
            if hit > 0 and hit >= 0.7 * len(sd):  # 7割以上同じprefixなら剥がす
                sd = {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
        # 残りは個別に "module." だけ剥がすなど
        out = {}
        for k, v in sd.items():
            if k.startswith("module."):
                k = k[len("module."):]
            out[k] = v
        return out


    if args.checkpoint is not None:
        print(f"[INFO] Loading checkpoint: {args.checkpoint}")

        if args.checkpoint.endswith(".safetensors"):
            state_dict = load_file(args.checkpoint, device="cpu")
        else:
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        # prefix の剥がし（必要ならここにあなたの remap も足す）
        state_dict = _strip_known_prefixes(state_dict)

        # まず“ロードせずに”不一致を表示
        model_sd = model.state_dict()
        only_in_model, only_in_ckpt, shape_mismatch, dtype_mismatch = _summarize_keysets(
            model_sd, state_dict, max_items=120
        )

        # strict を付けた場合は、ここで止める（デバッグ用途）
        if args.strict_ckpt:
            # 「一致してるものだけでOK」ではなく、完全一致を要求して止める
            raise RuntimeError(
                "strict_ckpt is set. See [CKPT DEBUG] mismatch report above."
            )

        # ここから先は “入るところだけ入れる” 実ロード（shape mismatch 回避のためフィルタ）
        SKIP_PREFIXES = (
            "depth_tokenizer.spnet.",
            "backbone.depth_fusion_blk."
        )
        filtered = {}
        skipped_shape = []
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in SKIP_PREFIXES):
                continue
            if k in model_sd:
                if _shape(v) == _shape(model_sd[k]):
                    filtered[k] = v
                else:
                    skipped_shape.append((k, _shape(model_sd[k]), _shape(v)))

        missing, unexpected = model.load_state_dict(filtered, strict=False)

        print(f"[INFO] Loaded tensors: {len(filtered)} / {len(state_dict)}")
        print(f"[INFO] missing={len(missing)}, unexpected={len(unexpected)}")
        print(f"[INFO] skipped_shape_mismatch={len(skipped_shape)}")

        if skipped_shape:
            print("\n[INFO] Skipped due to shape mismatch (showing up to 80):")
            for k, ms, cs in skipped_shape[:80]:
                print(f"  {k}: model{ms} vs ckpt{cs}")
            if len(skipped_shape) > 80:
                print(f"  ... ({len(skipped_shape) - 80} more)")

    if args.ckpt_spnet is not None:
        print(f"[INFO] Loading checkpoint: {args.ckpt_spnet}")
        
        if args.ckpt_spnet.endswith(".safetensors"):
            sd_sp = load_file(args.ckpt_spnet, device="cpu")
        else:
            ckpt = torch.load(args.ckpt_spnet, map_location="cpu", weights_only=True)
            sd_sp = ckpt["network"] if "network" in ckpt else ckpt

        # remove prefix
        sd_sp = _strip_known_prefixes(sd_sp)

        # まず“ロードせずに”不一致を表示
        model_sd = model.depth_tokenizer.spnet.state_dict()
        only_in_model, only_in_ckpt, shape_mismatch, dtype_mismatch = _summarize_keysets(
            model_sd, sd_sp, max_items=120
        )

        # ここから先は “入るところだけ入れる” 実ロード（shape mismatch 回避のためフィルタ）
        filtered = {}
        skipped_shape = []
        for k, v in sd_sp.items():
            if k in model_sd:
                if _shape(v) == _shape(model_sd[k]):
                    filtered[k] = v
                else:
                    skipped_shape.append((k, _shape(model_sd[k]), _shape(v)))

        missing, unexpected = model.depth_tokenizer.spnet.load_state_dict(filtered, strict=False)

        print(f"[INFO] Loaded tensors: {len(filtered)} / {len(sd_sp)}")
        print(f"[INFO] missing={len(missing)}, unexpected={len(unexpected)}")
        print(f"[INFO] skipped_shape_mismatch={len(skipped_shape)}")

        if skipped_shape:
            print("\n[INFO] Skipped due to shape mismatch (showing up to 80):")
            for k, ms, cs in skipped_shape[:80]:
                print(f"  {k}: model{ms} vs ckpt{cs}")
            if len(skipped_shape) > 80:
                print(f"  ... ({len(skipped_shape) - 80} more)")

    if True:
        print("[INFO] SAVE LOADED WEIGHTS")
        save_init_weights_pth(model, "checkpoints/mg3/mg3_base_init.pth")

    # -----------------------------------------------------
    # 4) Load and preprocess images
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

    # -----------------------------------------------------
    # 5) Inference
    # -----------------------------------------------------
    print("[INFO] Running inference...")
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        outputs = model(rgb_torch)

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
    parser.add_argument("--config", type=str, required=True, help="DA3 model YAML (e.g. da3_large.yaml)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path (.safetensors)")
    parser.add_argument("--ckpt_spnet", type=str, default=None, help="SPNet checkpoint path (.pth)")
    parser.add_argument("--outdir", type=str, default="output", help="Output directory")
    parser.add_argument("--resolution", type=int, default=518, help="Inference resolution")
    parser.add_argument(
        "--strict_ckpt",
        action="store_true",
        help="If set, enforce strict checkpoint loading (error on mismatch).",
    )

    args = parser.parse_args()
    run_demo(args)