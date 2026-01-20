import argparse
import numpy as np
import matplotlib.pyplot as plt

def visualize_depth(depth_path, save_path=None):
    depth = np.load(depth_path)

    # Safety: handle NaN / Inf
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    dmin, dmax = depth.min(), depth.max()
    if abs(dmax - dmin) < 1e-12:
        depth_norm = np.zeros_like(depth)
    else:
        depth_norm = (depth - dmin) / (dmax - dmin)

    plt.figure(figsize=(6, 6))
    plt.imshow(depth_norm, cmap="Spectral_r")
    plt.colorbar(label="Normalized depth")
    plt.title(f"Depth visualization\n{depth_path}")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"[INFO] Saved visualization to {save_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize depth.npy")
    parser.add_argument("depth", type=str, help="Path to depth.npy")
    parser.add_argument("--save", type=str, default=None, help="Optional output PNG path")
    args = parser.parse_args()

    visualize_depth(args.depth, args.save)

# python visualize_depth.py output/demo_depth.npy
