#!/usr/bin/env python3
"""
Color calibration script for wrist camera images.
Uses the first 5 image pairs from calibration_pairs_wrist to learn a color transform,
then applies it to all images and saves calibrated versions.

This version uses k-means clustering to fit a mixture of affine color transforms,
allowing different transforms for different color regions (e.g., black gripper,
yellow table, background/shadows).
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from skimage import color as skcolor


# Number of clusters for mixture of affine transforms
N_CLUSTERS = 4  # gripper, cube, table, background


def _format_helper_mixture(transforms: list, centroids: np.ndarray, use_lab: bool) -> str:
    """Format mixture of affine transforms for saving."""
    lines = [f"n_clusters: {len(transforms)}"]
    lines.append(f"use_lab: {use_lab}")
    
    # Format centroids
    lines.append("centroids:")
    for i, c in enumerate(centroids):
        c_str = ", ".join(f"{v:.6f}" for v in c)
        lines.append(f"  - [{c_str}]  # cluster {i}")
    
    # Format each transform
    for i, (A, b) in enumerate(transforms):
        lines.append(f"transform_{i}:")
        lines.append("  A:")
        for row in A:
            row_str = ", ".join(f"{v:.6f}" for v in row)
            lines.append(f"    - [{row_str}]")
        b_str = ", ".join(f"{v:.6f}" for v in b)
        lines.append(f"  b: [{b_str}]")
    
    return "\n".join(lines)


def _write_helper_file_mixture(transforms: list, centroids: np.ndarray, use_lab: bool, dest: Path) -> None:
    """Write mixture of affine transforms to file."""
    code = _format_helper_mixture(transforms, centroids, use_lab)
    dest.write_text(code)


def _get_aug(x: np.ndarray, add_ones: bool = True) -> np.ndarray:
    """
    Augment input features for affine regression (linear only, no quadratic).
    
    Args:
        x: Input pixels, shape (N, 3) where each row is [R, G, B]
        add_ones: Whether to add constant term (for bias)
    
    Returns:
        Augmented features, shape (N, 4) if add_ones=True, else (N, 3)
        Each row: [R, G, B, 1] (or without 1 if add_ones=False)
    """
    if add_ones:
        ones = np.ones((x.shape[0], 1), np.float64)
        return np.hstack([x, ones])
    return x


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB (0-1 range) to Lab color space."""
    # skimage expects (H, W, 3) or (N, 3), returns Lab
    # L: 0-100, a: -128 to 127, b: -128 to 127
    if rgb.ndim == 2:
        # (N, 3) -> (N, 1, 3) for skimage
        rgb_3d = rgb.reshape(-1, 1, 3)
        lab_3d = skcolor.rgb2lab(rgb_3d)
        return lab_3d.reshape(-1, 3)
    return skcolor.rgb2lab(rgb)


def _solve_for_cluster(S: np.ndarray, R: np.ndarray) -> tuple:
    """
    Solve for affine color transform for a single cluster.
    
    Args:
        S: Source pixels (GS render colors), shape (N, 3)
        R: Reference pixels (real camera colors), shape (N, 3)
    
    Returns:
        A: Affine transform matrix (3x3)
        b: Bias vector (3,)
        w: IRLS weights (N,)
    """
    if len(S) < 10:
        # Not enough pixels, return identity transform
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), np.ones(len(S), dtype=np.float64)
    
    # Augment source with ones for bias term
    S_aug = _get_aug(S)  # (N, 4)
    
    # Initial L2 solution
    X, *_ = np.linalg.lstsq(S_aug, R, rcond=None)
    if not np.all(np.isfinite(X)):
        print("  Warning: Initial least-squares failed, using identity transform")
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), np.ones(len(S), dtype=np.float64)

    # Robust IRLS with Tukey bi-weight
    max_iter = 50
    c = 4.685
    X_prev = X
    w = np.ones((S.shape[0],), np.float64)

    for n_iter in range(max_iter):
        pred = S_aug @ X_prev
        resid = np.linalg.norm(R - pred, axis=1)
        mad = np.median(np.abs(resid - np.median(resid)))
        mad = max(mad, 1e-6)
        scale = c * 1.4826 * mad
        u = resid / scale
        
        w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)

        if not np.any(w):
            break

        sqrt_w = np.sqrt(w)[:, None]
        X_new, *_ = np.linalg.lstsq(S_aug * sqrt_w, R * sqrt_w, rcond=None)
        if not np.all(np.isfinite(X_new)):
            break

        if np.linalg.norm(X_new - X_prev) < 1e-6:
            X_prev = X_new
            break

        X_prev = X_new

    A = X_prev[:-1, :].T.astype(np.float32)  # (3, 3)
    b = X_prev[-1, :].astype(np.float32)      # (3,)
    return A, b, w


def _match_clusters(centroids_S: np.ndarray, centroids_R: np.ndarray) -> np.ndarray:
    """
    Match clusters from S to R using Hungarian algorithm (minimum cost assignment).
    
    Args:
        centroids_S: Cluster centroids from S, shape (K, 3)
        centroids_R: Cluster centroids from R, shape (K, 3)
    
    Returns:
        mapping: Array where mapping[i] = j means S cluster i matches R cluster j
    """
    from scipy.optimize import linear_sum_assignment
    
    n_clusters = len(centroids_S)
    # Cost matrix: distance between each pair of centroids
    cost = np.zeros((n_clusters, n_clusters), dtype=np.float32)
    for i in range(n_clusters):
        for j in range(n_clusters):
            cost[i, j] = np.linalg.norm(centroids_S[i] - centroids_R[j])
    
    # Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = col_ind  # mapping[i] = j means S cluster i -> R cluster j
    
    return mapping


def _solve_mixture_transforms(S: np.ndarray, R: np.ndarray, n_clusters: int = 4, use_lab: bool = True) -> tuple:
    """
    Cluster S and R independently, match clusters, then fit affine transform per cluster.
    
    Args:
        S: Source pixels (GS render), shape (N, 3), RGB in [0, 1]
        R: Reference pixels (real camera), shape (N, 3), RGB in [0, 1]
        n_clusters: Number of clusters (k-means)
        use_lab: If True, cluster in Lab space; else RGB
    
    Returns:
        transforms: List of (A, b) tuples for each cluster
        centroids: Cluster centroids from S in clustering space (for inference)
        labels_S: Cluster assignment for S pixels
        kmeans_S: Fitted KMeans object for S (for inference)
        use_lab: Whether Lab space was used
        all_weights: IRLS weights for all matched pixel pairs
    """
    print(f"[INFO] Clustering {len(S)} pixels into {n_clusters} clusters (independently for S and R)...")
    
    # Convert to clustering space
    if use_lab:
        S_cluster = _rgb_to_lab(S)
        R_cluster = _rgb_to_lab(R)
        print("  Using Lab color space for clustering")
    else:
        S_cluster = S
        R_cluster = R
        print("  Using RGB color space for clustering")
    
    # Fit k-means on S
    kmeans_S = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_S = kmeans_S.fit_predict(S_cluster)
    centroids_S = kmeans_S.cluster_centers_
    
    # Fit k-means on R
    kmeans_R = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_R = kmeans_R.fit_predict(R_cluster)
    centroids_R = kmeans_R.cluster_centers_
    
    # Match clusters between S and R
    print("  Matching clusters between S and R...")
    cluster_mapping = _match_clusters(centroids_S, centroids_R)
    
    # Print cluster statistics
    print("  S clusters:")
    for k in range(n_clusters):
        mask = labels_S == k
        count = np.sum(mask)
        pct = 100.0 * count / len(S)
        mean_rgb = S[mask].mean(axis=0) if count > 0 else np.zeros(3)
        matched_R = cluster_mapping[k]
        print(f"    Cluster {k} -> R cluster {matched_R}: {count} pixels ({pct:.1f}%), "
              f"mean RGB: [{mean_rgb[0]:.3f}, {mean_rgb[1]:.3f}, {mean_rgb[2]:.3f}]")
    
    print("  R clusters:")
    for k in range(n_clusters):
        mask = labels_R == k
        count = np.sum(mask)
        pct = 100.0 * count / len(R)
        mean_rgb = R[mask].mean(axis=0) if count > 0 else np.zeros(3)
        print(f"    Cluster {k}: {count} pixels ({pct:.1f}%), "
              f"mean RGB: [{mean_rgb[0]:.3f}, {mean_rgb[1]:.3f}, {mean_rgb[2]:.3f}]")
    
    # Fit transform for each matched cluster pair
    transforms = []
    all_weights = np.zeros(len(S), dtype=np.float64)
    
    for k_S in range(n_clusters):
        k_R = cluster_mapping[k_S]  # Matched R cluster
        
        mask_S = labels_S == k_S
        mask_R = labels_R == k_R
        
        S_k = S[mask_S]
        R_k = R[mask_R]
        
        # Sample min(N_S, N_R) pixels from each
        n_S, n_R = len(S_k), len(R_k)
        n_sample = min(n_S, n_R)
        
        print(f"  Fitting transform for S cluster {k_S} <-> R cluster {k_R}:")
        print(f"    S has {n_S} pixels, R has {n_R} pixels, using {n_sample} matched pairs")
        
        if n_sample < 10:
            print(f"    Warning: Too few pixels ({n_sample}), using identity transform")
            transforms.append((np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)))
            continue
        
        # Random sampling to get equal-sized sets
        rng = np.random.default_rng(42)
        idx_S = rng.choice(n_S, size=n_sample, replace=False)
        idx_R = rng.choice(n_R, size=n_sample, replace=False)
        
        S_sampled = S_k[idx_S]
        R_sampled = R_k[idx_R]
        
        A_k, b_k, w_k = _solve_for_cluster(S_sampled, R_sampled)
        transforms.append((A_k, b_k))
        
        # Store weights (only for sampled pixels)
        # Map back to original indices
        orig_idx_S = np.where(mask_S)[0][idx_S]
        all_weights[orig_idx_S] = w_k
        
        print(f"    A:\n{A_k}")
        print(f"    b: {b_k}")
    
    return transforms, centroids_S, labels_S, labels_R, cluster_mapping, kmeans_S, use_lab, all_weights


def _apply_mixture_transform(img: np.ndarray, transforms: list, centroids: np.ndarray, use_lab: bool) -> np.ndarray:
    """
    Apply mixture of affine transforms to an image.
    
    For each pixel:
      1. Assign to nearest cluster centroid
      2. Apply that cluster's affine transform
    
    Args:
        img: Input image, shape (H, W, 3), uint8
        transforms: List of (A, b) tuples for each cluster
        centroids: Cluster centroids in clustering space
        use_lab: Whether centroids are in Lab space
    
    Returns:
        Transformed image, shape (H, W, 3), uint8
    """
    H, W = img.shape[:2]
    flat = img.reshape(-1, 3).astype(np.float32) / 255.0  # (N, 3)
    
    # Convert to clustering space for assignment
    if use_lab:
        flat_cluster = _rgb_to_lab(flat)
    else:
        flat_cluster = flat
    
    # Assign each pixel to nearest centroid
    # Compute distances to all centroids
    n_clusters = len(centroids)
    dists = np.zeros((len(flat), n_clusters), dtype=np.float32)
    for k in range(n_clusters):
        dists[:, k] = np.linalg.norm(flat_cluster - centroids[k], axis=1)
    
    labels = np.argmin(dists, axis=1)
    
    # Apply transform per cluster
    out = np.zeros_like(flat)
    for k in range(n_clusters):
        mask = labels == k
        if not np.any(mask):
            continue
        A_k, b_k = transforms[k]
        out[mask] = flat[mask] @ A_k.T + b_k
    
    out = np.clip(out, 0.0, 1.0)
    return (out.reshape(H, W, 3) * 255.0).astype(np.uint8)


def main():
    # Paths
    base_dir = Path("/home/tongmiao/Documents/lerobot_pi05/calibration_pairs_wrist")
    gs_dir = base_dir / "gs_renders"
    real_dir = base_dir / "real_captures"
    out_dir = base_dir / "calibrated"  # Save calibrated images here
    
    # Get first 5 image pairs (frame_000000, frame_000005, frame_000010, frame_000015, frame_000020)
    frame_indices = [0, 5, 10, 15, 20]
    src_img_paths = [gs_dir / f"frame_{i:04d}.png" for i in frame_indices]
    ref_img_paths = [real_dir / f"frame_{i:04d}.png" for i in frame_indices]
    
    # Verify files exist
    for src_path, ref_path in zip(src_img_paths, ref_img_paths):
        if not src_path.exists():
            raise FileNotFoundError(f"Source image not found: {src_path}")
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference image not found: {ref_path}")
    
    print(f"[INFO] Using {len(src_img_paths)} image pairs for calibration:")
    for src_path, ref_path in zip(src_img_paths, ref_img_paths):
        print(f"  {src_path.name} <-> {ref_path.name}")
    
    # Load images and collect pixels
    pixel_src = []
    pixel_ref = []
    image_height = None
    image_width = None
    
    for src_img_path, ref_img_path in zip(src_img_paths, ref_img_paths):
        src_img = np.array(Image.open(src_img_path).convert("RGB")).astype(np.float32) / 255.0
        ref_img = np.array(Image.open(ref_img_path).convert("RGB")).astype(np.float32) / 255.0
        
        assert src_img.shape == ref_img.shape, f"Source and reference images must have the same shape: {src_img.shape} vs {ref_img.shape}"
        
        if image_height is None:
            image_height, image_width = src_img.shape[:2]
        
        pixel_src.append(src_img.reshape(-1, 3))
        pixel_ref.append(ref_img.reshape(-1, 3))
    
    pixel_src = np.concatenate(pixel_src, axis=0)
    pixel_ref = np.concatenate(pixel_ref, axis=0)
    
    print(f"[INFO] Collected {pixel_src.shape[0]} pixels from {len(src_img_paths)} images")
    print(f"[INFO] Image dimensions: {image_height}x{image_width}")
    
    # Solve for mixture of affine color transforms
    print("[INFO] Solving for mixture of affine color transforms...")
    use_lab = True  # Use Lab space for clustering (better perceptual separation)
    transforms, centroids, labels_S, labels_R, cluster_mapping, kmeans, use_lab, weights = _solve_mixture_transforms(
        pixel_src, pixel_ref, n_clusters=N_CLUSTERS, use_lab=use_lab
    )
    
    # Save transform parameters
    os.makedirs(out_dir, exist_ok=True)
    transform_file = out_dir / "color_mapping_mixture.yaml"
    _write_helper_file_mixture(transforms, centroids, use_lab, transform_file)
    print(f"[INFO] Saved mixture color transform to {transform_file}")
    
    # Visualize cluster assignments and weights
    n_images = len(src_img_paths)
    labels_S_full = labels_S.reshape(n_images, image_height, image_width)
    labels_R_full = labels_R.reshape(n_images, image_height, image_width)
    weights_full = weights.reshape(n_images, image_height, image_width)
    
    # Create colormap for cluster visualization
    # Use same colors for matched clusters
    # Expected clusters: gripper (black/dark), cube (purple), table (yellow), background
    cluster_colors = np.array([
        [255, 0, 0],     # Red for cluster 0 (e.g., gripper)
        [0, 255, 0],     # Green for cluster 1 (e.g., cube)
        [0, 0, 255],     # Blue for cluster 2 (e.g., table)
        [255, 255, 0],   # Yellow for cluster 3 (e.g., background)
        [255, 0, 255],   # Magenta for cluster 4 (extra)
    ], dtype=np.uint8)
    
    # Create inverse mapping: R cluster -> S cluster (for consistent coloring)
    inverse_mapping = np.zeros(N_CLUSTERS, dtype=np.int32)
    for k_S, k_R in enumerate(cluster_mapping):
        inverse_mapping[k_R] = k_S
    
    for i in range(n_images):
        # S cluster visualization (original labels)
        cluster_vis_S = cluster_colors[labels_S_full[i]]
        Image.fromarray(cluster_vis_S).save(out_dir / f"clusters_S_{i:04d}.png")
        
        # R cluster visualization (mapped to S colors for consistency)
        # Map R labels to S cluster indices for consistent coloring
        labels_R_mapped = inverse_mapping[labels_R_full[i]]
        cluster_vis_R = cluster_colors[labels_R_mapped]
        Image.fromarray(cluster_vis_R).save(out_dir / f"clusters_R_{i:04d}.png")
        
        # Weight visualization
        w_vis = (weights_full[i] * 255.0).astype(np.uint8)
        Image.fromarray(w_vis).save(out_dir / f"weights_{i:04d}.png")
        
        # Weight mask (rejected pixels)
        w_mask = (weights_full[i] < 0.01).astype(np.uint8) * 255
        Image.fromarray(w_mask).save(out_dir / f"weights_mask_{i:04d}.png")
    
    # Apply transform to calibration image pairs
    print("[INFO] Applying mixture transform to calibration image pairs...")
    for src_img_path, ref_img_path in zip(src_img_paths, ref_img_paths):
        src_img = np.array(Image.open(src_img_path).convert("RGB"))
        corr_img = _apply_mixture_transform(src_img, transforms, centroids, use_lab)
        
        # Save calibrated image
        calibrated_path = out_dir / src_img_path.name
        Image.fromarray(corr_img).save(calibrated_path, quality=95)
        
        # Create comparison image (src | ref | calibrated)
        ref_img = np.array(Image.open(ref_img_path).convert("RGB"))
        combined_img = np.hstack((src_img, ref_img, corr_img))
        combined_path = out_dir / f"combined_{src_img_path.name}"
        Image.fromarray(combined_img).save(combined_path, quality=95)
        
        print(f"  Saved calibrated: {calibrated_path.name}")
    
    print(f"[INFO] Calibrated {len(src_img_paths)} image pairs")
    print(f"[INFO] All calibrated images saved to: {out_dir}")


if __name__ == "__main__":
    main()

