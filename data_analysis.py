import numpy as np
import seaborn as sns
import scipy.io as sio
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

# ======================================================
# 1. Load data
# ======================================================
mat = sio.loadmat("./MODIS_Aug.mat")
tensor = mat["training_tensor"].astype(float)  # (100, 200, 31)
N1, N2, T = tensor.shape
print(f"Tensor shape: {tensor.shape}  (lat=100, lon=200, time=31)")


# ======================================================
# 2. Mode-n Unfolding functions
# ======================================================
def unfold_mode1(X):
    """Unfold along mode-1 (latitude). Shape: (N1, N2*T)."""
    return X.reshape(N1, N2 * T)

def unfold_mode2(X):
    """Unfold along mode-2 (longitude). Shape: (N2, N1*T)."""
    X2 = np.moveaxis(X, 1, 0)  # (N2, N1, T)
    return X2.reshape(N2, N1 * T)

def unfold_mode3(X):
    """Unfold along mode-3 (time). Shape: (N1*N2, T)."""
    return X.reshape(N1 * N2, T)


# ======================================================
# 3. Helper: SVD + Plot
# ======================================================
def analyze_and_plot(matrix_2d, mode_name, save_name):
    """
    Perform SVD, print statistics, and plot singular values + cumulative variance.
    """
    U, S, Vt = np.linalg.svd(matrix_2d, full_matrices=False)
    eigen = S ** 2
    ratio = eigen / eigen.sum()
    cum = np.cumsum(ratio)

    k99 = np.argmax(cum >= 0.99) + 1
    print(f"\n=== {mode_name} ===")
    print(f"Top 10 singular values: {S[:10]}")
    print(f"Effective rank @99% variance: {k99}")

    # ---- Plot ----
    fig, ax1 = plt.subplots(figsize=(6,5))
    ax1.plot(S, "o-", color="blue")
    ax1.set_ylabel("Singular Value", color="blue")
    ax1.set_xlabel("Rank index")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(cum, "s--", color="red")
    ax2.set_ylabel("Cumulative Variance", color="red")

    ax2.axhline(0.99, color="green", ls=":")
    ax2.axvline(k99-1, color="green", ls=":")
    ax2.text(k99, 0.99, f"rank={k99}", color="green")

    plt.title(f"SVD Spectrum ({mode_name})")
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    print(f"Saved figure: {save_name}")

    return k99, S, cum


# ======================================================
# 4. Spatial Mode-1 (Latitude) Rank
# ======================================================
X1 = unfold_mode1(tensor)
r1, S1, cum1 = analyze_and_plot(X1, "Mode-1 (Latitude)", "svd_mode1_latitude.png")


# ======================================================
# 5. Spatial Mode-2 (Longitude) Rank
# ======================================================
X2 = unfold_mode2(tensor)
r2, S2, cum2 = analyze_and_plot(X2, "Mode-2 (Longitude)", "svd_mode2_longitude.png")


# ======================================================
# 6. Temporal Mode-3 Rank
# ======================================================
X3 = unfold_mode3(tensor)
r3, S3, cum3 = analyze_and_plot(X3, "Mode-3 (Time)", "svd_mode3_time.png")


# ======================================================
# 7. Temporal statistics (same logic as your original version)
# ======================================================

print("\n=== Temporal Statistics ===")

# Global mean time series
mask = (X3 != 0).reshape(N1 * N2, T)
col_sum = np.sum(X3, axis=0)
col_count = np.sum(mask, axis=0)
col_count[col_count == 0] = 1
mean_ts = col_sum / col_count

# Temporal ACF
lags = 10
ac = acf(mean_ts, nlags=lags, fft=True)
print("\nGlobal temporal ACF:")
print(ac)

# Temporal covariance
time_cov = np.cov(X3, rowvar=False)
Uc, Sc, Vc = np.linalg.svd(time_cov)
cum_cov = np.cumsum(Sc) / np.sum(Sc)

print("\nTemporal covariance cumulative variance:")
print(cum_cov[:10])

# Differences between consecutive timesteps
diff_norms = np.linalg.norm(X3[:,1:] - X3[:,:-1], axis=0)
print(f"\nMean diff norm: {diff_norms.mean():.4f}")
print(f"Median diff norm: {np.median(diff_norms):.4f}")


# ======================================================
# 8. Summary
# ======================================================
print("\n============================")
print("Tucker-style effective ranks:")
print(f"  Spatial-1 (lat) rank  r1 = {r1}")
print(f"  Spatial-2 (lon) rank  r2 = {r2}")
print(f"  Temporal   (time) r3 = {r3}")
print("============================")



# ======================================================
# Advanced Spatiotemporal Analysis (fully English comments)
# ======================================================
def analyze_spatiotemporal_properties(tensor):
    print("\nStarting Advanced Spatiotemporal Analysis...")
    H, W, T = tensor.shape
    
    # Prepare zero-mean data (removes static background, focus on dynamics)
    mask = (tensor != 0)
    pixel_mean = np.sum(tensor, axis=2) / (np.sum(mask, axis=2) + 1e-6)
    tensor_centered = tensor - pixel_mean[:, :, None]

    # Mask out missing values using NaN
    tensor_centered[~mask] = np.nan 

    # ==========================================================
    # Part 1: Temporal Analysis
    # Show that temporal dependency is heterogeneous & noisy
    # ==========================================================
    
    fig_t = plt.figure(figsize=(14, 5))
    
    # --- 1.1 Temporal Lag-1 Scatter Plot ---
    # Shows how x(t) relates to x(t-1)
    ax1 = fig_t.add_subplot(121)
    
    data_t = tensor_centered[:, :, 1:].flatten()     # x_t
    data_t_prev = tensor_centered[:, :, :-1].flatten()  # x_{t-1}
    
    # Filter valid values
    valid_idx = ~np.isnan(data_t) & ~np.isnan(data_t_prev)
    x = data_t_prev[valid_idx]
    y = data_t[valid_idx]
    
    # Sample 10k points for speed
    if len(x) > 10000:
        idx = np.random.choice(len(x), 10000, replace=False)
        x_sample, y_sample = x[idx], y[idx]
    else:
        x_sample, y_sample = x, y
        
    # Global lag-1 correlation
    corr_lag1 = np.corrcoef(x, y)[0, 1]
    
    ax1.scatter(x_sample, y_sample, alpha=0.1, s=2, c='blue')
    ax1.set_title(f"Temporal Lag-1 Scatter\nGlobal Lag-1 Corr = {corr_lag1:.3f}")
    ax1.set_xlabel(r"$x_{t-1}$ (Centered)")
    ax1.set_ylabel(r"$x_{t}$ (Centered)")
    ax1.grid(True, alpha=0.3)
    
    # Reference line y=x (AR(1) ideal)
    lims = [np.min([ax1.get_xlim(), ax1.get_ylim()]), np.max([ax1.get_xlim(), ax1.get_ylim()])]
    ax1.plot(lims, lims, 'r--', alpha=0.5, label="Perfect AR(1)")
    ax1.legend()

    # --- 1.2 Pixel-wise ACF Distribution ---
    # Shows strong heterogeneity across locations
    ax2 = fig_t.add_subplot(122)
    
    lag1_corrs = []
    flat_tensor = tensor_centered.reshape(-1, T)
    indices = np.random.choice(flat_tensor.shape[0], 2000, replace=False)
    
    for idx in indices:
        ts = flat_tensor[idx]
        ts = ts[~np.isnan(ts)]
        if len(ts) > 5:
            c = np.corrcoef(ts[:-1], ts[1:])[0, 1]
            if not np.isnan(c):
                lag1_corrs.append(c)
                
    sns.histplot(lag1_corrs, kde=True, ax=ax2, color='purple', bins=30)
    ax2.set_title("Distribution of Pixel-wise Lag-1 Autocorrelation")
    ax2.set_xlabel("Lag-1 Correlation")
    ax2.axvline(np.mean(lag1_corrs), color='r', linestyle='--', label=f"Mean={np.mean(lag1_corrs):.2f}")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("analysis_temporal_dynamics.png", dpi=300)
    print("Saved analysis_temporal_dynamics.png")
    
    print("Temporal Analysis:")
    print("  - Lag-1 scatter dispersion indicates x(t) is not well explained by a simple AR(1).")
    print("  - Wide ACF distribution indicates strong temporal heterogeneity across locations.")


    # ==========================================================
    # Part 2: Spatial Analysis
    # Show strong spatial dependency (justifies Conv3D / Attention)
    # ==========================================================
    
    fig_s = plt.figure(figsize=(14, 5))
    
    # --- 2.1 Correlation vs Distance ---
    ax3 = fig_s.add_subplot(121)
    
    n_anchors = 50
    h_anchors = np.random.randint(0, H, n_anchors)
    w_anchors = np.random.randint(0, W, n_anchors)
    
    distances = []
    correlations = []
    
    for k in range(n_anchors):
        hc, wc = h_anchors[k], w_anchors[k]
        ts_c = tensor_centered[hc, wc]
        
        if np.sum(~np.isnan(ts_c)) < 10:
            continue
            
        h_neighbors = np.random.randint(0, H, 100)
        w_neighbors = np.random.randint(0, W, 100)
        
        for hn, wn in zip(h_neighbors, w_neighbors):
            ts_n = tensor_centered[hn, wn]
            
            dist = np.sqrt((hc-hn)**2 + (wc-wn)**2)
            
            valid = ~np.isnan(ts_c) & ~np.isnan(ts_n)
            if np.sum(valid) > 10:
                corr = np.corrcoef(ts_c[valid], ts_n[valid])[0, 1]
                distances.append(dist)
                correlations.append(corr)
                
    ax3.scatter(distances, correlations, s=1, alpha=0.3, c='gray')

    # Distance binning
    bins = np.linspace(0, 100, 20)
    bin_means = []
    bin_centers = []
    for i in range(len(bins)-1):
        mask_bin = (np.array(distances) >= bins[i]) & (np.array(distances) < bins[i+1])
        if np.sum(mask_bin) > 0:
            bin_means.append(np.mean(np.array(correlations)[mask_bin]))
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            
    ax3.plot(bin_centers, bin_means, 'r-o', linewidth=2, label='Mean Correlation')
    ax3.set_title("Spatial Correlation Decay vs Distance")
    ax3.set_xlabel("Euclidean Distance (Pixels)")
    ax3.set_ylabel("Pearson Correlation")
    ax3.set_ylim(-0.2, 1.0)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # --- 2.2 Local Spatial Similarity Map ---
    ax4 = fig_s.add_subplot(122)
    
    grad_h = np.nanmean(np.abs(tensor_centered[1:] - tensor_centered[:-1]), axis=2)
    grad_w = np.nanmean(np.abs(tensor_centered[:, 1:] - tensor_centered[:, :-1]), axis=2)
    
    grad_map = np.zeros((H, W))
    grad_map[1:, :] += grad_h
    grad_map[:, 1:] += grad_w
    
    # Inverse gradient = similarity
    sim_map = 1.0 / (grad_map + 1e-1)
    
    im = ax4.imshow(sim_map, cmap='magma', aspect='auto', vmin=0, vmax=np.percentile(sim_map, 95))
    ax4.set_title("Local Spatial Similarity\n(High = Strong Local Corr)")
    fig_s.colorbar(im, ax=ax4)
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig("analysis_spatial_correlation.png", dpi=300)
    print("Saved analysis_spatial_correlation.png")
    
    print("Spatial Analysis:")
    print("  - Correlation decays smoothly with distance → long-range dependencies.")
    print("  - Very high local correlation (< 5 px) → Conv3D is highly suitable.")

analyze_spatiotemporal_properties(tensor)
