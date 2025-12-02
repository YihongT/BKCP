import matplotlib.pyplot as plt
import numpy as np
import os
import math
import copy
from matplotlib.gridspec import GridSpec


def _save_fig(fig, save_path):
    """
    Save figure without using bbox_inches='tight' to preserve square layout.
    The rect argument keeps space for the title.
    """
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved square plot to {save_path}")


def visualize_temporal_mean(tensor_data, title, save_path, cmap='plasma'):
    """
    Compute and visualize the temporal mean map (average along time dimension).
    Useful as a static background map of the spatiotemporal field.
    """
    # 1. Compute temporal mean (ignore zeros by turning them into NaN first)
    data_for_mean = tensor_data.copy()
    data_for_mean[data_for_mean == 0] = np.nan
    
    mean_map = np.nanmean(data_for_mean, axis=2)

    # 2. Plot (force square shape)
    fig, ax = plt.subplots(figsize=(10, 10))

    # Use original orientation (or add np.rot90 if matching earlier perspective)
    im = ax.imshow(mean_map, cmap=cmap, aspect='auto')

    ax.set_title(title, fontsize=16)
    ax.axis('off')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Mean Value')

    _save_fig(fig, save_path)


def visualize_3d_slices(tensor_data, title, save_path,
                        cmap='plasma', mask_zeros=False,
                        vmin=None, vmax=None):
    """
    Visualize all time slices of a spatiotemporal tensor.
    mask_zeros: display zero-valued pixels as white (emphasize missing data)
    """
    T = tensor_data.shape[2]
    n_cols = math.ceil(math.sqrt(T))
    n_rows = math.ceil(T / n_cols)

    # ============ 1. GridSpec layout with an extra column for colorbar ============
    panel_size = 2.2  # tune this for desired panel size

    fig_w = panel_size * n_cols + 1.0  # +1 for colorbar column
    fig_h = panel_size * n_rows

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(title, fontsize=16)

    gs = GridSpec(
        n_rows, n_cols + 1,
        figure=fig,
        width_ratios=[1] * n_cols + [0.2],  # last narrow panel for colorbar
        wspace=0.2,
        hspace=0.25
    )

    # Allocate subplot axes (first n_cols columns)
    axes = []
    for i in range(n_rows):
        for j in range(n_cols):
            axes.append(fig.add_subplot(gs[i, j]))
    axes_flat = np.array(axes)

    # Last column merged vertically for colorbar
    cax = fig.add_subplot(gs[:, -1])

    # ============ 2. Preprocess data ============
    plot_data = tensor_data
    if mask_zeros:
        plot_data = np.ma.masked_where(tensor_data == 0, tensor_data)
        current_cmap = copy.copy(plt.get_cmap(cmap))
        current_cmap.set_bad(color='white')
    else:
        current_cmap = cmap

    # Determine vmin/vmax dynamically
    if vmin is None:
        valid = tensor_data[tensor_data != 0] if mask_zeros else tensor_data
        vmin = np.nanmin(valid)
    if vmax is None:
        valid = tensor_data[tensor_data != 0] if mask_zeros else tensor_data
        vmax = np.nanmax(valid)

    # ============ 3. Plot each time slice ============
    im = None
    for t in range(n_rows * n_cols):
        ax = axes_flat[t]
        if t < T:
            slice_data = plot_data[:, :, t]
            im = ax.imshow(slice_data, cmap=current_cmap,
                           vmin=vmin, vmax=vmax, aspect='auto')
            ax.set_title(f"t={t}", fontsize=12)
        ax.axis('off')

    # ============ 4. Right-side colorbar (independent of subplots) ============
    if im is not None:
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.ax.tick_params(labelsize=13)

    _save_fig(fig, save_path)


def visualize_2d_flattened(tensor_data, title, save_path, cmap='plasma'):
    """
    Visualize the tensor flattened to (Space, Time).  
    Output is square by stretching the time axis (aspect='auto').
    """
    dim1, dim2, dim3 = tensor_data.shape
    flattened_data = tensor_data.reshape(dim1 * dim2, dim3)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(flattened_data, cmap=cmap, aspect='auto')

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Spatial Locations")

    fig.colorbar(im, ax=ax, shrink=0.8)
    _save_fig(fig, save_path)


def run_visualization(tensor_data, model_name, suffix, RESULTS_DIR,
                      is_std=False, mask_zeros=False, custom_range=None):
    """
    Run 3 visualization modes and save outputs.
    Automatically selects colormap depending on prediction/statistic type.
    """
    # Choose colormap
    cmap = 'viridis' if is_std else 'plasma'
    title_base = f"{model_name} {suffix}"

    if "Error" in suffix:
        cmap = 'seismic'  # Diverging colormap for error fields
    elif is_std:
        cmap = 'viridis'  # Standard deviation → green/yellow
    else:
        cmap = 'plasma'

    vmin, vmax = None, None
    if custom_range is not None:
        vmin, vmax = custom_range

    # 1. All time-slice visualization
    viz_3d_path = os.path.join(RESULTS_DIR,
                               f"{model_name}_{suffix.lower().replace(' ', '_')}_3d.png")
    visualize_3d_slices(
        tensor_data,
        f"{title_base} (3D Slices)",
        viz_3d_path,
        cmap=cmap,
        mask_zeros=mask_zeros,
        vmin=vmin,
        vmax=vmax
    )

    # # 2. Flattened 2D visualization (optional)
    # viz_2d_path = os.path.join(RESULTS_DIR,
    #                            f"{model_name}_{suffix.lower().replace(' ', '_')}_2d_flat.png")
    # visualize_2d_flattened(tensor_data, f"{title_base} (2D Space-Time)", viz_2d_path, cmap=cmap)

    # 3. Temporal Mean Map (only meaningful for predictions / ground truth)
    if not is_std:
        viz_mean_path = os.path.join(RESULTS_DIR,
                                     f"{model_name}_{suffix.lower().replace(' ', '_')}_temporal_mean.png")
        visualize_temporal_mean(
            tensor_data,
            f"{title_base} (Temporal Mean)",
            viz_mean_path,
            cmap=cmap
        )
