from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_wavefront(wavefront, cmap="twilight", title="wavefront", extent = None):
        
    fig, ax = plt.subplots()
   
    ax.set_title(title) 
    im = ax.imshow(wavefront, cmap= cmap, origin = "lower", extent = extent)
    fig.colorbar(im, ax=ax)
    
    

def plot_camImg(img, title="CamImg", extent=None):
   
    
    fig, ax = plt.subplots()
    
    ax.set_title(title) 
    
    im = ax.imshow(img, cmap= "nipy_spectral", origin = "lower", extent = extent)
    fig.colorbar(im, ax=ax)


def plot_com_positions(coms, com_ref, img=None, roi=None, title="COM positions", extent=None):
    """Plot all measured COM positions and the reference COM.

    coms can be a flat array/list of points with shape (K, 2) or a patch grid
    with shape (N, M, 2). Invalid rows containing NaN are ignored.
    """
    coms = np.asarray(coms, dtype=float).reshape(-1, 2)
    valid = np.all(np.isfinite(coms), axis=1)
    coms = coms[valid]

    com_ref = np.asarray(com_ref, dtype=float)
    if com_ref.shape != (2,) or not np.all(np.isfinite(com_ref)):
        raise ValueError("com_ref must be a finite point with shape (2,).")

    fig, ax = plt.subplots()
    ax.set_title(title)

    if img is not None:
        im = ax.imshow(img, cmap="nipy_spectral", origin="lower", extent=extent)
        fig.colorbar(im, ax=ax)

    if roi is not None:
        y0, y1, x0, x1 = roi
        ax.plot(
            [x0, x1, x1, x0, x0],
            [y0, y0, y1, y1, y0],
            color="white",
            linewidth=1.2,
            linestyle="--",
            label="ROI",
        )

    if coms.size > 0:
        ax.scatter(
            coms[:, 0],
            coms[:, 1],
            s=24,
            c="tab:blue",
            alpha=0.75,
            label="COM positions",
        )

    ax.scatter(
        com_ref[0],
        com_ref[1],
        s=90,
        c="tab:red",
        marker="x",
        linewidths=2.5,
        label="Reference COM",
    )

    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()

    return fig, ax


def plot_gradients_quiver(
    gx,
    gy,
    title="gradient field",
    extent=None,
    quiver_step=None,
    quiver_color="tab:blue",
    quiver_scale=None,
    show_magnitude=True,
    magnitude_cmap="viridis",
):
    """Plot x/y gradients as a quiver plot.

    gx and gy must be 2D arrays with the same shape and are interpreted as
    the gradient components in x and y direction.
    """
    gx = np.asarray(gx, dtype=float)
    gy = np.asarray(gy, dtype=float)

    if gx.ndim != 2 or gy.ndim != 2:
        raise ValueError("gx and gy must be 2D arrays.")
    if gx.shape != gy.shape:
        raise ValueError("gx and gy must have the same shape.")
    if not np.all(np.isfinite(gx)) or not np.all(np.isfinite(gy)):
        raise ValueError("gx and gy contain NaN or infinite values.")

    rows, cols = gx.shape
    if quiver_step is None:
        quiver_step = max(1, int(np.ceil(max(rows, cols) / 30)))
    if quiver_step <= 0:
        raise ValueError("quiver_step must be greater than 0.")

    y_idx = np.arange(0, rows, quiver_step)
    x_idx = np.arange(0, cols, quiver_step)

    if extent is None:
        x_coords = x_idx
        y_coords = y_idx
    else:
        x_min, x_max, y_min, y_max = extent
        x_coords = np.linspace(x_min, x_max, cols)[x_idx]
        y_coords = np.linspace(y_min, y_max, rows)[y_idx]

    X, Y = np.meshgrid(x_coords, y_coords)
    qx = gx[np.ix_(y_idx, x_idx)]
    qy = gy[np.ix_(y_idx, x_idx)]

    fig, ax = plt.subplots()
    ax.set_title(title)

    if show_magnitude:
        magnitude = np.hypot(gx, gy)
        im = ax.imshow(magnitude, cmap=magnitude_cmap, origin="lower", extent=extent)
        fig.colorbar(im, ax=ax, label="|gradient|")

    ax.quiver(
        X,
        Y,
        qx,
        qy,
        color=quiver_color,
        scale=quiver_scale,
        angles="xy",
        scale_units="xy",
        width=0.0025,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")

    return fig, ax


def plot_phase_gradient(
    phase,
    gx,
    gy,
    cmap="twilight",
    title="phase gradient",
    extent=None,
    quiver_step=None,
    quiver_color="black",
    quiver_scale=None,
    show_contours=True,
    contour_levels=12,
    contour_color="white",
):
    """Plot a phase map with downsampled gradient vectors and optional iso lines.

    gx and gy must be the same shape as phase and are interpreted as the
    gradient components in x and y direction.
    """
    phase = np.asarray(phase)
    gx = np.asarray(gx)
    gy = np.asarray(gy)

    if phase.ndim != 2:
        raise ValueError("phase must be a 2D array.")
    if gx.shape != phase.shape or gy.shape != phase.shape:
        raise ValueError("gx and gy must have the same shape as phase.")

    rows, cols = phase.shape
    if quiver_step is None:
        quiver_step = max(1, int(np.ceil(max(rows, cols) / 30)))
    if quiver_step <= 0:
        raise ValueError("quiver_step must be greater than 0.")

    y_idx = np.arange(0, rows, quiver_step)
    x_idx = np.arange(0, cols, quiver_step)

    if extent is None:
        x_coords = x_idx
        y_coords = y_idx
    else:
        x_min, x_max, y_min, y_max = extent
        x_coords = np.linspace(x_min, x_max, cols)[x_idx]
        y_coords = np.linspace(y_min, y_max, rows)[y_idx]

    X, Y = np.meshgrid(x_coords, y_coords)
    qx = gx[np.ix_(y_idx, x_idx)]
    qy = gy[np.ix_(y_idx, x_idx)]

    fig, ax = plt.subplots()
    ax.set_title(title)

    im = ax.imshow(phase, cmap=cmap, origin="lower", extent=extent)
    fig.colorbar(im, ax=ax)

    if show_contours:
        contour = ax.contour(
            phase,
            levels=contour_levels,
            colors=contour_color,
            linewidths=0.7,
            alpha=0.7,
            origin="lower",
            extent=extent,
        )
        ax.clabel(contour, inline=True, fontsize=8, fmt="%.2g")

    ax.quiver(
        X,
        Y,
        qx,
        qy,
        color=quiver_color,
        scale=quiver_scale,
        angles="xy",
        scale_units="xy",
        width=0.0025,
    )

    return fig, ax


def _save_calibration_figure(fig, output_path, dpi):
    """Save and close one calibration figure."""
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_calibration_image(ax, image, title, colorbar_label=None, cmap="viridis"):
    """Add a consistently formatted image and colorbar to an axes."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D image for '{title}', got shape {image.shape}."
        )
    im = ax.imshow(image, cmap=cmap, origin="lower", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    ax.figure.colorbar(im, ax=ax, label=colorbar_label)
    return im


def save_calibration_plots(data, log_directory, dpi=250):
    """Create the standard diagnostic plot suite for one calibration run.

    All labels are intentionally English. Missing optional measurements are
    skipped within a figure without preventing the remaining plots from being
    written.
    """
    required = ("phase", "gx", "gy", "A_patch", "power_patch", "com_patch", "com_ref")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Missing calibration data for plotting: {', '.join(missing)}")

    plots_directory = Path(log_directory) / "plots"
    plots_directory.mkdir(parents=True, exist_ok=True)

    phase = np.asarray(data["phase"], dtype=float)
    gx = np.asarray(data["gx"], dtype=float)
    gy = np.asarray(data["gy"], dtype=float)
    amplitude = np.asarray(data["A_patch"], dtype=float)
    power = np.asarray(data["power_patch"], dtype=float)
    com_patch = np.asarray(data["com_patch"], dtype=float)
    com_ref = np.asarray(data["com_ref"], dtype=float)

    if phase.ndim != 2 or gx.shape != phase.shape or gy.shape != phase.shape:
        raise ValueError("phase, gx and gy must be equally sized 2D arrays.")
    if amplitude.ndim != 2 or power.shape != amplitude.shape:
        raise ValueError("A_patch and power_patch must be equally sized 2D arrays.")
    if com_patch.shape != amplitude.shape + (2,):
        raise ValueError("com_patch must have shape A_patch.shape + (2,).")
    if com_ref.shape != (2,):
        raise ValueError("com_ref must have shape (2,).")

    valid_spots = np.all(np.isfinite(com_patch), axis=-1)
    spot_delta = com_patch - com_ref
    spot_error = np.where(valid_spots, np.linalg.norm(spot_delta, axis=-1), np.nan)
    gradient_magnitude = np.hypot(gx, gy)
    wrapped_phase = np.mod(phase, 2 * np.pi)
    amplitude_valid = np.asarray(
        data.get("amplitude_measurement_valid", valid_spots),
        dtype=bool,
    )
    if amplitude_valid.shape != amplitude.shape:
        amplitude_valid = valid_spots.copy()
    gradient_exposure_valid = np.asarray(
        data.get("gradient_exposure_valid", valid_spots),
        dtype=bool,
    )
    if gradient_exposure_valid.shape != amplitude.shape:
        gradient_exposure_valid = valid_spots.copy()
    invalid_patches = (
        ~valid_spots
        | (
            bool(data.get("separate_amplitude_measurement", False))
            & ~amplitude_valid
        )
        | (
            bool(data.get("adaptive_gradient_exposure", False))
            & ~gradient_exposure_valid
        )
    ).astype(float)
    saved_paths = []

    # 01: Compact overview
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    _plot_calibration_image(axes[0, 0], wrapped_phase, "Wrapped phase", "Phase [rad]", "twilight")
    _plot_calibration_image(axes[0, 1], gx, "x phase gradient", "Gradient [rad/m]")
    _plot_calibration_image(axes[0, 2], gy, "y phase gradient", "Gradient [rad/m]")
    _plot_calibration_image(axes[1, 0], amplitude, "Normalized patch amplitude", "Relative amplitude")
    power_label = (
        "Power rate [a.u./us]"
        if (
            data.get("separate_amplitude_measurement", False)
            or data.get("adaptive_gradient_exposure", False)
        )
        else "Power [a.u.]"
    )
    _plot_calibration_image(axes[1, 1], power, "Patch power", power_label)
    _plot_calibration_image(axes[1, 2], spot_error, "Final spot error", "Error [px]", "magma")
    fig.suptitle("Wavefront calibration summary", fontsize=16)
    output_path = plots_directory / "01_summary.png"
    _save_calibration_figure(fig, output_path, dpi)
    saved_paths.append(output_path)

    # 02: Camera background and reference measurement
    background = data.get("background_image")
    reference = data.get("reference_image")
    camera_images = [
        (background, "Camera background"),
        (reference, "Reference spot image"),
    ]
    available_images = [
        (np.asarray(image), title)
        for image, title in camera_images
        if image is not None and np.asarray(image).ndim == 2
    ]
    if available_images:
        fig, axes = plt.subplots(1, len(available_images), figsize=(7 * len(available_images), 6))
        axes = np.atleast_1d(axes)
        roi = np.asarray(data.get("roi", []), dtype=float)
        for ax, (image, title) in zip(axes, available_images):
            _plot_calibration_image(ax, image, title, "Intensity [a.u.]", "nipy_spectral")
            if roi.shape == (4,):
                y0, y1, x0, x1 = roi
                ax.add_patch(
                    Rectangle(
                        (x0, y0),
                        x1 - x0,
                        y1 - y0,
                        fill=False,
                        edgecolor="white",
                        linewidth=1.5,
                        linestyle="--",
                        label="ROI",
                    )
                )
            if np.all(np.isfinite(com_ref)):
                ax.scatter(
                    com_ref[0],
                    com_ref[1],
                    marker="x",
                    s=90,
                    linewidths=2,
                    color="red",
                    label="Reference COM",
                )
            if roi.shape == (4,) or np.all(np.isfinite(com_ref)):
                ax.legend(loc="best")
        fig.suptitle("Camera reference measurement", fontsize=16)
        output_path = plots_directory / "02_camera_reference.png"
        _save_calibration_figure(fig, output_path, dpi)
        saved_paths.append(output_path)

    # 03: Continuous and wrapped phase
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _plot_calibration_image(axes[0], phase, "Reconstructed phase", "Phase [rad]", "viridis")
    _plot_calibration_image(axes[1], wrapped_phase, "Wrapped phase (0 to 2 pi)", "Phase [rad]", "twilight")
    fig.suptitle("Reconstructed wavefront phase", fontsize=16)
    output_path = plots_directory / "03_phase.png"
    _save_calibration_figure(fig, output_path, dpi)
    saved_paths.append(output_path)

    # 04: Gradient components, magnitude and direction
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    _plot_calibration_image(axes[0, 0], gx, "x phase gradient", "Gradient [rad/m]")
    _plot_calibration_image(axes[0, 1], gy, "y phase gradient", "Gradient [rad/m]")
    _plot_calibration_image(axes[1, 0], gradient_magnitude, "Gradient magnitude", "Magnitude [rad/m]")
    step = max(1, int(np.ceil(max(gx.shape) / 30)))
    y_idx = np.arange(0, gx.shape[0], step)
    x_idx = np.arange(0, gx.shape[1], step)
    sampled_magnitude = gradient_magnitude[np.ix_(y_idx, x_idx)]
    if np.any(np.isfinite(sampled_magnitude) & (sampled_magnitude > 0)):
        axes[1, 1].quiver(
            x_idx,
            y_idx,
            gx[np.ix_(y_idx, x_idx)],
            gy[np.ix_(y_idx, x_idx)],
            sampled_magnitude,
            cmap="viridis",
            angles="xy",
        )
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "No non-zero gradients available",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
    axes[1, 1].set_title("Gradient direction")
    axes[1, 1].set_xlabel("x index")
    axes[1, 1].set_ylabel("y index")
    axes[1, 1].set_aspect("equal", adjustable="box")
    fig.suptitle("Wavefront gradient diagnostics", fontsize=16)
    output_path = plots_directory / "04_gradients.png"
    _save_calibration_figure(fig, output_path, dpi)
    saved_paths.append(output_path)

    # 05: Patch signal quality and invalid measurements
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _plot_calibration_image(axes[0], amplitude, "Normalized patch amplitude", "Relative amplitude")
    _plot_calibration_image(axes[1], power, "Measured patch power", power_label)
    im = axes[2].imshow(invalid_patches, cmap="gray_r", origin="lower", vmin=0, vmax=1, aspect="auto")
    axes[2].set_title("Invalid or skipped patches")
    axes[2].set_xlabel("Patch x index")
    axes[2].set_ylabel("Patch y index")
    fig.colorbar(im, ax=axes[2], ticks=[0, 1], label="0 = valid, 1 = invalid")
    fig.suptitle("Patch measurement quality", fontsize=16)
    output_path = plots_directory / "05_patch_quality.png"
    _save_calibration_figure(fig, output_path, dpi)
    saved_paths.append(output_path)

    # 06: Final spot displacement relative to the reference spot
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    _plot_calibration_image(axes[0, 0], spot_delta[..., 0], "x spot deviation", "Deviation [px]", "coolwarm")
    _plot_calibration_image(axes[0, 1], spot_delta[..., 1], "y spot deviation", "Deviation [px]", "coolwarm")
    _plot_calibration_image(axes[1, 0], spot_error, "Spot positioning error", "Error [px]", "magma")
    patch_y, patch_x = np.indices(amplitude.shape)
    if np.any(valid_spots & (spot_error > 0)):
        axes[1, 1].quiver(
            patch_x[valid_spots],
            patch_y[valid_spots],
            spot_delta[..., 0][valid_spots],
            spot_delta[..., 1][valid_spots],
            spot_error[valid_spots],
            cmap="magma",
            angles="xy",
            scale_units="xy",
        )
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "No non-zero spot deviations available",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
    axes[1, 1].set_title("Spot deviation vectors")
    axes[1, 1].set_xlabel("Patch x index")
    axes[1, 1].set_ylabel("Patch y index")
    axes[1, 1].set_aspect("equal", adjustable="box")
    fig.suptitle("Spot positioning diagnostics", fontsize=16)
    output_path = plots_directory / "06_spot_deviation.png"
    _save_calibration_figure(fig, output_path, dpi)
    saved_paths.append(output_path)

    # 07: Error history for every patch plus aggregate convergence
    convergence = np.asarray(data.get("convergence", []), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if convergence.size:
        convergence = convergence.reshape(-1, 4)
        patch_ids = np.unique(convergence[:, :2], axis=0)
        for patch_x_index, patch_y_index in patch_ids:
            rows = convergence[
                (convergence[:, 0] == patch_x_index)
                & (convergence[:, 1] == patch_y_index)
            ]
            rows = rows[np.argsort(rows[:, 2])]
            axes[0].plot(rows[:, 2], rows[:, 3], color="tab:blue", alpha=0.18, linewidth=0.8)

        iterations = np.unique(convergence[:, 2]).astype(int)
        medians = []
        means = []
        for iteration in iterations:
            errors = convergence[convergence[:, 2] == iteration, 3]
            medians.append(np.nanmedian(errors))
            means.append(np.nanmean(errors))
        axes[0].plot(iterations, medians, marker="o", color="black", linewidth=2, label="Median error")
        axes[0].plot(iterations, means, marker="s", color="tab:red", linewidth=1.5, label="Mean error")
        eps_px = data.get("eps_px")
        if eps_px is not None:
            axes[0].axhline(float(eps_px), color="tab:green", linestyle="--", label="Target error")
        axes[0].legend()
    else:
        axes[0].text(
            0.5,
            0.5,
            "No convergence measurements available",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )

    axes[0].set_title("Patch convergence traces")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Spot error [px]")
    axes[0].grid(alpha=0.25)

    finite_error = spot_error[np.isfinite(spot_error)]
    if finite_error.size:
        bins = min(30, max(5, int(np.sqrt(finite_error.size))))
        axes[1].hist(finite_error, bins=bins, color="tab:blue", alpha=0.8)
    else:
        axes[1].text(
            0.5,
            0.5,
            "No valid final spot measurements",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
    axes[1].set_title("Final spot error distribution")
    axes[1].set_xlabel("Spot error [px]")
    axes[1].set_ylabel("Patch count")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Calibration convergence", fontsize=16)
    output_path = plots_directory / "07_convergence.png"
    _save_calibration_figure(fig, output_path, dpi)
    saved_paths.append(output_path)

    # 08: Fitted Zernike coefficient for every requested Fringe mode
    zernike_indices = np.asarray(data.get("zernike_indices", []), dtype=int)
    zernike_weights = np.asarray(
        data.get("zernike_coefficients_waves", []),
        dtype=float,
    )
    zernike_names = np.asarray(data.get("zernike_names", []), dtype=str)
    figure_width = max(12, 0.8 * max(1, zernike_indices.size))
    fig, axes = plt.subplots(2, 1, figsize=(figure_width, 10), sharex=True)
    if zernike_indices.size and zernike_weights.shape == zernike_indices.shape:
        positions = np.arange(zernike_indices.size)
        colors = np.where(zernike_weights >= 0, "tab:blue", "tab:orange")
        axes[0].bar(positions, zernike_weights, color=colors, alpha=0.85)
        if zernike_names.shape == zernike_indices.shape:
            labels = [
                f"Z{index}\n{name}"
                for index, name in zip(zernike_indices, zernike_names)
            ]
        else:
            labels = [f"Z{index}" for index in zernike_indices]
        axes[1].set_xticks(positions, labels, rotation=45, ha="right")
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].margins(y=0.12)

        magnitudes = np.abs(zernike_weights)
        nonzero = magnitudes > 0
        if np.any(nonzero):
            axes[1].bar(
                positions[nonzero],
                magnitudes[nonzero],
                color=colors[nonzero],
                alpha=0.85,
            )
            axes[1].set_yscale("log")
        else:
            axes[1].text(
                0.5,
                0.5,
                "All fitted weights are zero",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )
    else:
        message = str(data.get("zernike_fit_error", "No Zernike fit available"))
        for ax in axes:
            ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    axes[0].set_title("Signed coefficients")
    axes[0].set_ylabel("Coefficient [waves]")
    axes[1].set_title("Coefficient magnitudes")
    axes[1].set_xlabel("Zernike polynomial")
    axes[1].set_ylabel("Absolute coefficient [waves]")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Zernike weights by Fringe polynomial", fontsize=16)
    output_path = plots_directory / "08_zernike_weights.png"
    _save_calibration_figure(fig, output_path, dpi)
    saved_paths.append(output_path)

    # 09: Dedicated patch-amplitude diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _plot_calibration_image(
        axes[0],
        amplitude,
        "Patch amplitude map",
        "Normalized amplitude",
    )
    axes[0].set_xlabel("Patch x index")
    axes[0].set_ylabel("Patch y index")

    valid_amplitudes = amplitude[valid_spots & np.isfinite(amplitude)]
    if valid_amplitudes.size:
        bins = min(30, max(5, int(np.sqrt(valid_amplitudes.size))))
        axes[1].hist(valid_amplitudes, bins=bins, color="tab:blue", alpha=0.85)
        axes[1].axvline(
            np.mean(valid_amplitudes),
            color="tab:red",
            linestyle="--",
            label=f"Mean = {np.mean(valid_amplitudes):.3f}",
        )
        axes[1].legend()
    else:
        axes[1].text(
            0.5,
            0.5,
            "No valid patch amplitudes available",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
    axes[1].set_title("Patch amplitude distribution")
    axes[1].set_xlabel("Normalized amplitude")
    axes[1].set_ylabel("Patch count")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Patch amplitudes", fontsize=16)
    output_path = plots_directory / "09_patch_amplitude.png"
    _save_calibration_figure(fig, output_path, dpi)
    saved_paths.append(output_path)

    # 10: Adaptive exposure diagnostics for the separate amplitude pass
    if data.get("separate_amplitude_measurement", False):
        exposure = np.asarray(data.get("amplitude_exposure_us", []), dtype=float)
        peak_fraction = np.asarray(
            data.get("amplitude_peak_fraction", []),
            dtype=float,
        )
        saturation_fraction = np.asarray(
            data.get("amplitude_saturation_fraction", []),
            dtype=float,
        )
        if (
            exposure.shape == amplitude.shape
            and peak_fraction.shape == amplitude.shape
            and saturation_fraction.shape == amplitude.shape
        ):
            fig, axes = plt.subplots(2, 2, figsize=(13, 10))
            _plot_calibration_image(
                axes[0, 0],
                exposure,
                "Selected exposure time",
                "Exposure [us]",
            )
            _plot_calibration_image(
                axes[0, 1],
                100 * peak_fraction,
                "Raw peak level",
                "Full scale [%]",
                "magma",
            )
            _plot_calibration_image(
                axes[1, 0],
                100 * saturation_fraction,
                "Saturated pixel fraction",
                "ROI pixels [%]",
                "magma",
            )
            validity_image = amplitude_valid.astype(float)
            im = axes[1, 1].imshow(
                validity_image,
                cmap="RdYlGn",
                origin="lower",
                vmin=0,
                vmax=1,
                aspect="auto",
            )
            axes[1, 1].set_title("Amplitude measurement validity")
            axes[1, 1].set_xlabel("Patch x index")
            axes[1, 1].set_ylabel("Patch y index")
            fig.colorbar(
                im,
                ax=axes[1, 1],
                ticks=[0, 1],
                label="0 = outside target, 1 = valid",
            )
            fig.suptitle("Adaptive amplitude measurement", fontsize=16)
            output_path = plots_directory / "10_amplitude_exposure.png"
            _save_calibration_figure(fig, output_path, dpi)
            saved_paths.append(output_path)

    # 11: Adaptive exposure diagnostics for reference and gradient scanning
    if data.get("adaptive_gradient_exposure", False):
        exposure = np.asarray(data.get("gradient_exposure_us", []), dtype=float)
        peak_fraction = np.asarray(
            data.get("gradient_peak_fraction", []),
            dtype=float,
        )
        saturation_fraction = np.asarray(
            data.get("gradient_saturation_fraction", []),
            dtype=float,
        )
        if (
            exposure.shape == amplitude.shape
            and peak_fraction.shape == amplitude.shape
            and saturation_fraction.shape == amplitude.shape
        ):
            fig, axes = plt.subplots(2, 2, figsize=(13, 10))
            _plot_calibration_image(
                axes[0, 0],
                exposure,
                "Gradient-scan exposure time",
                "Exposure [us]",
            )
            _plot_calibration_image(
                axes[0, 1],
                100 * peak_fraction,
                "Gradient-scan raw peak level",
                "Full scale [%]",
                "magma",
            )
            _plot_calibration_image(
                axes[1, 0],
                100 * saturation_fraction,
                "Gradient-scan saturated pixel fraction",
                "ROI pixels [%]",
                "magma",
            )
            validity_image = gradient_exposure_valid.astype(float)
            im = axes[1, 1].imshow(
                validity_image,
                cmap="RdYlGn",
                origin="lower",
                vmin=0,
                vmax=1,
                aspect="auto",
            )
            axes[1, 1].set_title("Gradient exposure validity")
            axes[1, 1].set_xlabel("Patch x index")
            axes[1, 1].set_ylabel("Patch y index")
            fig.colorbar(
                im,
                ax=axes[1, 1],
                ticks=[0, 1],
                label="0 = outside target, 1 = valid",
            )
            reference_exposure = data.get("reference_exposure_us")
            reference_note = (
                ""
                if reference_exposure is None
                else f"; reference exposure = {float(reference_exposure):.1f} us"
            )
            fig.suptitle(
                f"Adaptive gradient exposure{reference_note}",
                fontsize=16,
            )
            output_path = plots_directory / "11_gradient_exposure.png"
            _save_calibration_figure(fig, output_path, dpi)
            saved_paths.append(output_path)

    return saved_paths
