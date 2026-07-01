import numpy as np

import matplotlib.pyplot as plt


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
