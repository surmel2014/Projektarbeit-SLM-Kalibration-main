import json
from datetime import datetime
from pathlib import Path
import plotUtils
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light
from scipy.signal import czt
from zernpy.zernikepol import ZernPol

from scipy.ndimage import center_of_mass, gaussian_filter
from skimage.filters import threshold_otsu
from scipy.interpolate import RectBivariateSpline, griddata
from numpy.fft import fft2, ifft2, fftfreq
from scipy.interpolate import RegularGridInterpolator

def frequencyToWaveLength(freq):
     lam = speed_of_light / freq
     return lam

def WaveLengthToFrequency(lam):
    freq = speed_of_light / lam
    return freq


def _coordinate_grid(resolution, center=None):
    rows, cols = resolution
    if center is None:
        center = ((cols - 1) / 2, (rows - 1) / 2)

    x0, y0 = center
    y, x = np.indices((rows, cols), dtype=float)
    return x - x0, y - y0


def _normalize_pixel_pitch(pixelPitch, name="pixelPitch"):
    if np.isscalar(pixelPitch):
        x_pitch = y_pitch = float(pixelPitch)
    else:
        pixelPitch = tuple(pixelPitch)
        if len(pixelPitch) != 2:
            raise ValueError(f"{name} must be a scalar or a two-value tuple.")
        x_pitch = float(pixelPitch[0])
        y_pitch = float(pixelPitch[1])

    if x_pitch <= 0 or y_pitch <= 0:
        raise ValueError(f"{name} values must be greater than 0.")

    return x_pitch, y_pitch


def _normalize_patch_size(S, name="S"):
    """Return a patch size as (Sx, Sy) in pixels."""
    if np.isscalar(S):
        Sx = Sy = int(S)
    else:
        try:
            S = tuple(S)
        except TypeError as exc:
            raise ValueError(f"{name} must be a scalar or a two-value tuple.") from exc
        if len(S) != 2:
            raise ValueError(f"{name} must be a scalar or a two-value tuple.")
        Sx, Sy = int(S[0]), int(S[1])

    if Sx <= 0 or Sy <= 0:
        raise ValueError(f"{name} values must be greater than 0.")

    return Sx, Sy


def axiconPhase(resolution, axiconAngle=1.0, center=None, wavelength=532e-9, pixelPitch=10e-6):
    """Create a wrapped conical axicon phase map for an SLM.

    axiconAngle is the desired cone/deflection angle in degrees.
    wavelength and pixelPitch are measured in meters.
    """
    if wavelength <= 0:
        raise ValueError("wavelength must be greater than 0.")
    x_pitch, y_pitch = _normalize_pixel_pitch(pixelPitch, "pixelPitch")

    x, y = _coordinate_grid(resolution, center)
    radius_m = np.hypot(x * x_pitch, y * y_pitch)
    angle_rad = np.deg2rad(axiconAngle)
    phase = (2 * np.pi / wavelength) * np.sin(angle_rad) * radius_m
    return np.mod(phase, 2 * np.pi)


def blazegrating(resolution, period=32, direction="x"):
    """Create a wrapped blazed grating phase map for an SLM.

    period is the grating period in pixels. direction can be "x" or "y".
    """
    if period <= 0:
        raise ValueError("period must be greater than 0.")

    rows, cols = resolution
    if direction == "x":
        coordinate = np.tile(np.arange(cols, dtype=float), (rows, 1))
    elif direction == "y":
        coordinate = np.tile(np.arange(rows, dtype=float)[:, None], (1, cols))
    else:
        raise ValueError('direction must be "x" or "y".')

    return np.mod(2 * np.pi * coordinate / period, 2 * np.pi)


def propagateToFarfield(field, wavelength):
    return propagateField(field, 10e-6, 10e-6, wavelength=wavelength)


def getGaussianBeam(
    resolution,
    waistDiameter,
    wavelength=532e-9,
    waistPosition=0.0,
    z=0.0,
    center=None,
    pixelPitch=10e-6,
    amplitude=1.0,
    phaseOffset=0.0,
):
    """Create the complex field of a Gaussian beam on a sampled plane.

    waistDiameter is the 1/e^2 intensity diameter at the beam waist, measured
    in meters. waistPosition and z are axial positions in meters; the returned
    field is evaluated at z relative to the waist position. center is given in
    pixels as (x, y), matching the other phase helpers in this module.
    """
    if waistDiameter <= 0:
        raise ValueError("waistDiameter must be greater than 0.")
    if wavelength <= 0:
        raise ValueError("wavelength must be greater than 0.")
    x_pitch, y_pitch = _normalize_pixel_pitch(pixelPitch, "pixelPitch")

    x, y = _coordinate_grid(resolution, center)
    radius_squared = (x * x_pitch) ** 2 + (y * y_pitch) ** 2

    waist_radius = waistDiameter / 2.0
    axial_distance = z - waistPosition
    wave_number = 2.0 * np.pi / wavelength
    rayleigh_range = np.pi * waist_radius ** 2 / wavelength

    beam_radius = waist_radius * np.sqrt(1.0 + (axial_distance / rayleigh_range) ** 2)
    envelope = amplitude * (waist_radius / beam_radius) * np.exp(-radius_squared / beam_radius ** 2)

    gouy_phase = np.arctan2(axial_distance, rayleigh_range)
    phase = phaseOffset + wave_number * axial_distance - gouy_phase

    if axial_distance != 0:
        radius_of_curvature = axial_distance * (1.0 + (rayleigh_range / axial_distance) ** 2)
        phase += wave_number * radius_squared / (2.0 * radius_of_curvature)

    return envelope * np.exp(1j * phase)


def _sampling_axis(samples, sampling, axis_name):
    """Return one physical coordinate axis and its constant sample spacing."""
    if np.isscalar(sampling):
        step = float(sampling)
        if step <= 0:
            raise ValueError(f"{axis_name} pixel pitch must be greater than 0.")
        return (np.arange(samples, dtype=float) - (samples - 1) / 2) * step, step

    sampling = np.asarray(sampling, dtype=float)

    if sampling.ndim == 1:
        if sampling.size == samples:
            axis = sampling
        elif sampling.size == 2:
            axis = np.linspace(sampling[0], sampling[1], samples)
        else:
            raise ValueError(f"{axis_name} must have length {samples} or contain two range limits.")
    else:
        raise ValueError(f"{axis_name} must be a scalar, a coordinate axis, or two range limits.")

    spacing = np.diff(axis)
    if not np.allclose(spacing, spacing[0]):
        raise ValueError(f"{axis_name} sampling must be uniformly spaced.")
    if spacing[0] == 0:
        raise ValueError(f"{axis_name} sampling spacing must not be zero.")

    return axis, float(abs(spacing[0]))


def propagateField(
    field,
    xRange,
    yRange,
    wavelength=532e-9,
    focalLength=1.0,
    outputPitch=None,
    outputResolution=None,
    outputCenter=(0.0, 0.0),
):
    """Propagate a complex field through an ideal 2f lens setup.

    The input field is assumed to be in the front focal plane of a thin lens,
    and the returned field is in the back focal plane. A chirp z-transform is
    used so the output-plane sampling can be chosen directly.

    xRange and yRange describe the input-plane sampling in meters. They can be
    scalar pixel pitches, one-dimensional coordinate axes, or two-value limits.
    focalLength is measured in meters. outputPitch can be a scalar or
    (xPitch, yPitch) in meters. outputResolution defaults to the input shape.
    outputCenter is the focal-plane coordinate at the center pixel, in meters.

    Returns:
        focalField: complex field in the back focal plane.
        xFocal: 1D x-coordinate axis of the focal plane in meters.
        yFocal: 1D y-coordinate axis of the focal plane in meters.
    """
    field = np.asarray(field, dtype=complex)
    if field.ndim != 2:
        raise ValueError(f"Expected a 2D complex field, got shape {field.shape}.")
    if wavelength <= 0:
        raise ValueError("wavelength must be greater than 0.")
    if focalLength <= 0:
        raise ValueError("focalLength must be greater than 0.")

    rows, cols = field.shape
    x_axis, dx = _sampling_axis(cols, xRange, "xRange")
    y_axis, dy = _sampling_axis(rows, yRange, "yRange")

    if outputResolution is None:
        out_rows, out_cols = rows, cols
    else:
        out_rows, out_cols = outputResolution
        if out_rows <= 0 or out_cols <= 0:
            raise ValueError("outputResolution values must be greater than 0.")

    if outputPitch is None:
        outputPitch = (wavelength * focalLength / (cols * dx), wavelength * focalLength / (rows * dy))
    elif np.isscalar(outputPitch):
        outputPitch = (float(outputPitch), float(outputPitch))
    else:
        outputPitch = tuple(outputPitch)
        if len(outputPitch) != 2:
            raise ValueError("outputPitch must be a scalar or a two-value tuple.")

    x_output_pitch, y_output_pitch = outputPitch
    if x_output_pitch <= 0 or y_output_pitch <= 0:
        raise ValueError("outputPitch values must be greater than 0.")

    x_center, y_center = outputCenter
    x_focal_axis = x_center + (np.arange(out_cols, dtype=float) - (out_cols - 1) / 2) * x_output_pitch
    y_focal_axis = y_center + (np.arange(out_rows, dtype=float) - (out_rows - 1) / 2) * y_output_pitch

    fx_start = x_focal_axis[0] / (wavelength * focalLength)
    fy_start = y_focal_axis[0] / (wavelength * focalLength)
    dfx = x_output_pitch / (wavelength * focalLength)
    dfy = y_output_pitch / (wavelength * focalLength)

    x_transform = czt(
        field,
        m=out_cols,
        a=np.exp(2j * np.pi * fx_start * dx),
        w=np.exp(-2j * np.pi * dfx * dx),
        axis=1,
    )
    x_transform *= np.exp(-2j * np.pi * x_focal_axis[None, :] * x_axis[0] / (wavelength * focalLength))

    spectrum = czt(
        x_transform,
        m=out_rows,
        a=np.exp(2j * np.pi * fy_start * dy),
        w=np.exp(-2j * np.pi * dfy * dy),
        axis=0,
    )
    spectrum *= np.exp(-2j * np.pi * y_focal_axis[:, None] * y_axis[0] / (wavelength * focalLength))
    spectrum *= dx * dy

    wave_number = 2.0 * np.pi / wavelength
    focalField = np.exp(1j * 2.0 * wave_number * focalLength) / (1j * wavelength * focalLength)
    focalField *= spectrum

    return focalField, x_focal_axis, y_focal_axis



def wavefront_from_fringe(coeffs, grid_size=512):
    """
    coeffs:
        dict {fringe_index: coefficient_in_waves}

    returns:
        wavefront phase in radiants
    """

    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)

    X, Y = np.meshgrid(x, y)

    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    mask = rho <= 1

    W = np.zeros_like(rho)

    for fringe_index, coeff in coeffs.items():

        zp = ZernPol(fringe=fringe_index)

        Z = zp.polynomial_value(rho, theta)

        W += coeff * Z

    W[~mask] = np.nan

    return W*2*np.pi


def _normalize_resolution(resolution):
    if len(resolution) != 2:
        raise ValueError("resolution must be a two-value tuple (rows, cols).")
    rows, cols = int(resolution[0]), int(resolution[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("resolution values must be greater than 0.")
    return rows, cols


def _zernike_aperture_parameters(
    resolution,
    slm_pitch,
    aperture_mode="full_slm_diagonal",
    center=None,
    radius=None,
):
    rows, cols = _normalize_resolution(resolution)
    x_pitch, y_pitch = _normalize_pixel_pitch(slm_pitch, "slm_pitch")

    if center is None:
        cx = (cols - 1) / 2
        cy = (rows - 1) / 2
    else:
        if len(center) != 2:
            raise ValueError("center must be a two-value tuple (cx, cy).")
        cx, cy = float(center[0]), float(center[1])

    width = cols * x_pitch
    height = rows * y_pitch

    mode = str(aperture_mode).lower()
    if mode in ("full_slm_diagonal", "diagonal", "full"):
        rx = ry = np.hypot(width / 2, height / 2)
    elif mode in ("pupil_circle", "inscribed_circle", "circle"):
        rx = ry = min(width, height) / 2
    elif mode in ("ellipse", "inscribed_ellipse"):
        rx = width / 2
        ry = height / 2
    elif mode == "custom":
        if radius is None:
            raise ValueError('radius must be provided when aperture_mode="custom".')
        if np.isscalar(radius):
            rx = ry = float(radius)
        else:
            if len(radius) != 2:
                raise ValueError("radius must be a scalar or a two-value tuple.")
            rx, ry = float(radius[0]), float(radius[1])
    else:
        raise ValueError(
            'aperture_mode must be "full_slm_diagonal", "pupil_circle", '
            '"ellipse", or "custom".'
        )

    if rx <= 0 or ry <= 0:
        raise ValueError("aperture radius values must be greater than 0.")

    return {
        "mode": aperture_mode,
        "center": (cx, cy),
        "radius": (rx, ry),
        "slm_pitch": (x_pitch, y_pitch),
        "physical_size": (width, height),
    }


def _zernike_normalized_coordinates(x, y, rx, ry):
    u = x / rx
    v = y / ry
    rho = np.hypot(u, v)
    theta = np.mod(np.arctan2(v, u), 2 * np.pi)
    return u, v, rho, theta


def _zernike_fringe_value(fringe_index, u, v):
    rho = np.hypot(u, v)
    theta = np.mod(np.arctan2(v, u), 2 * np.pi)
    return np.asarray(ZernPol(fringe=int(fringe_index)).polynomial_value(rho, theta), dtype=float)


def _zernike_fringe_gradient(fringe_index, u, v, rx, ry, eps=1e-5):
    if eps <= 0:
        raise ValueError("eps must be greater than 0.")

    dZ_du = (
        _zernike_fringe_value(fringe_index, u + eps, v)
        - _zernike_fringe_value(fringe_index, u - eps, v)
    ) / (2 * eps)
    dZ_dv = (
        _zernike_fringe_value(fringe_index, u, v + eps)
        - _zernike_fringe_value(fringe_index, u, v - eps)
    ) / (2 * eps)

    return dZ_du / rx, dZ_dv / ry


def _patch_center_coordinates(gradients, resolution, patch_size, slm_pitch, center):
    gradients = np.asarray(gradients, dtype=float)
    if gradients.ndim != 3 or gradients.shape[-1] != 2:
        raise ValueError("gradients must have shape (N, M, 2).")

    rows, cols = _normalize_resolution(resolution)
    Sx, Sy = _normalize_patch_size(patch_size, "patch_size")
    x_pitch, y_pitch = _normalize_pixel_pitch(slm_pitch, "slm_pitch")
    cx, cy = center

    N, M, _ = gradients.shape
    if M * Sx > cols or N * Sy > rows:
        raise ValueError("patch_size is too large for the gradient grid and SLM resolution.")

    x_px = (np.arange(M, dtype=float) + 0.5) * Sx
    y_px = (np.arange(N, dtype=float) + 0.5) * Sy
    x_grid_px, y_grid_px = np.meshgrid(x_px, y_px)

    x = (x_grid_px - cx) * x_pitch
    y = (y_grid_px - cy) * y_pitch
    return x, y


def fit_zernike_from_gradients(
    gradients,
    resolution,
    patch_size,
    slm_pitch,
    zernike_indices=None,
    amplitudes=None,
    aperture_mode="full_slm_diagonal",
    center=None,
    radius=None,
    regularization=0.0,
    remove_tip_tilt=False,
    amplitude_threshold=None,
    derivative_eps=1e-5,
    output_wrapped=False,
    outside_aperture_value=0.0,
    return_details=False,
):
    """Fit Zernike modes directly to measured phase gradients.

    The input gradients are expected in physical units [rad/m] on the SLM
    plane. The Zernike basis is evaluated at the corresponding patch centers
    after normalizing physical SLM coordinates to the selected aperture.

    Parameters
    ----------
    gradients : ndarray (N, M, 2)
        Patch gradients with gradients[..., 0] = dphi/dx and
        gradients[..., 1] = dphi/dy in rad/m.
    resolution : tuple
        Full SLM resolution as (rows, cols).
    patch_size : int or tuple
        Patch size in SLM pixels. A tuple is interpreted as (Sx, Sy).
    slm_pitch : float or tuple
        SLM pixel pitch in meters. A tuple is interpreted as
        (x_pitch, y_pitch).
    zernike_indices : sequence of int, optional
        Fringe indices passed to zernpy. Piston (fringe 1) is removed because
        it cannot be determined from gradients. Defaults to fringe 2..15.
    amplitudes : ndarray (N, M), optional
        Optional patch amplitudes used as measurement weights.
    aperture_mode : {"full_slm_diagonal", "pupil_circle", "ellipse", "custom"}
        Coordinate normalization used for the Zernike aperture.
    center : tuple, optional
        Aperture center in SLM pixels as (cx, cy). Defaults to display center.
    radius : float or tuple, optional
        Custom aperture radius in meters for aperture_mode="custom".
    regularization : float, optional
        Ridge regularization strength. 0 means ordinary least squares.
    remove_tip_tilt : bool, optional
        If True, fringe indices 2 and 3 are removed from the fit.
    amplitude_threshold : float, optional
        Relative amplitude threshold in [0, 1]. Patches below this normalized
        amplitude are excluded.
    derivative_eps : float, optional
        Finite-difference step in normalized Zernike coordinates.
    output_wrapped : bool, optional
        If True, wrap the returned phase to [0, 2*pi).
    outside_aperture_value : float, optional
        Value assigned to SLM pixels outside rho <= 1.
    return_details : bool, optional
        If True, return a diagnostics dictionary instead of the compact tuple.

    Returns
    -------
    phase_fit : ndarray (rows, cols)
        Fitted unwrapped phase mask in radians, or wrapped if output_wrapped is
        True.
    residuals : ndarray (N, M, 2)
        Measured minus fitted gradients at the patch centers in rad/m.
    coefficients : dict
        Mapping {fringe_index: coefficient_in_radians}.
    """
    gradients = np.asarray(gradients, dtype=float)
    if gradients.ndim != 3 or gradients.shape[-1] != 2:
        raise ValueError("gradients must have shape (N, M, 2).")
    if not np.any(np.isfinite(gradients)):
        raise ValueError("gradients contain no finite values.")
    if regularization < 0:
        raise ValueError("regularization must not be negative.")

    rows, cols = _normalize_resolution(resolution)
    aperture = _zernike_aperture_parameters(
        (rows, cols),
        slm_pitch,
        aperture_mode=aperture_mode,
        center=center,
        radius=radius,
    )
    x_pitch, y_pitch = aperture["slm_pitch"]
    rx, ry = aperture["radius"]
    cx, cy = aperture["center"]

    if zernike_indices is None:
        zernike_indices = list(range(2, 16))
    zernike_indices = [int(idx) for idx in zernike_indices if int(idx) != 1]
    if remove_tip_tilt:
        zernike_indices = [idx for idx in zernike_indices if idx not in (2, 3)]
    if len(zernike_indices) == 0:
        raise ValueError("At least one non-piston Zernike index is required.")

    x_patch, y_patch = _patch_center_coordinates(
        gradients,
        (rows, cols),
        patch_size,
        (x_pitch, y_pitch),
        (cx, cy),
    )
    u_patch, v_patch, rho_patch, _ = _zernike_normalized_coordinates(
        x_patch,
        y_patch,
        rx,
        ry,
    )

    valid = (
        np.isfinite(gradients[..., 0])
        & np.isfinite(gradients[..., 1])
        & (rho_patch <= 1.0 + 1e-12)
    )

    patch_weights = np.ones(gradients.shape[:2], dtype=float)
    if amplitudes is not None:
        amplitudes = np.asarray(amplitudes, dtype=float)
        if amplitudes.shape != gradients.shape[:2]:
            raise ValueError("amplitudes must have shape gradients.shape[:2].")
        finite_amp = np.isfinite(amplitudes)
        valid &= finite_amp
        if np.any(finite_amp):
            amp_max = np.nanmax(amplitudes[finite_amp])
        else:
            amp_max = 0.0
        if amp_max > 0:
            patch_weights = np.clip(amplitudes / amp_max, 0.0, None)
        else:
            patch_weights = np.zeros_like(amplitudes, dtype=float)
        valid &= patch_weights > 0
        if amplitude_threshold is not None:
            if amplitude_threshold < 0:
                raise ValueError("amplitude_threshold must not be negative.")
            valid &= patch_weights >= amplitude_threshold

    if not np.any(valid):
        raise ValueError("No valid patches are available for the Zernike fit.")

    u_valid = u_patch[valid]
    v_valid = v_patch[valid]
    gx_valid = gradients[..., 0][valid]
    gy_valid = gradients[..., 1][valid]
    weights_valid = np.sqrt(np.clip(patch_weights[valid], 0.0, None))

    num_points = u_valid.size
    num_modes = len(zernike_indices)
    A = np.empty((2 * num_points, num_modes), dtype=float)
    b = np.empty(2 * num_points, dtype=float)
    b[0::2] = gx_valid
    b[1::2] = gy_valid

    for col_idx, fringe_index in enumerate(zernike_indices):
        dz_dx, dz_dy = _zernike_fringe_gradient(
            fringe_index,
            u_valid,
            v_valid,
            rx,
            ry,
            eps=derivative_eps,
        )
        A[0::2, col_idx] = dz_dx
        A[1::2, col_idx] = dz_dy

    row_weights = np.empty(2 * num_points, dtype=float)
    row_weights[0::2] = weights_valid
    row_weights[1::2] = weights_valid
    A_weighted = A * row_weights[:, None]
    b_weighted = b * row_weights

    finite_rows = np.all(np.isfinite(A_weighted), axis=1) & np.isfinite(b_weighted)
    if not np.any(finite_rows):
        raise ValueError("Zernike design matrix contains no finite rows.")

    A_solve = A_weighted[finite_rows]
    b_solve = b_weighted[finite_rows]
    if regularization > 0:
        normal = A_solve.T @ A_solve
        rhs = A_solve.T @ b_solve
        coeff_array = np.linalg.solve(
            normal + regularization * np.eye(num_modes),
            rhs,
        )
    else:
        coeff_array, *_ = np.linalg.lstsq(A_solve, b_solve, rcond=None)

    coefficients = {
        int(fringe_index): float(coeff)
        for fringe_index, coeff in zip(zernike_indices, coeff_array)
    }

    fit_flat = A @ coeff_array
    gx_fit_valid = fit_flat[0::2]
    gy_fit_valid = fit_flat[1::2]

    gx_fit = np.full(gradients.shape[:2], np.nan, dtype=float)
    gy_fit = np.full(gradients.shape[:2], np.nan, dtype=float)
    gx_fit[valid] = gx_fit_valid
    gy_fit[valid] = gy_fit_valid

    residuals = np.full_like(gradients, np.nan, dtype=float)
    residuals[..., 0][valid] = gradients[..., 0][valid] - gx_fit_valid
    residuals[..., 1][valid] = gradients[..., 1][valid] - gy_fit_valid

    x_px = np.arange(cols, dtype=float)
    y_px = np.arange(rows, dtype=float)
    x = (x_px - cx) * x_pitch
    y = (y_px - cy) * y_pitch
    X, Y = np.meshgrid(x, y)
    u_full, v_full, rho_full, _ = _zernike_normalized_coordinates(X, Y, rx, ry)

    phase_fit = np.zeros((rows, cols), dtype=float)
    for fringe_index, coeff in coefficients.items():
        phase_fit += coeff * _zernike_fringe_value(fringe_index, u_full, v_full)

    inside = rho_full <= 1.0 + 1e-12
    if outside_aperture_value is not None:
        phase_fit[~inside] = outside_aperture_value

    if np.any(inside):
        phase_fit[inside] -= np.nanmean(phase_fit[inside])
    if output_wrapped:
        phase_fit = wrap_phase(phase_fit)

    residual_rms = np.sqrt(np.nanmean(residuals[..., 0] ** 2 + residuals[..., 1] ** 2))

    if return_details:
        return {
            "phase": phase_fit,
            "residuals": residuals,
            "coefficients": coefficients,
            "coefficients_waves": {
                int(idx): float(coeff / (2 * np.pi))
                for idx, coeff in coefficients.items()
            },
            "zernike_indices": list(zernike_indices),
            "gx_fit": gx_fit,
            "gy_fit": gy_fit,
            "valid_mask": valid,
            "residual_rms": float(residual_rms),
            "aperture": aperture,
        }

    return phase_fit, residuals, coefficients




def wrap_phase(phi):
    return np.mod(phi, 2 * np.pi)


def phaseMaskToGradients(phaseMask, pixelPitch=1.0, unwrap=True):
    """Calculate x/y phase gradients from a 2D phase mask.

    phaseMask is interpreted in radians. Complex input is converted to phase
    with np.angle. If unwrap is True, 2*pi jumps are removed before taking the
    derivative. pixelPitch can be a scalar or (xPitch, yPitch); with meter
    pitches the returned gradients are in rad/m.

    Returns:
        gx: d phase / d x
        
        gy: d phase / d y
    """
    phase = np.asarray(phaseMask)
    if phase.ndim != 2:
        raise ValueError(f"Expected a 2D phase mask, got shape {phase.shape}.")

    if np.iscomplexobj(phase):
        phase = np.angle(phase)
    else:
        phase = phase.astype(float, copy=False)

    if not np.all(np.isfinite(phase)):
        raise ValueError("phaseMask contains NaN or infinite values.")

    dx, dy = _normalize_pixel_pitch(pixelPitch, "pixelPitch")

    if unwrap:
        phase = np.unwrap(np.unwrap(phase, axis=1), axis=0)

    gy, gx = np.gradient(phase, dy, dx)
    return gx, gy


def sphericalPhaseTestMap(
    resolution,
    pixelPitch,
    wavelength=633e-9,
    radiusOfCurvature=1.0,
    center=None,
    exact=True,
    wrap=False,
):
    """Create a spherical phase test map and its analytic gradients.

    The returned gradients are in rad/m and can be passed directly to
    poisson_reconstruct_phase. If exact is True, the phase is
    k * (sqrt(R^2 + x^2 + y^2) - abs(R)); otherwise the paraxial
    approximation k * (x^2 + y^2) / (2R) is used.

    Returns:
        phase: phase map in radians
        gx: d phase / d x in rad/m
        gy: d phase / d y in rad/m
    """
    if wavelength <= 0:
        raise ValueError("wavelength must be greater than 0.")
    if radiusOfCurvature == 0:
        raise ValueError("radiusOfCurvature must not be 0.")

    dx, dy = _normalize_pixel_pitch(pixelPitch, "pixelPitch")

    x_px, y_px = _coordinate_grid(resolution, center)
    x = x_px * dx
    y = y_px * dy
    r_squared = x ** 2 + y ** 2

    wave_number = 2.0 * np.pi / wavelength

    if exact:
        propagation_distance = np.sqrt(radiusOfCurvature ** 2 + r_squared)
        phase = np.sign(radiusOfCurvature) * wave_number * (propagation_distance - abs(radiusOfCurvature))
        gx = np.sign(radiusOfCurvature) * wave_number * x / propagation_distance
        gy = np.sign(radiusOfCurvature) * wave_number * y / propagation_distance
    else:
        phase = wave_number * r_squared / (2.0 * radiusOfCurvature)
        gx = wave_number * x / radiusOfCurvature
        gy = wave_number * y / radiusOfCurvature

    if wrap:
        phase = wrap_phase(phase)

    return phase, gx, gy



def make_patch_ramp(P, Q, S, m, n, u_eff, v_eff, slm_pitch):
    """
    P,Q: SLM-Größe in Pixeln
    S: Patchgröße in Pixeln. Scalar (Sx=Sy=S) or (Sx, Sy).
    m,n: Patchindex
    u_eff,v_eff: effektive Deflektorfrequenz [1/m]
    slm_pitch: SLM-Pixelpitch [m]
    """
    Sx, Sy = _normalize_patch_size(S, "S")

    phase = np.zeros((Q, P), dtype=np.float32)
    amp = np.zeros((Q, P), dtype=np.float32)

    x0, x1 = m * Sx, (m + 1) * Sx
    y0, y1 = n * Sy, (n + 1) * Sy

    # Koordinaten relativ zum Patchzentrum, in Metern
    x_pitch, y_pitch = _normalize_pixel_pitch(slm_pitch, "slm_pitch")
    xs = (np.arange(x0, x1) - (x0 + x1 - 1) / 2) * x_pitch
    ys = (np.arange(y0, y1) - (y0 + y1 - 1) / 2) * y_pitch
    X, Y = np.meshgrid(xs, ys)

    phase[y0:y1, x0:x1] = 2 * np.pi * (X * u_eff + Y * v_eff)
    amp[y0:y1, x0:x1] = 1
    return wrap_phase(phase), amp


def make_centered_patch_ramp(P, Q, S, u_eff, v_eff, slm_pitch):
    """Create a ramp patch centered on the SLM, independently of the patch grid.

    Unlike :func:`make_patch_ramp`, the patch position is not derived from a
    patch index. This is intended for the calibration reference patch, which
    must remain centered even when the regular patch grid has an even number
    of rows or columns. Returns ``(phase, amplitude, center)``; ``center`` uses
    the same pixel-boundary coordinate convention as the calibration view.
    """
    P = int(P)
    Q = int(Q)
    Sx, Sy = _normalize_patch_size(S, "S")
    if P <= 0 or Q <= 0:
        raise ValueError("P and Q must be greater than 0.")
    if Sx > P or Sy > Q:
        raise ValueError("S must not be larger than the SLM resolution.")

    # Center the aperture itself instead of selecting a nominally central
    # member of the regular measurement grid. For a parity mismatch between
    # display and patch, half-pixel-perfect centering is physically impossible;
    # integer slicing then chooses the lower/left of the two equivalent pixels.
    x0 = (P - Sx) // 2
    y0 = (Q - Sy) // 2
    x1 = x0 + Sx
    y1 = y0 + Sy

    phase = np.zeros((Q, P), dtype=np.float32)
    amp = np.zeros((Q, P), dtype=np.float32)

    x_pitch, y_pitch = _normalize_pixel_pitch(slm_pitch, "slm_pitch")
    xs = (np.arange(x0, x1) - (x0 + x1 - 1) / 2) * x_pitch
    ys = (np.arange(y0, y1) - (y0 + y1 - 1) / 2) * y_pitch
    X, Y = np.meshgrid(xs, ys)

    phase[y0:y1, x0:x1] = 2 * np.pi * (X * u_eff + Y * v_eff)
    amp[y0:y1, x0:x1] = 1
    center = ((x0 + x1) / 2, (y0 + y1) / 2)
    return wrap_phase(phase), amp, center


def make_full_display_patch(resolution, u0, v0, slm_pitch):
    """Create one ramp patch spanning the full display."""
    Q, P = resolution
    if Q <= 0 or P <= 0:
        raise ValueError("resolution values must be greater than 0.")
    x_pitch, y_pitch = _normalize_pixel_pitch(slm_pitch, "slm_pitch")

    xs = (np.arange(P) - (P - 1) / 2) * x_pitch
    ys = (np.arange(Q) - (Q - 1) / 2) * y_pitch
    X, Y = np.meshgrid(xs, ys)

    phase = 2 * np.pi * (X * u0 + Y * v0)
    return wrap_phase(phase).astype(np.float32)


def make_patch_mosaic_mask(best_alphas, best_betas, S, slm_pitch, u0=0.0, v0=0.0, resolution=None):
    """Create a full SLM mask from per-patch best alpha/beta corrections.

    best_alphas and best_betas must have shape (N, M), matching patch indices
    n and m. Each patch uses u_eff = u0 - alpha and v_eff = v0 - beta.
    NaN entries are treated as no correction for that patch.
    """
    best_alphas = np.asarray(best_alphas, dtype=float)
    best_betas = np.asarray(best_betas, dtype=float)

    if best_alphas.shape != best_betas.shape:
        raise ValueError("best_alphas and best_betas must have the same shape.")
    if best_alphas.ndim != 2:
        raise ValueError("best_alphas and best_betas must be 2D arrays.")
    Sx, Sy = _normalize_patch_size(S, "S")
    _normalize_pixel_pitch(slm_pitch, "slm_pitch")

    N, M = best_alphas.shape
    if resolution is None:
        Q, P = N * Sy, M * Sx
    else:
        Q, P = resolution
        if Q < N * Sy or P < M * Sx:
            raise ValueError("resolution is too small for the patch arrays and patch size.")

    mask = np.zeros((Q, P), dtype=np.float32)

    for n in range(N):
        for m in range(M):
            alpha = 0.0 if np.isnan(best_alphas[n, m]) else best_alphas[n, m]
            beta = 0.0 if np.isnan(best_betas[n, m]) else best_betas[n, m]
            patch,_ = make_patch_ramp(P, Q, (Sx, Sy), m, n, u0 - alpha, v0 - beta, slm_pitch)
            mask += patch

    return wrap_phase(mask)


def compute_patch_size_for_physical_square(
    resolution,
    slm_pitch,
    prefer_larger=True,
    max_patches_x=None,
    max_patches_y=None,
):
    """
    Bestimme eine geeignete Patchgröße in Pixeln so dass Patches physikalisch
    annähernd quadratisch sind.

    - `resolution` ist ein (Q, P)-Tupel (rows, cols) im Pixelraum.
    - `slm_pitch` kann ein Skalar oder ein (x_pitch, y_pitch)-Tuple in Metern sein.
    - `max_patches_x` / `max_patches_y` begrenzen die maximale Anzahl von
      Patches pro Achse, damit es nicht zu viele werden.
    - Rückgabe: dict mit Feldern `Sx`, `Sy` (Patchgröße in Pixeln für x/y),
      `num_patches_x`, `num_patches_y`, `physical_aspect_ratio` (physischer
      Seitenverhältnis der Patchgröße in Metern, also `(Sx*x_pitch)/(Sy*y_pitch)`),
      und `S_common` als gemeinsamer Divisor (gcd) von Sx und Sy.

    Algorithmus: wir durchlaufen alle Teiler von P bzw. Q (Patchgrößen in
    Pixeln), wählen das Paar (Sx,Sy) mit minimaler Differenz der physikalischen
    Kantenlängen |Sx*x_pitch - Sy*y_pitch|. Bei Gleichstand kann `prefer_larger`
    größere Patches bevorzugen. Wenn `max_patches_x` bzw. `max_patches_y`
    gesetzt sind, werden nur Kandidaten akzeptiert, die nicht mehr als diese
    Anzahl von Patches pro Achse erzeugen.
    """
    import math

    Q, P = resolution
    x_pitch, y_pitch = _normalize_pixel_pitch(slm_pitch, "slm_pitch")

    def divisors(n):
        small = []
        large = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                small.append(i)
                if i != n // i:
                    large.append(n // i)
            i += 1
        return small + large[::-1]

    divP = divisors(P)
    divQ = divisors(Q)

    best = None
    best_pair = (1, 1)

    for Sx in divP:
        num_patches_x = P // Sx
        if max_patches_x is not None and num_patches_x > max_patches_x:
            continue

        phys_x = Sx * x_pitch
        for Sy in divQ:
            num_patches_y = Q // Sy
            if max_patches_y is not None and num_patches_y > max_patches_y:
                continue

            phys_y = Sy * y_pitch
            diff = abs(phys_x - phys_y)
            if best is None:
                best = diff
                best_pair = (Sx, Sy)
            else:
                if diff < best - 1e-18:
                    best = diff
                    best_pair = (Sx, Sy)
                elif abs(diff - best) <= 1e-18 and prefer_larger:
                    # prefer larger physical area
                    area_curr = Sx * Sy * x_pitch * y_pitch
                    area_best = best_pair[0] * best_pair[1] * x_pitch * y_pitch
                    if area_curr > area_best:
                        best_pair = (Sx, Sy)

    Sx, Sy = best_pair
    S_common = math.gcd(Sx, Sy)
    num_patches_x = P // Sx
    num_patches_y = Q // Sy
    physical_aspect_ratio = (Sx * x_pitch) / (Sy * y_pitch)

    return {
        "Sx": int(Sx),
        "Sy": int(Sy),
        "num_patches_x": int(num_patches_x),
        "num_patches_y": int(num_patches_y),
        "physical_aspect_ratio": float(physical_aspect_ratio),
        "S_common": int(S_common),
    }


def spot_com_and_power(image, roi, power_in_mask=False):
    """
    roi = (y0, y1, x0, x1)
    COM wird nach Otsu-Segmentierung berechnet.
    """
    y0, y1, x0, x1 = roi
    crop = image[y0:y1, x0:x1].astype(float)
    crop_blur = gaussian_filter(crop, 5)
    th = threshold_otsu(crop_blur)
    mask = crop_blur > th

    if mask.sum() == 0:
        return None, 0.0

    weighted = crop * mask
    
    cy, cx = center_of_mass(weighted)

    com_x = x0 + cx
    com_y = y0 + cy
    power = weighted.sum() if power_in_mask else crop.sum()
    return np.array([com_x, com_y]), power


def interpolate_patch_values(values, P, Q, S, kind="linear"):
    N, M = values.shape
    Sx, Sy = _normalize_patch_size(S, "S")

    patch_x = (np.arange(M) + 0.5) * Sx
    patch_y = (np.arange(N) + 0.5) * Sy

    k = 3 if kind == "cubic" else 1
    spline = RectBivariateSpline(patch_y, patch_x, values, kx=k, ky=k)

    y = np.arange(Q)
    x = np.arange(P)
    return spline(y, x)


def patch_gradients_to_maps(gradients, P, Q, S=None, remove_mean=False):
    """Expand patch gradients to full-resolution SLM maps without interpolation.

    gradients must have shape (N, M, 2), as stored in data["gradients"] by the
    calibration code. Each patch gradient is copied as a constant block into
    the corresponding SLM region. The returned maps have shape (Q, P).
    """
    gradients = np.asarray(gradients, dtype=float)

    if gradients.ndim != 3 or gradients.shape[-1] != 2:
        raise ValueError("gradients must have shape (N, M, 2).")
    if P <= 0 or Q <= 0:
        raise ValueError("P and Q must be greater than 0.")

    N, M, _ = gradients.shape
    if N <= 0 or M <= 0:
        raise ValueError("gradients must contain at least one patch.")

    if S is not None:
        Sx, Sy = _normalize_patch_size(S, "S")
        if N * Sy > Q or M * Sx > P:
            raise ValueError("S is too large for the requested SLM resolution.")
        y_edges = np.arange(N + 1) * Sy
        x_edges = np.arange(M + 1) * Sx
        y_edges[-1] = Q
        x_edges[-1] = P
    else:
        y_edges = np.rint(np.linspace(0, Q, N + 1)).astype(int)
        x_edges = np.rint(np.linspace(0, P, M + 1)).astype(int)

    gx_map = np.empty((Q, P), dtype=float)
    gy_map = np.empty((Q, P), dtype=float)

    for n in range(N):
        y0, y1 = y_edges[n], y_edges[n + 1]
        for m in range(M):
            x0, x1 = x_edges[m], x_edges[m + 1]
            gx_map[y0:y1, x0:x1] = gradients[n, m, 0]
            gy_map[y0:y1, x0:x1] = gradients[n, m, 1]

    if remove_mean:
        gx_map -= np.nanmean(gx_map)
        gy_map -= np.nanmean(gy_map)

    return gx_map, gy_map


def _fill_missing_patch_values(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a 2D array.")
    if np.all(np.isfinite(values)):
        return values

    y, x = np.indices(values.shape)
    valid = np.isfinite(values)

    if not np.any(valid):
        raise ValueError("Cannot fill patch values because all entries are invalid.")

    points = np.column_stack((y[valid], x[valid]))
    missing_points = np.column_stack((y[~valid], x[~valid]))
    filled = values.copy()

    if points.shape[0] >= 3:
        linear = griddata(points, values[valid], missing_points, method="linear")
        linear_valid = np.isfinite(linear)
        filled[~valid] = linear

        if np.all(linear_valid):
            return filled

        remaining_mask = ~valid.copy()
        remaining_indices = np.flatnonzero(~valid)
        remaining_mask.flat[remaining_indices[linear_valid]] = False
    else:
        remaining_mask = ~valid

    remaining_points = np.column_stack((y[remaining_mask], x[remaining_mask]))
    nearest = griddata(points, values[valid], remaining_points, method="nearest")
    filled[remaining_mask] = nearest

    return filled


def gradient_maps_from_com_deviation(
    com_patch,
    com_ref,
    focal_length,
    wavelength,
    cam_pitch,
    P,
    Q,
    S,
    kind="linear",
    sign=-1.0,
    remove_mean=True,
    fill_missing=True,
):
    """Calculate phase-gradient maps from COM deviations to a reference COM.

    com_patch must have shape (N, M, 2) in camera pixels. com_ref is the
    reference COM [x, y]. The returned gradients are in rad/m and can be passed
    to poisson_reconstruct_phase.

    sign=-1 matches the current calibration convention:
    gradients = -2*pi*(com_patch - com_ref)*cam_pitch/(focal_length*wavelength).
    Flip sign to 1 if the camera/SLM geometry requires the opposite direction.
    """
    com_patch = np.asarray(com_patch, dtype=float)
    com_ref = np.asarray(com_ref, dtype=float)

    if com_patch.ndim != 3 or com_patch.shape[-1] != 2:
        raise ValueError("com_patch must have shape (N, M, 2).")
    if com_ref.shape != (2,) or not np.all(np.isfinite(com_ref)):
        raise ValueError("com_ref must be a finite point with shape (2,).")
    if focal_length <= 0:
        raise ValueError("focal_length must be greater than 0.")
    if wavelength <= 0:
        raise ValueError("wavelength must be greater than 0.")
    if cam_pitch <= 0:
        raise ValueError("cam_pitch must be greater than 0.")
    if P <= 0 or Q <= 0:
        raise ValueError("P and Q must be greater than 0.")

    Sx, Sy = _normalize_patch_size(S, "S")
    if Sx <= 0 or Sy <= 0:
        raise ValueError("S must contain positive patch sizes.")

    px_to_freq = cam_pitch / (focal_length * wavelength)
    gradients_patch = sign * 2 * np.pi * (com_patch - com_ref) * px_to_freq

    gx_patch = gradients_patch[..., 0]
    gy_patch = gradients_patch[..., 1]

    if fill_missing:
        gx_for_interp = _fill_missing_patch_values(gx_patch)
        gy_for_interp = _fill_missing_patch_values(gy_patch)
    else:
        gx_for_interp = gx_patch
        gy_for_interp = gy_patch

    gx_map = interpolate_patch_values(gx_for_interp, P, Q, S, kind=kind)
    gy_map = interpolate_patch_values(gy_for_interp, P, Q, S, kind=kind)

    if remove_mean:
        gx_map -= np.nanmean(gx_map)
        gy_map -= np.nanmean(gy_map)

    return gx_map, gy_map, gradients_patch


def _normalize_pad_width(pad_width, shape):
    rows, cols = shape
    if pad_width is None:
        pad_y = max(1, rows // 2)
        pad_x = max(1, cols // 2)
    elif np.isscalar(pad_width):
        pad_y = pad_x = int(pad_width)
    else:
        pad_y, pad_x = pad_width
        pad_y = int(pad_y)
        pad_x = int(pad_x)

    if pad_y < 0 or pad_x < 0:
        raise ValueError("pad_width must not be negative.")

    return ((pad_y, pad_y), (pad_x, pad_x))

def reconstruct_phase_from_gradients(gx, gy, dx):
    Q, P = gx.shape

    Gx = fft2(gx)
    Gy = fft2(gy)

    dx_x, dx_y = _normalize_pixel_pitch(dx, "dx")
    fx = fftfreq(P, d=dx_x)
    fy = fftfreq(Q, d=dx_y)
    U, V = np.meshgrid(fx, fy)

    denom = (2*np.pi)**2 * (U**2 + V**2)
    denom[0, 0] = np.inf

    phi_hat = (-1j * 2*np.pi * U * Gx - 1j * 2*np.pi * V * Gy) / denom
    phi_hat[0, 0] = 0

    phi = np.real(ifft2(phi_hat))
    phi -= phi.mean()
    return phi


def find_peak_and_threshold(img):
    peak = np.max


def poisson_reconstruct_phase(gx, gy, dx, pad_width=None, pad_mode="edge"):
    """
    Löst ∇² phi = div(g) im Fourier-Raum.
    gx, gy in rad/m; dx = SLM-Pixelpitch in m, scalar or (xPitch, yPitch).

    The gradient fields are padded before the FFT solve and cropped back to the
    original shape afterwards. This reduces edge artifacts from the periodic
    boundary condition implied by the FFT.
    """
    gx = np.asarray(gx, dtype=float)
    gy = np.asarray(gy, dtype=float)
    if gx.shape != gy.shape:
        raise ValueError("gx and gy must have the same shape.")
    if gx.ndim != 2:
        raise ValueError(f"Expected 2D gradient fields, got shape {gx.shape}.")

    dx_x, dx_y = _normalize_pixel_pitch(dx, "dx")
    if not np.all(np.isfinite(gx)) or not np.all(np.isfinite(gy)):
        raise ValueError("gx and gy must contain only finite values.")

    original_rows, original_cols = gx.shape
    pad_spec = _normalize_pad_width(pad_width, gx.shape)
    gx_padded = np.pad(gx, pad_spec, mode=pad_mode)
    gy_padded = np.pad(gy, pad_spec, mode=pad_mode)

    Q, P = gx_padded.shape

    dgx_dx = np.gradient(gx_padded, dx_x, axis=1)
    dgy_dy = np.gradient(gy_padded, dx_y, axis=0)
    div_g = dgx_dx + dgy_dy

    F_div = fft2(div_g)

    fx = fftfreq(P, d=dx_x)
    fy = fftfreq(Q, d=dx_y)
    U, V = np.meshgrid(fx, fy)

    denom = (2 * np.pi) ** 2 * (U**2 + V**2)
    denom[0, 0] = np.inf

    phi_padded = -np.real(ifft2(F_div / denom))

    y_pad = pad_spec[0][0]
    x_pad = pad_spec[1][0]
    phi = phi_padded[y_pad:y_pad + original_rows, x_pad:x_pad + original_cols]
    phi -= phi.mean()
    return phi
    
def poisson_reconstruct_phase_direct_Fourier_Integration(gx, gy, dx):
    Q, P = gx.shape

    Gx = fft2(gx)
    Gy = fft2(gy)

    dx_x, dx_y = _normalize_pixel_pitch(dx, "dx")
    fx = fftfreq(P, d=dx_x)
    fy = fftfreq(Q, d=dx_y)
    U, V = np.meshgrid(fx, fy)

    denom = (2*np.pi)**2 * (U**2 + V**2)
    denom[0, 0] = np.inf

    phi_hat = (-1j * 2*np.pi * U * Gx - 1j * 2*np.pi * V * Gy) / denom
    phi_hat[0, 0] = 0

    phi = np.real(ifft2(phi_hat))
    phi -= phi.mean()
    return phi




def reconstruct_phase_from_patch_gradients(
    gradients,
    P,
    Q,
    S,
    slm_pitch,
    interpolation_method="cubic",
):
    """
    Rekonstruiert eine Phase aus Patch-Gradienten.

    Vorgehen:
    1. Phase auf dem Patch-Gitter berechnen
    2. Phase auf das volle SLM-Gitter interpolieren

    Parameters
    ----------
    gradients : ndarray (N, M, 2)
        Patchgradienten [rad/m]
        gradients[...,0] = gx
        gradients[...,1] = gy

    P : int
        SLM-Breite in Pixeln

    Q : int
        SLM-Höhe in Pixeln

    S : int
        Patchgröße in Pixeln

    slm_pitch : float
        Pixelpitch des SLM [m]

    phase_reconstruct_func : callable
        Funktion der Form

            phi = phase_reconstruct_func(gx, gy, dx)

    interpolation_method : str
        "linear", "nearest", "cubic"

    Returns
    -------
    phi_map : ndarray (Q, P)
        Phase auf dem SLM-Gitter [rad]

    phi_patch : ndarray (N, M)
        Phase auf dem Patch-Gitter [rad]
    """

    gx_patch = gradients[..., 0]
    gy_patch = gradients[..., 1]

    N, M = gx_patch.shape

    # Abstand zwischen zwei Patchzentren
    Sx, Sy = _normalize_patch_size(S, "S")
    x_pitch, y_pitch = _normalize_pixel_pitch(slm_pitch, "slm_pitch")
    dx_patch = (Sx * x_pitch, Sy * y_pitch)

    # Phase auf Patch-Gitter rekonstruieren
    phi_patch = poisson_reconstruct_phase_direct_Fourier_Integration(
        gx_patch,
        gy_patch,
        dx_patch,
    )

    phi_patch -= phi_patch.mean()

    # --------------------------------------------------
    # Koordinaten der Patchzentren
    # --------------------------------------------------

    patch_x = (np.arange(M) + 0.5) * Sx
    patch_y = (np.arange(N) + 0.5) * Sy

    interpolator = RegularGridInterpolator(
        (patch_y, patch_x),
        phi_patch,
        method=interpolation_method,
        bounds_error=False,
        fill_value=None,
    )

    # --------------------------------------------------
    # Zielkoordinaten: alle SLM Pixel
    # --------------------------------------------------

    yy, xx = np.meshgrid(
        np.arange(Q),
        np.arange(P),
        indexing="ij",
    )

    points = np.column_stack([
        yy.ravel(),
        xx.ravel(),
    ])

    phi_map = interpolator(points).reshape(Q, P)

    phi_map -= phi_map.mean()

    return phi_map, phi_patch

def curl_2d(gx, gy, dx):
    """
    Berechnet die z-Komponente des Curls eines 2D-Vektorfelds (gx, gy).

    Parameter
    ----------
    gx, gy : 2D ndarray
        Komponenten des Vektorfelds.
    dx : float or tuple
        Gitterabstand für x und y. Kann ein Skalar oder ein (dx_x, dx_y)-Tuple sein.

    Returns
    -------
    curl : 2D ndarray
        z-Komponente des Curls.
    """
    dx_x, dx_y = _normalize_pixel_pitch(dx, "dx")
    dgy_dy, dgy_dx = np.gradient(gy, dx_y, dx_x)
    dgx_dy, dgx_dx = np.gradient(gx, dx_y, dx_x)

    return dgy_dx - dgx_dy

def ifta(
    shape=(1080, 1920),
    iterations=50,
    seed=None,
    div=10,
    numX=10,
    numY=10,
    numSpotsX=None,
    numSpotsY=None,
):
    """Create a phase-only hologram for a rectangular multispot farfield pattern."""
    if len(shape) != 2:
        raise ValueError("shape must be a tuple of (height, width).")
    if iterations < 1:
        raise ValueError("iterations must be at least 1.")
    if div <= 0:
        raise ValueError("div must be greater than 0.")

    if numSpotsX is not None:
        numX = numSpotsX
    if numSpotsY is not None:
        numY = numSpotsY

    numX = int(numX)
    numY = int(numY)
    if numX < 1 or numY < 1:
        raise ValueError("numX and numY must be at least 1.")

    target = np.zeros(shape, dtype=float)
    rng = np.random.default_rng(seed)

    rows, cols = shape
    center_y = (rows - 1) / 2
    center_x = (cols - 1) / 2

    def _spot_positions(center, half_width, count, max_index):
        if count == 1:
            positions = np.array([center])
        else:
            positions = np.linspace(center - half_width, center + half_width, count)
        return np.clip(np.rint(positions).astype(int), 0, max_index)

    x_positions = _spot_positions(center_x, cols / div, numX, cols - 1)
    y_positions = _spot_positions(center_y, rows / div, numY, rows - 1)

    for y in y_positions:
        for x in x_positions:
            target[y, x] = 1.0

    phase = rng.uniform(0, 2*np.pi, shape)
    field = np.exp(1j * phase)

    for i in range(iterations):
        print(f"it {i + 1} of {iterations}")
        farfield = np.fft.fftshift(np.fft.fft2(field))

        farfield_phase = np.angle(farfield)
        farfield = target * np.exp(1j * farfield_phase)

        field_back = np.fft.ifft2(np.fft.ifftshift(farfield))

        phase = np.angle(field_back)
        field = np.exp(1j * phase)

    return wrap_phase(phase)


def periodic_gs_multispot_slm(
    slm_shape,
    spots=((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)),
    weights=None,
    iterations=300,
    seed=None,
    crop=True,
):
    """
    Periodische DOE-Maske für eine gegebene SLM-Shape.

    Parameters
    ----------
    slm_shape : tuple
        Shape des SLMs: (height, width), z.B. (1080, 1920).

    cell_shape : tuple
        Shape der periodischen Einheitszelle: (height, width).

    spots : list of tuple
        Gewünschte Beugungsordnungen relativ zur 0. Ordnung.
        Beispiel: (1, 0), (-1, 0), (0, 1), ...

    weights : list or None
        Relative Spot-Intensitäten.

    iterations : int
        Anzahl der Gerchberg-Saxton-Iterationen.

    seed : int or None
        Zufallsseed.

    crop : bool
        Wenn True, wird die Maske auf slm_shape zugeschnitten.
        Wenn False, muss slm_shape exakt durch cell_shape teilbar sein.

    Returns
    -------
    phase_cell : ndarray
        Optimierte Einheitszelle in Radiant.

    phase_slm : ndarray
        Periodische SLM-Maske mit Shape slm_shape.

    target_amplitude : ndarray
        Zielamplitude in der Fourier-Ebene der Einheitszelle.
    """

    slm_y, slm_x = slm_shape
    
    gcd = np.gcd(slm_x,slm_y)
    cell_shape = (gcd, gcd)

    cell_y, cell_x = cell_shape

    if not crop:
        if slm_y % cell_y != 0 or slm_x % cell_x != 0:
            raise ValueError(
                "slm_shape muss exakt durch cell_shape teilbar sein, "
                "oder crop=True verwenden."
            )

    rng = np.random.default_rng(seed)

    cy, cx = cell_y // 2, cell_x // 2

    if weights is None:
        weights = np.ones(len(spots), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    if len(weights) != len(spots):
        raise ValueError("weights muss dieselbe Länge wie spots haben.")

    # Zielintensitäten -> Zielamplituden
    weights = np.sqrt(weights / np.max(weights))

    target_amplitude = np.zeros(cell_shape, dtype=float)

    for (mx, my), amp in zip(spots, weights):
        y = cy + my
        x = cx + mx

        if not (0 <= y < cell_y and 0 <= x < cell_x):
            raise ValueError(
                f"Spot {(mx, my)} liegt außerhalb der Fourier-Ebene "
                f"der Einheitszelle {cell_shape}."
            )

        target_amplitude[y, x] = amp

    # Startphase der Einheitszelle
    phase = rng.uniform(0, 2*np.pi, cell_shape)
    field = np.exp(1j * phase)

    for _ in range(iterations):
        farfield = np.fft.fftshift(np.fft.fft2(field))

        farfield_phase = np.angle(farfield)
        farfield = target_amplitude * np.exp(1j * farfield_phase)

        field_back = np.fft.ifft2(np.fft.ifftshift(farfield))

        phase = np.angle(field_back)
        field = np.exp(1j * phase)

    phase_cell = np.mod(phase, 2*np.pi)

    # So oft tilen, dass das gesamte SLM bedeckt ist
    reps_y = int(np.ceil(slm_y / cell_y))
    reps_x = int(np.ceil(slm_x / cell_x))

    phase_slm_large = np.tile(phase_cell, (reps_y, reps_x))

    phase_slm = phase_slm_large[:slm_y, :slm_x]

    return phase_cell, phase_slm, target_amplitude


def saveJson(data, path):
    """Save a dictionary as JSON at the given path."""
    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary.")

    def _json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, default=_json_default)


def saveNpz(data, path):
    """Save a dictionary of array-like data as one compressed NumPy file."""
    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **data)


def createCalibrationLogDir(log_root="log"):
    """Create and return a unique log directory for one calibration run."""
    log_root = Path(log_root)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = f"calibration_{timestamp}"

    for suffix in range(1000):
        directory_name = base_name if suffix == 0 else f"{base_name}_{suffix:02d}"
        log_directory = log_root / directory_name
        try:
            log_directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return log_directory

    raise RuntimeError(f"Could not create a unique calibration log directory in {log_root}.")


def loadNpz(path=None):
    """Load a NumPy archive into a plain dictionary.

    If path is None, a file dialog is opened to select an .npz file.
    """
    if path is None:
        try:
            from tkinter import Tk, filedialog
        except ImportError as exc:
            raise RuntimeError("tkinter is required to open a file dialog.") from exc

        root = Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Load NumPy archive",
            filetypes=(("NumPy archives", "*.npz"), ("All files", "*.*")),
        )
        root.destroy()

        if not path:
            raise FileNotFoundError("No file was selected.")

    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def getTimestamp():
    """Return the current timestamp as DD-MM-YYYY-HH-MM."""
    return datetime.now().strftime("%d-%m-%Y-%H-%M")


if __name__ == "__main__": 
    
    resolution = (1080, 1920)
    # Beispielaufruf: berechne Patchgrößen für pt3 (asymmetrischer Pitch)
    pt3_resolution = (1440, 7680)
    pt3_pitch = (25e-6, 75e-6)
    sizes_pt3 = compute_patch_size_for_physical_square(pt3_resolution, pt3_pitch, max_patches_y=5, max_patches_x=15)
    print(f"pt3 patch suggestion: {sizes_pt3}")
    print(f"pt3 physical aspect ratio: {sizes_pt3['physical_aspect_ratio']:.3f}")
    
    # _, phase, _ = periodic_gs_multispot_slm(resolution)
    
    try:
        data = loadNpz()
    except FileNotFoundError:
        print("No .npz calibration file found; skipping downstream demo plotting.")
        plt.close("all")
        raise SystemExit(0)

    resolution = data["phase"].shape
    # S = int(np.asarray(data["S"]).item()) if "S" in data else np.gcd(resolution[0], resolution[1])
    # mosaic_mask = make_patch_mosaic_mask(data["best_alphas"], data["best_betas"], S, 8e-6, data["u0"], data["v0"], resolution=resolution)
    # gx_map, gy_map = data["gx"], data["gy"]
    
    # gx_raw = data["gradients"][...,0]
    # gy_raw = data["gradients"][...,1]
   
    # # phase, gx, gy = sphericalPhaseTestMap(resolution, 8e-6)
    # # phase -= phase.mean()
    # # plotUtils.plot_phase_gradient(phase, gx, gy, title="Test Input")
    
    # # dx = 8e-6
    # # phi = poisson_reconstruct_phase(gx, gy, 8e-6)
    
    # # phase, gx, gy = sphericalPhaseTestMap(resolution, 8e-6, exact=False)
    # # phase -= phase.mean()
    # # plotUtils.plot_phase_gradient(phase, gx, gy, title="Test Input")

    # phi = poisson_reconstruct_phase_direct_Fourier_Integration(gx_map, gy_map, 8e-6)
    # # phi -= phi.mean()
    # plotUtils.plot_camImg(gx_raw-gx_raw.mean(), "Gradient X")
    # plotUtils.plot_camImg(gy_raw-gy_raw.mean(), "Gradient Y")
    
    # plotUtils.plot_camImg(gx_map, "Gradient X interp")
    # plotUtils.plot_camImg(gy_map, "Gradient Y interp")



    # gx_phi, gy_phi = phaseMaskToGradients(phi, 8e-6, False)
    
    # errGrad = np.sqrt((gy_map-gy_phi)**2+(gx_map-gx_phi)**2)
    # plotUtils.plot_camImg(errGrad, "Error Grad")
    
    # # plotUtils.plot_wavefront(mosaic_mask, title= "mosaic")
    # # plotUtils.plot_wavefront(data["gx"], title = "gx")
    # # plotUtils.plot_wavefront(data["gy"], title = "gy")
    # # plotUtils.plot_wavefront(phi, title="phase")
    # plotUtils.plot_phase_gradient(phi, gx_map, gy_map, title="Recon")
    #patch = make_patch_ramp(1024,1024, 64, 10,10, 0.1/10e-6,0/10e-6, 10e-6)

    # Beispiel: Zernike-Fit direkt aus den gemessenen Patch-Gradienten.
    # Fuer pt3 sind die Pixel asymmetrisch; ggf. hier die kalibrierten Werte einsetzen.
    slm_pitch_zernike = (25e-6, 75e-6)
    zernike_amplitudes = None
    if "A_patch" in data:
        zernike_amplitudes = data["A_patch"]
    elif "power_patch" in data:
        zernike_amplitudes = np.sqrt(np.maximum(data["power_patch"], 0))
    S = data["S"]
    print(S)
    zernike_fit = fit_zernike_from_gradients(
        data["gradients"],
        resolution=resolution,
        patch_size=S,
        slm_pitch=slm_pitch_zernike,
        zernike_indices=range(2, 16),
        amplitudes=zernike_amplitudes,
        aperture_mode="full_slm_diagonal",
        amplitude_threshold=0.05 if zernike_amplitudes is not None else None,
        regularization=0.0,
        return_details=True,
    )
    Q, P = data["gx"].shape
    gx_raw, gy_raw = patch_gradients_to_maps(data["gradients"],P,Q,S)

    phase_zernike = zernike_fit["phase"]
    residuals_zernike = zernike_fit["residuals"]
    gx_zernike, gy_zernike = phaseMaskToGradients(
        phase_zernike,
        slm_pitch_zernike,
        unwrap=False,
    )
    residual_mag = np.sqrt(
        residuals_zernike[..., 0] ** 2
        + residuals_zernike[..., 1] ** 2
    )

    print("Zernike coefficients [rad]:")
    for fringe_index, coefficient in zernike_fit["coefficients"].items():
        coefficient_waves = zernike_fit["coefficients_waves"][fringe_index]
        print(f"  fringe {fringe_index:2d}: {coefficient: .6e} rad ({coefficient_waves: .6e} waves)")
    print(f"Zernike gradient residual RMS: {zernike_fit['residual_rms']:.6e} rad/m")

    plotUtils.plot_wavefront(phase_zernike, title="Zernike fitted phase")

    plotUtils.plot_wavefront(phase_zernike%(2*np.pi), title = "phase")
    
    plotUtils.plot_phase_gradient(
        phase_zernike,
        gx_raw,
        gy_raw,
        title="Zernike phase with measured gradients",
    )

    plotUtils.plot_phase_gradient(
        phase_zernike,
        gx_zernike,
        gy_zernike,
        title="Zernike fitted phase gradient",
    )

    plotUtils.plot_phase_gradient(
        phase_zernike,
        gx_raw,
        gy_raw,
        title="Zernike phase with measured gradients",
    )
    plotUtils.plot_camImg(
        residuals_zernike[..., 0],
        title="Zernike residual gx on patches",
    )
    plotUtils.plot_camImg(
        residuals_zernike[..., 1],
        title="Zernike residual gy on patches",
    )
    plotUtils.plot_camImg(
        residual_mag,
        title="Zernike residual magnitude on patches",
    )
    
    
    plt.show()
