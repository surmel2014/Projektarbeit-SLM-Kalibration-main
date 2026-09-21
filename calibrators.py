import hardware
import calcUtils
import plotUtils
import contextlib
import functools
import inspect
import numpy as np
import matplotlib.pyplot as plt
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from math import gcd
from scipy.ndimage import gaussian_filter


def _normalize_roi(roi, image_shape=None):
    y0, y1, x0, x1 = np.rint(np.asarray(roi, dtype=float)).astype(int)
    y0, y1 = sorted((y0, y1))
    x0, x1 = sorted((x0, x1))

    if image_shape is not None:
        rows, cols = image_shape
        y0 = int(np.clip(y0, 0, rows))
        y1 = int(np.clip(y1, 0, rows))
        x0 = int(np.clip(x0, 0, cols))
        x1 = int(np.clip(x1, 0, cols))

    if y1 <= y0 or x1 <= x0:
        raise ValueError("ROI must have a positive width and height.")

    return (y0, y1, x0, x1)


class _TeeStream:
    """Write console output to the terminal and a calibration log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def fileno(self):
        return self.streams[0].fileno()

    def isatty(self):
        return self.streams[0].isatty()

    @property
    def encoding(self):
        return self.streams[0].encoding

    @property
    def errors(self):
        return self.streams[0].errors


def _iso_timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_compatible(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, range):
        return list(value)
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _write_calibration_status(
    log_directory,
    status,
    started_at,
    finished_at=None,
    duration_seconds=None,
    error=None,
):
    status_data = {
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration_seconds,
    }
    if error is not None:
        status_data["error_type"] = type(error).__name__
        status_data["error_message"] = str(error)
    if status == "completed":
        status_data["artifacts"] = sorted(
            str(path.relative_to(log_directory))
            for path in log_directory.rglob("*")
            if path.is_file() and path.name != "status.json"
        )
    calcUtils.saveJson(status_data, log_directory / "status.json")


def _calibration_run(method):
    """Create a run directory and maintain status/console logs around a calibration."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        log_directory = calcUtils.createCalibrationLogDir()
        started_at = _iso_timestamp()
        start_time = time.perf_counter()

        bound = inspect.signature(method).bind(self, *args, **kwargs)
        bound.apply_defaults()
        call_parameters = {
            key: _json_compatible(value)
            for key, value in bound.arguments.items()
            if key != "self"
        }
        initial_metadata = {
            "schema_version": 1,
            "calibration_method": method.__name__,
            "started_at": started_at,
            "parameters": call_parameters,
        }
        calcUtils.saveJson(initial_metadata, log_directory / "metadata.json")
        _write_calibration_status(log_directory, "running", started_at)

        self._active_calibration_log_directory = log_directory
        self._active_calibration_metadata = initial_metadata

        with (log_directory / "run.log").open("a", encoding="utf-8", buffering=1) as log_file:
            stdout_tee = _TeeStream(sys.stdout, log_file)
            stderr_tee = _TeeStream(sys.stderr, log_file)
            with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
                print(f"Calibration log directory: {log_directory}")
                try:
                    result = method(self, *args, **kwargs)
                except BaseException as exc:
                    finished_at = _iso_timestamp()
                    duration = time.perf_counter() - start_time
                    print(f"Calibration failed after {duration:.3f} s: {exc}")
                    traceback.print_exc()
                    _write_calibration_status(
                        log_directory,
                        "failed",
                        started_at,
                        finished_at=finished_at,
                        duration_seconds=duration,
                        error=exc,
                    )
                    raise
                else:
                    finished_at = _iso_timestamp()
                    duration = time.perf_counter() - start_time
                    _write_calibration_status(
                        log_directory,
                        "completed",
                        started_at,
                        finished_at=finished_at,
                        duration_seconds=duration,
                    )
                    print(f"Calibration completed in {duration:.3f} s")
                    return result
                finally:
                    self._active_calibration_log_directory = None
                    self._active_calibration_metadata = None

    return wrapper


def _add_zernike_fit(data):
    """Fit Zernike modes 2..15 and attach coefficients and diagnostics."""
    try:
        zernike_fit = calcUtils.fit_zernike_from_gradients(
            data["gradients"],
            resolution=np.asarray(data["phase"]).shape,
            patch_size=data["S"],
            slm_pitch=data["slm_pitch"],
            zernike_indices=range(2, 16),
            amplitudes=data.get("A_patch"),
            aperture_mode="full_slm_diagonal",
            amplitude_threshold=0.05,
            return_details=True,
        )
    except Exception as exc:
        data["zernike_fit_error"] = str(exc)
        print(f"Zernike fit skipped: {exc}")
        return None

    indices = np.asarray(zernike_fit["zernike_indices"], dtype=int)
    coefficients = np.asarray(
        [zernike_fit["coefficients"][int(index)] for index in indices],
        dtype=float,
    )
    coefficients_waves = np.asarray(
        [zernike_fit["coefficients_waves"][int(index)] for index in indices],
        dtype=float,
    )
    names = np.asarray(
        [
            calcUtils.ZernPol(fringe=int(index)).get_polynomial_name(short=True)
            or f"Fringe {index}"
            for index in indices
        ]
    )

    data.update(
        {
            "zernike_phase": zernike_fit["phase"],
            "zernike_residuals": zernike_fit["residuals"],
            "zernike_gx_fit": zernike_fit["gx_fit"],
            "zernike_gy_fit": zernike_fit["gy_fit"],
            "zernike_valid_mask": zernike_fit["valid_mask"],
            "zernike_indices": indices,
            "zernike_names": names,
            "zernike_coefficients": coefficients,
            "zernike_coefficients_waves": coefficients_waves,
            "zernike_residual_rms": zernike_fit["residual_rms"],
            "zernike_aperture_radius": zernike_fit["aperture"]["radius"],
            "zernike_aperture_center": zernike_fit["aperture"]["center"],
        }
    )
    return zernike_fit


def _build_calibration_metadata(data, log_directory):
    parameter_keys = (
        "S",
        "slm_pitch",
        "u0",
        "v0",
        "roi",
        "focal_length",
        "wavelength",
        "cam_pitch",
        "eps_px",
        "max_iter",
        "settle_s",
        "discard_frames",
        "camera_frames",
        "live_view",
        "live_every",
        "skip_gradient_search",
        "probe_px",
        "probe_freq",
        "max_step_px",
        "damping",
        "useCorrection",
        "adaptive_gradient_exposure",
        "separate_amplitude_measurement",
        "amplitude_exposure_min_us",
        "amplitude_exposure_max_us",
        "amplitude_target_low",
        "amplitude_target_high",
        "amplitude_exposure_factor",
        "amplitude_max_attempts",
        "amplitude_saturation_value",
        "exposure_smoothing_sigma",
    )
    com_patch = np.asarray(data["com_patch"])
    valid_patches = np.all(np.isfinite(com_patch), axis=-1)
    metadata = {
        "schema_version": 1,
        "calibration_method": data.get("calibration_method", "wavefront"),
        "run_directory": log_directory.name,
        "started_at": data.get("started_at"),
        "parameters": {
            key: _json_compatible(data[key])
            for key in parameter_keys
            if key in data
        },
        "dimensions": {
            "slm_resolution": list(np.asarray(data["phase"]).shape),
            "patch_grid": list(np.asarray(data["A_patch"]).shape),
        },
        "hardware": {
            "display_type": data.get("display_type"),
            "simulated": bool(data.get("is_simulative", False)),
        },
        "results": {
            "valid_patch_count": int(np.count_nonzero(valid_patches)),
            "total_patch_count": int(valid_patches.size),
            "reference_com_px": _json_compatible(data["com_ref"]),
            "reference_power": _json_compatible(data["power_ref"]),
        },
    }
    if data.get("separate_amplitude_measurement", False):
        amplitude_valid = np.asarray(
            data.get("amplitude_measurement_valid", []),
            dtype=bool,
        )
        metadata["results"]["valid_amplitude_measurement_count"] = int(
            np.count_nonzero(amplitude_valid)
        )
        metadata["results"]["patch_power_unit"] = "a.u./us"
    else:
        metadata["results"]["patch_power_unit"] = (
            "a.u./us"
            if data.get("adaptive_gradient_exposure", False)
            else "a.u."
        )
    if data.get("adaptive_gradient_exposure", False):
        gradient_exposure_valid = np.asarray(
            data.get("gradient_exposure_valid", []),
            dtype=bool,
        )
        metadata["results"]["valid_gradient_exposure_count"] = int(
            np.count_nonzero(gradient_exposure_valid)
        )
        metadata["results"]["reference_exposure_us"] = _json_compatible(
            data.get("reference_exposure_us")
        )
        metadata["results"]["reference_exposure_valid"] = bool(
            data.get("reference_exposure_valid", False)
        )
    if "zernike_indices" in data:
        metadata["zernike_fit"] = {
            "index_convention": "Fringe",
            "indices": _json_compatible(data["zernike_indices"]),
            "names": _json_compatible(data["zernike_names"]),
            "coefficients_rad": _json_compatible(data["zernike_coefficients"]),
            "coefficients_waves": _json_compatible(data["zernike_coefficients_waves"]),
            "gradient_residual_rms_rad_per_m": _json_compatible(
                data["zernike_residual_rms"]
            ),
        }
    elif "zernike_fit_error" in data:
        metadata["zernike_fit"] = {"error": str(data["zernike_fit_error"])}
    return metadata


def _save_calibration_outputs(data, log_directory):
    """Save numerical calibration data and the complete diagnostic plot suite."""
    convergence = np.asarray(data.get("convergence", []), dtype=float)
    data["convergence"] = convergence.reshape(-1, 4)

    zernike_fit = _add_zernike_fit(data)

    np.save(log_directory / "phase.npy", data["phase"])
    np.save(log_directory / "background.npy", data["background_image"])
    np.save(log_directory / "reference.npy", data["reference_image"])
    calcUtils.saveNpz(
        {
            key: np.asarray(data[key])
            for key in (
                "A_patch",
                "power_patch",
                "amplitude_power_raw",
                "amplitude_exposure_us",
                "amplitude_peak_fraction",
                "amplitude_saturation_fraction",
                "amplitude_measurement_valid",
                "gradient_exposure_us",
                "gradient_peak_fraction",
                "gradient_saturation_fraction",
                "gradient_exposure_valid",
            )
            if key in data
        },
        log_directory / "A_patch.npz",
    )
    if zernike_fit is not None:
        np.save(log_directory / "zernike_phase.npy", data["zernike_phase"])
        calcUtils.saveNpz(
            {
                "phase": data["zernike_phase"],
                "residuals": data["zernike_residuals"],
                "gx_fit": data["zernike_gx_fit"],
                "gy_fit": data["zernike_gy_fit"],
                "valid_mask": data["zernike_valid_mask"],
                "indices": data["zernike_indices"],
                "names": data["zernike_names"],
                "coefficients_rad": data["zernike_coefficients"],
                "coefficients_waves": data["zernike_coefficients_waves"],
                "residual_rms": data["zernike_residual_rms"],
            },
            log_directory / "zernike_fit.npz",
        )

    calcUtils.saveNpz(data, log_directory / "data.npz")
    plot_paths = plotUtils.save_calibration_plots(data, log_directory)
    shutil.copyfile(plot_paths[0], log_directory / "diagnostics.png")
    metadata = _build_calibration_metadata(data, log_directory)
    calcUtils.saveJson(metadata, log_directory / "metadata.json")
    print(f"Saved calibration data and {len(plot_paths)} plots to {log_directory}")


class LiveCalibrationView:
    def __init__(self, cam_extent=None, roi=None):
        plt.ion()

        self.fig, (self.ax_slm, self.ax_cam) = plt.subplots(1, 2, figsize=(10, 4))
        self.fig.suptitle("Wavefront calibration live view")

        self.ax_slm.set_title("SLM phase")
        self.ax_cam.set_title("Camera")

        self.slm_image = None
        self.cam_image = None
        self.cam_extent = cam_extent
        self.roi_patch = None
        self.slm_patch_center_marker = None
        self.cam_com_marker = None
        self.cam_ref_com_marker = None

        if roi is not None:
            y0, y1, x0, x1 = roi
            self.roi_patch = Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="white",
                linewidth=1.5,
            )
            self.ax_cam.add_patch(self.roi_patch)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def set_roi(self, roi):
        y0, y1, x0, x1 = roi
        if self.roi_patch is None:
            self.roi_patch = Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="white",
                linewidth=1.5,
            )
            self.ax_cam.add_patch(self.roi_patch)
        else:
            self.roi_patch.set_xy((x0, y0))
            self.roi_patch.set_width(x1 - x0)
            self.roi_patch.set_height(y1 - y0)
            self.roi_patch.set_visible(True)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def select_roi(self, image_shape=None, initial_roi=None):
        selected_roi = {"value": None}
        selection_state = {"finished": False, "closed": False}

        if initial_roi is not None:
            self.set_roi(initial_roi)

        def finish_selection():
            selection_state["finished"] = True
            self.fig.canvas.stop_event_loop()

        def onselect(eclick, erelease):
            if eclick.xdata is None or eclick.ydata is None:
                return
            if erelease.xdata is None or erelease.ydata is None:
                return

            roi = _normalize_roi(
                (eclick.ydata, erelease.ydata, eclick.xdata, erelease.xdata),
                image_shape,
            )
            selected_roi["value"] = roi
            self.set_roi(roi)

        def on_key_press(event):
            if event.key in ("enter", "return"):
                finish_selection()
            elif event.key == "escape":
                selected_roi["value"] = None
                finish_selection()

        def on_close(_event):
            selection_state["closed"] = True
            finish_selection()

        selector_kwargs = dict(
            useblit=True,
            button=[1],
            minspanx=2,
            minspany=2,
            spancoords="data",
            interactive=True,
        )
        selector_props = dict(facecolor="none", edgecolor="white", linewidth=1.5)
        try:
            selector = RectangleSelector(
                self.ax_cam,
                onselect,
                props=selector_props,
                **selector_kwargs,
            )
        except TypeError as exc:
            if "props" not in str(exc):
                raise
            selector = RectangleSelector(
                self.ax_cam,
                onselect,
                rectprops=selector_props,
                **selector_kwargs,
            )

        previous_title = self.fig._suptitle.get_text() if self.fig._suptitle else ""
        self.fig.suptitle(
            "Wavefront calibration - draw ROI, then press Enter in this window"
        )
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        key_connection = self.fig.canvas.mpl_connect("key_press_event", on_key_press)
        close_connection = self.fig.canvas.mpl_connect("close_event", on_close)
        print("Draw the ROI in the live view and press Enter in the plot window.")

        try:
            while (
                not selection_state["finished"]
                and plt.fignum_exists(self.fig.number)
            ):
                self.fig.canvas.start_event_loop(0.1)
        finally:
            selector.set_active(False)
            selector.disconnect_events()
            self.fig.canvas.mpl_disconnect(key_connection)
            self.fig.canvas.mpl_disconnect(close_connection)
            if plt.fignum_exists(self.fig.number):
                self.fig.suptitle(previous_title)

        if selected_roi["value"] is None:
            if initial_roi is None:
                if selection_state["closed"]:
                    raise RuntimeError("The live view was closed before an ROI was selected.")
                raise RuntimeError("No ROI was selected.")
            selected_roi["value"] = initial_roi

        if plt.fignum_exists(self.fig.number):
            self.set_roi(selected_roi["value"])
        return selected_roi["value"]

    def update(
        self,
        holo=None,
        cam_img=None,
        title=None,
        cam_com=None,
        patch_center=None,
        cam_ref_com=None,
        roi=None,
    ):
        if holo is not None:
            slm_preview = np.mod(np.asarray(holo), 2 * np.pi)
            if self.slm_image is None:
                self.slm_image = self.ax_slm.imshow(slm_preview, cmap="twilight")
                self.fig.colorbar(self.slm_image, ax=self.ax_slm, fraction=0.046, pad=0.04)
            else:
                self.slm_image.set_data(slm_preview)
                self.slm_image.set_clim(float(np.nanmin(slm_preview)), float(np.nanmax(slm_preview)))

        if patch_center is not None:
            if self.slm_patch_center_marker is None:
                (self.slm_patch_center_marker,) = self.ax_slm.plot(
                    patch_center[0],
                    patch_center[1],
                    marker="+",
                    color="white",
                    markersize=14,
                    markeredgewidth=2,
                    linestyle="None",
                )
            else:
                self.slm_patch_center_marker.set_data([patch_center[0]], [patch_center[1]])
            self.slm_patch_center_marker.set_visible(True)
        elif self.slm_patch_center_marker is not None:
            self.slm_patch_center_marker.set_visible(False)

        if cam_img is not None:
            cam_preview = np.asarray(cam_img)
            finite_display = cam_preview[np.isfinite(cam_preview)]
            if finite_display.size:
                display_min = float(np.min(finite_display))
                display_max = float(np.max(finite_display))
                if display_max <= display_min:
                    display_max = display_min + 1.0
            else:
                display_min, display_max = 0.0, 1.0
            if self.cam_image is None:
                self.cam_image = self.ax_cam.imshow(
                    cam_preview,
                    cmap="nipy_spectral",
                    origin="lower",
                    extent=self.cam_extent,
                    vmin=display_min,
                    vmax=display_max,
                )
                self.fig.colorbar(self.cam_image, ax=self.ax_cam, fraction=0.046, pad=0.04)
            else:
                self.cam_image.set_data(cam_preview)
                self.cam_image.set_clim(display_min, display_max)

        if cam_com is not None:
            if self.cam_com_marker is None:
                (self.cam_com_marker,) = self.ax_cam.plot(
                    cam_com[0],
                    cam_com[1],
                    marker="+",
                    color="white",
                    markersize=14,
                    markeredgewidth=2,
                    linestyle="None",
                )
            else:
                self.cam_com_marker.set_data([cam_com[0]], [cam_com[1]])
            self.cam_com_marker.set_visible(True)
        elif self.cam_com_marker is not None:
            self.cam_com_marker.set_visible(False)

        if cam_ref_com is not None:
            if self.cam_ref_com_marker is None:
                (self.cam_ref_com_marker,) = self.ax_cam.plot(
                    cam_ref_com[0],
                    cam_ref_com[1],
                    marker="x",
                    color="red",
                    markersize=12,
                    markeredgewidth=2,
                    linestyle="None",
                )
            else:
                self.cam_ref_com_marker.set_data([cam_ref_com[0]], [cam_ref_com[1]])
            self.cam_ref_com_marker.set_visible(True)


        if roi is not None:
            self.set_roi(roi)


        if title is not None:
            self.fig.suptitle(title)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

class Server:
    def __init__(self, display_type = "pt3", isSimulative = False, useCam=True):
        
        self.waveLength = 633e-9
        
        self.isSimulative = isSimulative
        if useCam:
            self.CAM = hardware.CAM(isSimulative=isSimulative)
        self.SLM = hardware.SLM(display_type=display_type, isSimulative=isSimulative)
        self.display_type = display_type
    
    def getCamImg(self):

        return self.CAM.getCamImg()

    def getSettledCamImg(self, settle_s=0.1, discard_frames=2):
        """Wait for the SLM/camera pipeline to settle, then return a fresh image."""
        time.sleep(settle_s)
        for _ in range(discard_frames):
            self.getCamImg()
        return self.getCamImg()
    
    def showHologram(self, holo, amp= None, use_correction=True):
        if self.display_type == "pt3":
            self.SLM.showStackedField(phase= holo, amp = amp)
        else:
            self.SLM.showHologram(holo, use_correction=use_correction)
    
    def showCamImg(self):
        img = self.getCamImg()
        
        title = "Cam Image"
        if self.isSimulative:
            title += " - Simulative"
        if self.isSimulative and self.CAM.simExtent is not None:
            extent = self.CAM.simExtent
        else:
            rows, cols = img.shape
            extent = [0, cols*self.CAM.pitch, 0, rows*self.CAM.pitch]
        
        plotUtils.plot_camImg(img, title, extent)
        
        
    def simulatePropagation(self):
        
        gaussianBeam = calcUtils.getGaussianBeam(self.SLM.resolution, 0.005, self.waveLength, pixelPitch=self.SLM.pitch)
        holo = self.SLM.currentHolo   
        
        field = gaussianBeam*np.exp(1j*holo)
        slm_pitch = self.SLM.pitch
        if np.isscalar(slm_pitch):
            x_pitch = y_pitch = float(slm_pitch)
        else:
            x_pitch, y_pitch = slm_pitch

        propField, xFocal, yFocal = calcUtils.propagateField(
            field,
            x_pitch,
            y_pitch,
            self.waveLength,
            focalLength=0.1,
            outputPitch=1e-6,
            outputResolution=(256, 256),
        )
        print(f"focal Pitch is {xFocal[1]-xFocal[0]}")
        intensity = np.abs(propField)**2
        
        
        waveFrontError = calcUtils.wavefront_from_fringe({})
        
        
        self.CAM.simImg = intensity
        self.CAM.simExtent = [xFocal.min(), xFocal.max(), yFocal.min(), yFocal.max()]

    
    def showStandardPatches(self):
        Q,P = self.SLM.resolution
        
      
        S = gcd(P, Q)

        if hasattr(self, "CAM"):
            self.CAM.nFrames = 1

        slm_pitch = self.SLM.pitch
        if np.isscalar(slm_pitch):
            x_pitch = y_pitch = float(slm_pitch)
        else:
            x_pitch, y_pitch = slm_pitch

        u0 = 0.1 / x_pitch
        v0 = -0.1 / y_pitch

        M = P // S
        N = Q // S

        # 1. Referenzpunkt: zentrales Patch
        m_ref, n_ref = M // 2, N // 2
        phase_ref = calcUtils.make_patch_ramp(P, Q, S, m_ref, n_ref, u0, v0, slm_pitch)
        self.showHologram(phase_ref)
        time.sleep(0.01)

        for n in range(N):
            for m in range(M):
                # Iterative Schätzung der Gegensteigung: alpha_tilde, beta_tilde in [1/m]
                alpha_tilde = 0.0
                beta_tilde = 0.0

                # Secant-artige Historie getrennt für x/y
                hist = []
                best_err = np.inf
                best_alpha_tilde = alpha_tilde
                best_beta_tilde = beta_tilde
                best_power = 0.0
                worsening_steps = 0

                
                
                phase = calcUtils.make_patch_ramp(
                    P, Q, S, m, n,
                    u0 - alpha_tilde,
                    v0 - beta_tilde,
                    slm_pitch,
                )
                self.showHologram(phase)
                time.sleep(0.01)


    @_calibration_run
    def calibrate_wavefront(
        self,
        S="GCD",
        focal_length=0.06,
        u0=None,
        v0=None,
        roi=(500, 1500, 300, 1700),
        eps_px=0.1,
        max_iter=6,
        settle_s=0.1,
        discard_frames=2,
        camera_frames=3,
        live_view=True,
        live_every=1,
        skip_gradient_search=False,
    ):
        """
        Gibt zurück:
        A_map: rekonstruierte Amplitude auf SLM-Pixelgitter
        phi_map: rekonstruierte Phase auf SLM-Pixelgitter
        gradients: lokale Gradienten pro Patch [rad/m]
        A_patch: lokale Amplituden pro Patch

        Wenn skip_gradient_search=True:
        Gradienten werden direkt aus den COM-Abweichungen zum Referenz-COM
        berechnet, ohne iterative Spot-Zentrierung.
        """
        #
        # self.CAM.set_integration_time(3)
        
        old_camera_frames = None
        Q,P = self.SLM.resolution
        
        if S == "GCD":
            S = gcd(P, Q)

        if settle_s < 0:
            raise ValueError("settle_s must be greater than or equal to 0.")
        if discard_frames < 0:
            raise ValueError("discard_frames must be greater than or equal to 0.")
        if camera_frames <= 0:
            raise ValueError("camera_frames must be greater than 0.")

        log_directory = self._active_calibration_log_directory

        if hasattr(self, "CAM"):
            old_camera_frames = self.CAM.nFrames
            self.CAM.nFrames = camera_frames

        roi = _normalize_roi(roi)

        slm_pitch = self.SLM.pitch
        if np.isscalar(slm_pitch):
            x_pitch = y_pitch = float(slm_pitch)
        else:
            x_pitch, y_pitch = slm_pitch

        live = None
        if live_view:
            live = LiveCalibrationView(roi=roi)

        if u0 is None:
            u0 = 2000 #-0.1 / x_pitch
        if v0 is None:
            v0 = 0 #0.1 / y_pitch

        M = P // S
        N = Q // S

        # 0. Capture the zeroth-order/background image.
        black = np.zeros((Q, P)) + 1e-16
        if self.display_type == "pt3":
            self.SLM.showStackedField(field=[black, black])
        else:
            self.showHologram(black, use_correction=False)

        thresh_zerothorder = self.getSettledCamImg(settle_s, discard_frames)

        if live is not None:
            live.update(
                black,
                thresh_zerothorder,
                "Wavefront calibration - background",
                roi=roi,
            )


        # 1. Referenzpunkt: zentrales Patch
        m_ref, n_ref = M // 2, N // 2
        phase_ref, amp_ref = calcUtils.make_patch_ramp(P, Q, S, m_ref, n_ref, u0, v0, slm_pitch)
   
        self.showHologram(phase_ref, amp_ref)
        
        while True:
            img_ref = self.getSettledCamImg(settle_s, discard_frames) - thresh_zerothorder
            if live is not None:
                live.update(
                    phase_ref,
                    img_ref,
                    "Wavefront calibration - reference patch",
                    patch_center=((m_ref + 0.5) * S, (n_ref + 0.5) * S),
                    roi=roi,
                )
                roi = live.select_roi(img_ref.shape, roi)
                #int_time = float(input("input integration time [µs]"))
                #self.CAM.set_integration_time(int_time)
            else:
                roi = _normalize_roi(
                    np.fromstring(input("Give ROI y0, y1, x0, x1: "), sep=",", dtype=np.int32),
                    img_ref.shape,
                )

            com_ref, power_ref = calcUtils.spot_com_and_power(img_ref, roi)
            print(f"ref power: {power_ref}")
            if com_ref is not None:
                if live is not None:
                    live.update(
                        phase_ref,
                        img_ref,
                        "Wavefront calibration - reference patch",
                        cam_com=com_ref,
                        patch_center=((m_ref + 0.5) * S, (n_ref + 0.5) * S),
                        cam_ref_com=com_ref,
                        roi=roi,
                    )
                break

            print("Referenzspot nicht gefunden. Bitte ROI/Belichtung pruefen und ROI erneut waehlen.")

        gradients = np.zeros((N, M, 2), dtype=float)  # gx, gy in rad/m
        A_patch = np.zeros((N, M), dtype=float)

        # Hilfsfaktor: Kamerapixel -> Fourier-Ebene in Meter -> spatial frequency
        # δu = δx' / (f λ), δx' = pixel_shift * cam_pitch
        px_to_freq = self.CAM.pitch / (focal_length * self.waveLength)

        data = {
            "calibration_method": "calibrate_wavefront",
            "started_at": self._active_calibration_metadata["started_at"],
            "S": S,
            "slm_pitch": slm_pitch,
            "u0": u0,
            "v0": v0,
            "roi": np.asarray(roi),
            "com_ref": com_ref,
            "power_ref": power_ref,
            "skip_gradient_search": skip_gradient_search,
            "focal_length": focal_length,
            "wavelength": self.waveLength,
            "cam_pitch": self.CAM.pitch,
            "eps_px": eps_px,
            "max_iter": max_iter,
            "settle_s": settle_s,
            "discard_frames": discard_frames,
            "camera_frames": camera_frames,
            "live_view": live_view,
            "live_every": live_every,
            "display_type": self.display_type,
            "is_simulative": self.isSimulative,
            "background_image": thresh_zerothorder,
            "reference_image": img_ref,
            "convergence": [],
        }
        data["coms"] = []
        com_patch = np.full((N, M, 2), np.nan, dtype=float)
        power_patch = np.zeros((N, M), dtype=float)
        best_alphas = np.full((N, M), np.nan, dtype=float)
        best_betas = np.full((N, M), np.nan, dtype=float)
        # 2. Patch scanning
        for n in range(N):
            for m in range(M):
                # Iterative Schätzung der Gegensteigung: alpha_tilde, beta_tilde in [1/m]
                alpha_tilde = 0.0
                beta_tilde = 0.0

                # Secant-artige Historie getrennt für x/y
                hist = []
                best_err = np.inf
                best_alpha_tilde = alpha_tilde
                best_beta_tilde = beta_tilde
                best_power = 0.0
                best_com = None
                worsening_steps = 0

                iter_count = 1 if skip_gradient_search else max_iter
                for k in range(iter_count):
                    
                    phase, amp = calcUtils.make_patch_ramp(
                        P, Q, S, m, n,
                        u0 - alpha_tilde,
                        v0 - beta_tilde,
                        slm_pitch,
                    )
                    
                    
                    self.showHologram(phase, amp)
                    iter_start = time.time()

                    img = self.getSettledCamImg(settle_s, discard_frames) - thresh_zerothorder
                    com, power = calcUtils.spot_com_and_power(img, roi)
                    if com is not None:
                        com_patch[n, m] = com
                    data["coms"].append(com_patch[n, m].copy())
                    power_patch[n, m] = power
                    print(f"patch power: {power}")
                    #print(f"iter took {time.time()-iter_start}")
                    # if power < power_ref /1.7:
                    #     print("no Spot in ROI, skipping patch...")
                    #     hist.append((0, 0, 0))
                    #     break

                    if live is not None and k % live_every == 0:
                        live.update(
                            phase,
                            img,
                            f"Wavefront calibration - patch ({m}, {n}), iteration {k + 1}",
                            cam_com=com,
                            patch_center=((m + 0.5) * S, (n + 0.5) * S),
                        )
                    if com is None:
                        break

                    delta_px = com - com_ref
                    err = np.linalg.norm(delta_px)
                    data["convergence"].append((m, n, k, err))

                    if skip_gradient_search:
                        A_patch[n, m] = np.sqrt(power)
                        break

                    hist.append((alpha_tilde, beta_tilde, delta_px.copy()))

                    print(f"Pixel:({m},{n}) iteration {k}, err: {err}, eps_px: {eps_px}")

                    if err < best_err:
                        best_err = err
                        best_alpha_tilde = alpha_tilde
                        best_beta_tilde = beta_tilde
                        best_power = power
                        best_com = com.copy()
                        worsening_steps = 0
                    else:
                        worsening_steps += 1

                    if err <= eps_px:
                        break

                    if worsening_steps >= 2:
                        print(
                            f"error increased twice; using best iteration with err {best_err}"
                        )
                        break

                    # Schritt 2(b): direkte Korrektur aus Spot-Verschiebung
                    if k == 0:
                        alpha_tilde += delta_px[0] * px_to_freq
                        beta_tilde += delta_px[1] * px_to_freq
                    else:
                        # Vereinfachter Secant-Schritt statt Brent:
                        # getrennt für x und y, robust genug als Startimplementierung.
                        a0, b0, d0 = hist[-2]
                        a1, b1, d1 = hist[-1]

                        if abs(d1[0] - d0[0]) > 1e-9:
                            alpha_tilde = a1 - d1[0] * (a1 - a0) / (d1[0] - d0[0])
                        else:
                            alpha_tilde += delta_px[0] * px_to_freq

                        if abs(d1[1] - d0[1]) > 1e-9:
                            beta_tilde = b1 - d1[1] * (b1 - b0) / (d1[1] - d0[1])
                        else:
                            beta_tilde += delta_px[1] * px_to_freq

                if np.isfinite(best_err):
                    alpha_tilde = best_alpha_tilde
                    beta_tilde = best_beta_tilde
                    best_alphas[n, m] = best_alpha_tilde
                    best_betas[n, m] = best_beta_tilde
                    A_patch[n, m] = np.sqrt(best_power)
                    power_patch[n, m] = best_power
                    if best_com is not None:
                        com_patch[n, m] = best_com

                # final: g_mn = -g_tilde = -2π(alpha_tilde, beta_tilde)
                gradients[n, m, 0] = -2 * np.pi * alpha_tilde
                gradients[n, m, 1] = -2 * np.pi * beta_tilde

        data["best_alphas"] = best_alphas
        data["best_betas"] = best_betas
        data["gradients"] = gradients
        data["com_patch"] = com_patch
        data["power_patch"] = power_patch

        if skip_gradient_search:
            gx_map, gy_map, gradients = calcUtils.gradient_maps_from_com_deviation(
                com_patch,
                com_ref,
                focal_length=focal_length,
                wavelength=self.waveLength,
                cam_pitch=self.CAM.pitch,
                P=P,
                Q=Q,
                S=S,
                kind="linear",
                sign=-1.0,
                remove_mean=True,
                fill_missing=True,
            )

            A_patch /= np.max(A_patch) + 1e-12
            ##A_map = calcUtils.interpolate_patch_values(A_patch, P, Q, S, kind="cubic")
            phi_map = calcUtils.poisson_reconstruct_phase(gx_map, gy_map, slm_pitch)

            plotUtils.plot_wavefront(gx_map, title="gx")
            plotUtils.plot_wavefront(gy_map, title="gy")
            #plotUtils.plot_camImg(A_map, title="A_map")

            data["gradients"] = gradients
            data["A_patch"] = A_patch
            data["gx"] = gx_map
            data["gy"] = gy_map
            data["phase"] = phi_map
            _save_calibration_outputs(data, log_directory)
            if old_camera_frames is not None:
                self.CAM.nFrames = old_camera_frames
            return  phi_map, gradients, A_patch 


        A_patch /= np.max(A_patch) + 1e-12

        # 3. Rekonstruktion
        #A_map = calcUtils.interpolate_patch_values(A_patch, P, Q, S, kind="cubic")
        gx_map = calcUtils.interpolate_patch_values(gradients[..., 0], P, Q, S, kind="linear")
        gy_map = calcUtils.interpolate_patch_values(gradients[..., 1], P, Q, S, kind="linear")

        # Optional: globalen Tip/Tilt entfernen
        gx_map -= gx_map.mean()
        gy_map -= gy_map.mean()

        phi_map = calcUtils.poisson_reconstruct_phase(gx_map, gy_map, slm_pitch)

        plotUtils.plot_wavefront(gx_map, title="gx")
        plotUtils.plot_wavefront(gy_map, title="gy")
        #plotUtils.plot_camImg(A_map, title="A_map")

        data["gx"] = gx_map
        data["gy"] = gy_map

        data["phase"] = phi_map
        data["A_patch"] = A_patch
        
        _save_calibration_outputs(data, log_directory)

        # np.save("log/gx_map.npy", gx_map)
        # np.save("log/gy_map.npy", gy_map)
        # plt.imsave("log/Wavefront.png", calcUtils.wrap_phase(phi_map))
        # np.save("log/wavefrontmap.npy", phi_map)

        if old_camera_frames is not None:
            self.CAM.nFrames = old_camera_frames

        return phi_map, gradients, A_patch  #A_map, 
    

    @_calibration_run
    def calibrate_wavefront_pt3(
        self,
        S="GCD",
        focal_length=0.06,
        Sx=None,
        Sy=None,
        u0=None,
        v0=None,
        roi=(500, 1500, 300, 1700),
        eps_px=0.6,
        max_iter=6,
        settle_s=0.1,
        discard_frames=2,
        camera_frames=3,
        live_view=True,
        live_every=4,
        skip_gradient_search=False,
        probe_px=20.0,
        max_step_px=300.0,
        damping=1,
        useCorrection=False,
        adaptive_gradient_exposure=True,
        separate_amplitude_measurement=True,
        amplitude_exposure_min_us=50.0,
        amplitude_exposure_max_us=1_000_000.0,
        amplitude_target_low=0.10,
        amplitude_target_high=0.85,
        amplitude_exposure_factor=4.0,
        amplitude_max_attempts=8,
        amplitude_saturation_value=None,
        exposure_smoothing_sigma=2.0,
        ):
        """
        Wavefront-Kalibration mit gemessener lokaler 2x2-Jacobi-Matrix.

        Gibt zurück:
            phi_map: rekonstruierte Phase auf SLM-Pixelgitter [rad]
            gradients: lokale Patch-Gradienten [rad/m], shape (N, M, 2)
            A_patch: lokale Amplituden pro Patch

        Neu gegenüber vorher:
            - Die erste Korrekturrichtung wird nicht angenommen,
            sondern über Probe-Schritte gemessen.
            - x/y-Kopplung zwischen SLM-Frequenz und Kamera-Spotposition
            wird über eine 2x2-Jacobi-Matrix berücksichtigt.
            - Die Jacobi-Matrix wird während der Iteration per Broyden-Update
            verbessert.
            - Mit separate_amplitude_measurement=True wird nach dem Finden des
            finalen Patch-Gradienten eine adaptive, hintergrundkorrigierte und
            auf die Belichtungszeit normierte Amplitudenmessung ausgeführt.
              False verwendet wie bisher die Leistung aus der Positionssuche.
            - Mit adaptive_gradient_exposure=True wird die Belichtungszeit für
              den Referenzspot und einmal zu Beginn jedes neuen Patches adaptiv
              bestimmt. Probes und Newton-Iterationen verwenden danach dieselbe
              Patch-Belichtungszeit. Die Grenzwerte werden mit den
              amplitude_*-Parametern konfiguriert.
        """

        import time
        import numpy as np
        from math import gcd

        def _limited_step(step, max_norm):
            norm = np.linalg.norm(step)
            if norm > max_norm and norm > 0:
                step = step * (max_norm / norm)
            return step

        def _safe_newton_step(J, delta_px, max_step):
            """
            Gesucht ist step mit:
                J @ step ~= -delta_px

            J beschreibt:
                Änderung der Spotposition [px]
                pro Änderung von [alpha_tilde, beta_tilde] [1/m]
            """
            try:
                step = -np.linalg.solve(J, delta_px)
            except np.linalg.LinAlgError:
                step = -np.linalg.pinv(J) @ delta_px

            return _limited_step(step, max_step)

        old_camera_frames = None
        old_integration_time = None
        if useCorrection:
            correctionPhase = np.load("log/wavefrontmap_pt3.npy")

        Q, P = self.SLM.resolution

        if S == "GCD":
            S = gcd(P, Q)

        if settle_s < 0:
            raise ValueError("settle_s must be greater than or equal to 0.")
        if discard_frames < 0:
            raise ValueError("discard_frames must be greater than or equal to 0.")
        if camera_frames <= 0:
            raise ValueError("camera_frames must be greater than 0.")
        if probe_px <= 0:
            raise ValueError("probe_px must be greater than 0.")
        if max_step_px <= 0:
            raise ValueError("max_step_px must be greater than 0.")
        if amplitude_exposure_min_us <= 0:
            raise ValueError("amplitude_exposure_min_us must be greater than 0.")
        if amplitude_exposure_max_us < amplitude_exposure_min_us:
            raise ValueError(
                "amplitude_exposure_max_us must be greater than or equal to "
                "amplitude_exposure_min_us."
            )
        if not 0 < amplitude_target_low < amplitude_target_high < 1:
            raise ValueError(
                "Amplitude target fractions must satisfy "
                "0 < low < high < 1."
            )
        if amplitude_exposure_factor <= 1:
            raise ValueError("amplitude_exposure_factor must be greater than 1.")
        if amplitude_max_attempts <= 0:
            raise ValueError("amplitude_max_attempts must be greater than 0.")
        if amplitude_saturation_value is not None and amplitude_saturation_value <= 0:
            raise ValueError("amplitude_saturation_value must be greater than 0.")
        if exposure_smoothing_sigma < 0:
            raise ValueError("exposure_smoothing_sigma must be non-negative.")

        log_directory = self._active_calibration_log_directory

        if hasattr(self, "CAM"):
            old_camera_frames = self.CAM.nFrames
            self.CAM.nFrames = camera_frames
            if hasattr(self.CAM, "get_integration_time"):
                old_integration_time = self.CAM.get_integration_time()
        if (
            adaptive_gradient_exposure or separate_amplitude_measurement
        ) and old_integration_time is None:
            raise RuntimeError(
                "Adaptive exposure requires a camera with "
                "get_integration_time()."
            )

        try:
            roi = _normalize_roi(roi)

            slm_pitch = self.SLM.pitch
            if np.isscalar(slm_pitch):
                x_pitch = y_pitch = float(slm_pitch)
            else:
                x_pitch, y_pitch = slm_pitch
                print(f"Pixels are asymmetric")

            # Bestimme einmalig die Patchgröße so dass Patches physikalisch
            # annähernd quadratisch sind. Wenn Sx/Sy nicht explizit übergeben
            # werden, wird entweder S als scalar/tuple interpretiert oder die
            # helper-Funktion verwendet.
            if Sx is None and Sy is None:
                if S == "GCD" or S is None:
                    sizes = calcUtils.compute_patch_size_for_physical_square((Q, P), slm_pitch, max_patches_x=10, max_patches_y=5)
                    Sx = sizes["Sx"]
                    Sy = sizes["Sy"]
                    print(f"Patch sizes: Sx={Sx}, Sy={Sy}")
                else:
                    Sx, Sy = calcUtils._normalize_patch_size(S, "S")
            else:
                if Sx is None or Sy is None:
                    raise ValueError("Sx and Sy must both be provided together.")
                Sx = int(Sx)
                Sy = int(Sy)

            if Sx <= 0 or Sy <= 0:
                raise ValueError("Patch sizes must be greater than 0.")

            live = None
            if live_view:
                live = LiveCalibrationView(roi=roi)

            if u0 is None:
                u0 = 0.1 / x_pitch
                u0 *= 1.5
            if v0 is None:
                v0 = -0.1 / y_pitch

            M = P // Sx
            N = Q // Sy

            # ------------------------------------------------------------
            # 0. Zeroth-order / Hintergrund aufnehmen
            # ------------------------------------------------------------
            black = np.zeros((Q, P)) + 1e-16
            self.SLM.showStackedField(field=[black, black])
            
            # p0, a0 = calcUtils.make_patch_ramp(P,Q,S, 0,0,u0,v0,slm_pitch)
            # self.showHologram(p0, a0)
            # input()

            background_image = np.asarray(
                self.getSettledCamImg(settle_s, discard_frames),
                dtype=float,
            )

            

            if live is not None:
                live.update(
                    black,
                    background_image,
                    "Wavefront calibration - background",
                )
                roi = live.select_roi(background_image.shape, roi)
            # Preserve the established position-calibration behaviour. The
            # separately measured amplitudes below use an exposure-specific
            # background image instead.
            thresh_zerothorder = 0

            def _set_camera_exposure(exposure_us):
                if not hasattr(self.CAM, "set_integration_time"):
                    raise RuntimeError(
                        "Adaptive exposure requires a camera with "
                        "set_integration_time()."
                    )
                self.CAM.set_integration_time(float(exposure_us))
                if hasattr(self.CAM, "get_integration_time"):
                    return float(self.CAM.get_integration_time())
                return float(exposure_us)

            def _camera_saturation_value():
                saturation_value = amplitude_saturation_value
                if saturation_value is None and hasattr(
                    self.CAM,
                    "get_saturation_value",
                ):
                    saturation_value = self.CAM.get_saturation_value()
                return 255.0 if saturation_value is None else float(saturation_value)

            def _smoothed_spot_metrics(raw_crop, signal_crop=None):
                """Measure exposure from a smoothed spot instead of one hot pixel."""
                raw_crop = np.asarray(raw_crop, dtype=float)
                if signal_crop is None:
                    local_background = float(np.percentile(raw_crop, 10))
                    signal_crop = np.clip(raw_crop - local_background, 0, None)
                else:
                    signal_crop = np.clip(
                        np.asarray(signal_crop, dtype=float),
                        0,
                        None,
                    )

                smoothed_signal = gaussian_filter(
                    signal_crop,
                    sigma=exposure_smoothing_sigma,
                )
                peak_y, peak_x = np.unravel_index(
                    np.argmax(smoothed_signal),
                    smoothed_signal.shape,
                )
                core_radius = max(
                    2,
                    int(np.ceil(3 * exposure_smoothing_sigma)),
                )
                core_y0 = max(0, peak_y - core_radius)
                core_y1 = min(raw_crop.shape[0], peak_y + core_radius + 1)
                core_x0 = max(0, peak_x - core_radius)
                core_x1 = min(raw_crop.shape[1], peak_x + core_radius + 1)
                raw_core = raw_crop[core_y0:core_y1, core_x0:core_x1]

                saturation_value = _camera_saturation_value()
                signal_peak_fraction = float(
                    smoothed_signal[peak_y, peak_x] / saturation_value
                )
                raw_peak_fraction = float(np.max(raw_core) / saturation_value)
                saturated_core = raw_core >= 0.98 * saturation_value
                saturation_fraction = float(np.mean(saturated_core))
                if np.count_nonzero(saturated_core) < 3:
                    saturation_fraction = 0.0
                return (
                    signal_peak_fraction,
                    raw_peak_fraction,
                    saturation_fraction,
                )

            def _next_adaptive_exposure(
                actual_exposure,
                signal_peak_fraction,
                saturation_fraction,
                dark_bound_us,
                bright_bound_us,
            ):
                """Update an exposure bracket without oscillating between endpoints."""
                too_bright = (
                    saturation_fraction > 0.0
                    or signal_peak_fraction > amplitude_target_high
                )
                if too_bright:
                    bright_bound_us = (
                        actual_exposure
                        if bright_bound_us is None
                        else min(bright_bound_us, actual_exposure)
                    )
                else:
                    dark_bound_us = (
                        actual_exposure
                        if dark_bound_us is None
                        else max(dark_bound_us, actual_exposure)
                    )

                if (
                    dark_bound_us is not None
                    and bright_bound_us is not None
                    and dark_bound_us < bright_bound_us
                ):
                    next_exposure = np.sqrt(dark_bound_us * bright_bound_us)
                elif too_bright:
                    next_exposure = actual_exposure / amplitude_exposure_factor
                else:
                    next_exposure = actual_exposure * amplitude_exposure_factor

                next_exposure = float(
                    np.clip(
                        next_exposure,
                        amplitude_exposure_min_us,
                        amplitude_exposure_max_us,
                    )
                )
                return next_exposure, dark_bound_us, bright_bound_us

            def _capture_gradient_image(label, initial_exposure_us):
                """Capture a spot image with ROI-controlled adaptive exposure."""
                exposure_us = float(
                    np.clip(
                        initial_exposure_us,
                        amplitude_exposure_min_us,
                        amplitude_exposure_max_us,
                    )
                )
                best_unsaturated = None
                last_candidate = None
                dark_bound_us = None
                bright_bound_us = None
                target_peak_fraction = 0.5 * (
                    amplitude_target_low + amplitude_target_high
                )

                for attempt in range(amplitude_max_attempts):
                    actual_exposure = _set_camera_exposure(exposure_us)
                    image = np.asarray(
                        self.getSettledCamImg(settle_s, discard_frames),
                        dtype=float,
                    )
                    if image.ndim != 2:
                        raise ValueError(
                            "Expected a 2D gradient image, got "
                            f"shape {image.shape}."
                        )

                    y0, y1, x0, x1 = roi
                    crop = image[y0:y1, x0:x1]
                    (
                        signal_peak_fraction,
                        raw_peak_fraction,
                        saturation_fraction,
                    ) = _smoothed_spot_metrics(crop)
                    candidate = {
                        "image": image,
                        "exposure_us": actual_exposure,
                        "raw_peak_fraction": raw_peak_fraction,
                        "signal_peak_fraction": signal_peak_fraction,
                        "saturation_fraction": saturation_fraction,
                    }
                    last_candidate = candidate
                    is_unsaturated = saturation_fraction == 0.0
                    candidate_distance = abs(
                        signal_peak_fraction - target_peak_fraction
                    )
                    if is_unsaturated and (
                        best_unsaturated is None
                        or candidate_distance < best_unsaturated["distance"]
                    ):
                        best_unsaturated = {**candidate, "distance": candidate_distance}

                    print(
                        f"Gradient {label} exposure attempt {attempt + 1}: "
                        f"exposure={actual_exposure:.1f} us, "
                        f"smoothed spot peak={100 * signal_peak_fraction:.1f}%, "
                        f"saturated pixels={100 * saturation_fraction:.4f}%"
                    )

                    if (
                        is_unsaturated
                        and amplitude_target_low <= signal_peak_fraction
                        <= amplitude_target_high
                    ):
                        candidate["valid"] = True
                        return candidate

                    (
                        next_exposure,
                        dark_bound_us,
                        bright_bound_us,
                    ) = _next_adaptive_exposure(
                        actual_exposure,
                        signal_peak_fraction,
                        saturation_fraction,
                        dark_bound_us,
                        bright_bound_us,
                    )

                    if np.isclose(next_exposure, actual_exposure):
                        break
                    exposure_us = next_exposure

                selected = best_unsaturated or last_candidate
                if selected is not None:
                    selected.pop("distance", None)
                    selected["valid"] = False
                return selected

            # ------------------------------------------------------------
            # 1. Referenzpatch
            # ------------------------------------------------------------
            m_ref, n_ref = M // 2, N // 2

            phase_ref, amp_ref = calcUtils.make_patch_ramp(
                P, Q,
                (Sx, Sy),
                m_ref, n_ref,
                u0, v0,
                slm_pitch,
            )

            self.showHologram(phase_ref, amp_ref)

            reference_exposure_us = old_integration_time
            reference_exposure_valid = not adaptive_gradient_exposure
            reference_peak_fraction = np.nan
            reference_saturation_fraction = np.nan
            while True:
                if adaptive_gradient_exposure:
                    reference_measurement = _capture_gradient_image(
                        "reference",
                        reference_exposure_us,
                    )
                    img_ref = reference_measurement["image"]
                    reference_exposure_us = reference_measurement["exposure_us"]
                    reference_exposure_valid = reference_measurement["valid"]
                    reference_peak_fraction = reference_measurement[
                        "signal_peak_fraction"
                    ]
                    reference_saturation_fraction = reference_measurement[
                        "saturation_fraction"
                    ]
                else:
                    img_ref = self.getSettledCamImg(settle_s, discard_frames)
                img_ref = np.clip(img_ref, 0, None)

                if adaptive_gradient_exposure and not reference_exposure_valid:
                    raise RuntimeError(
                        "No reliably exposed reference spot was found in the ROI. "
                        "Check the ROI, exposure limits, or smoothing sigma."
                    )
                
                if live is not None:
                    live.update(
                        phase_ref,
                        img_ref,
                        "Wavefront calibration - reference patch",
                        patch_center=((m_ref + 0.5) * Sx, (n_ref + 0.5) * Sy),
                        roi=roi,
                    )
                    # roi = live.select_roi(img_ref.shape, roi)
                else:
                    roi = _normalize_roi(
                        np.fromstring(
                            input("Give ROI y0, y1, x0, x1: "),
                            sep=",",
                            dtype=np.int32,
                        ),
                        img_ref.shape,
                    )

                com_ref, power_ref_raw = calcUtils.spot_com_and_power(img_ref, roi)
                power_ref = (
                    power_ref_raw / reference_exposure_us
                    if adaptive_gradient_exposure
                    else power_ref_raw
                )
                print(
                    f"ref power: {power_ref}"
                    + (
                        f" a.u./us at {reference_exposure_us:.1f} us"
                        if adaptive_gradient_exposure
                        else ""
                    )
                )

                if com_ref is not None:
                    if live is not None:
                        live.update(
                            phase_ref,
                            img_ref,
                            "Wavefront calibration - reference patch",
                            cam_com=com_ref,
                            patch_center=((m_ref + 0.5) * Sx, (n_ref + 0.5) * Sy),
                            cam_ref_com=com_ref,
                            roi=roi,
                        )
                    break

                print("Referenzspot nicht gefunden. Bitte ROI/Belichtung pruefen.")

            # ------------------------------------------------------------
            # Speicher
            # ------------------------------------------------------------
            gradients = np.zeros((N, M, 2), dtype=float)
            A_patch = np.zeros((N, M), dtype=float)

            com_patch = np.full((N, M, 2), np.nan, dtype=float)
            power_patch = np.zeros((N, M), dtype=float)
            best_alphas = np.full((N, M), np.nan, dtype=float)
            best_betas = np.full((N, M), np.nan, dtype=float)
            amplitude_power_raw = np.full((N, M), np.nan, dtype=float)
            amplitude_exposure_us = np.full((N, M), np.nan, dtype=float)
            amplitude_peak_fraction = np.full((N, M), np.nan, dtype=float)
            amplitude_saturation_fraction = np.full((N, M), np.nan, dtype=float)
            amplitude_measurement_valid = np.zeros((N, M), dtype=bool)
            gradient_exposure_us = np.full((N, M), np.nan, dtype=float)
            gradient_peak_fraction = np.full((N, M), np.nan, dtype=float)
            gradient_saturation_fraction = np.full((N, M), np.nan, dtype=float)
            gradient_exposure_valid = np.zeros((N, M), dtype=bool)

            px_to_freq = self.CAM.pitch / (focal_length * self.waveLength)

            probe_freq = probe_px * px_to_freq
            max_step = max_step_px * px_to_freq

            data = {
                "calibration_method": "calibrate_wavefront_pt3",
                "started_at": self._active_calibration_metadata["started_at"],
                "S": (Sx, Sy),
                "slm_pitch": slm_pitch,
                "u0": u0,
                "v0": v0,
                "roi": np.asarray(roi),
                "com_ref": com_ref,
                "power_ref": power_ref,
                "reference_exposure_us": reference_exposure_us,
                "reference_exposure_valid": reference_exposure_valid,
                "reference_peak_fraction": reference_peak_fraction,
                "reference_saturation_fraction": reference_saturation_fraction,
                "skip_gradient_search": skip_gradient_search,
                "probe_px": probe_px,
                "probe_freq": probe_freq,
                "max_step_px": max_step_px,
                "damping": damping,
                "useCorrection": useCorrection,
                "adaptive_gradient_exposure": adaptive_gradient_exposure,
                "separate_amplitude_measurement": separate_amplitude_measurement,
                "amplitude_exposure_min_us": amplitude_exposure_min_us,
                "amplitude_exposure_max_us": amplitude_exposure_max_us,
                "amplitude_target_low": amplitude_target_low,
                "amplitude_target_high": amplitude_target_high,
                "amplitude_exposure_factor": amplitude_exposure_factor,
                "amplitude_max_attempts": amplitude_max_attempts,
                "amplitude_saturation_value": (
                    "auto"
                    if amplitude_saturation_value is None
                    else float(amplitude_saturation_value)
                ),
                "exposure_smoothing_sigma": exposure_smoothing_sigma,
                "focal_length": focal_length,
                "wavelength": self.waveLength,
                "cam_pitch": self.CAM.pitch,
                "eps_px": eps_px,
                "max_iter": max_iter,
                "settle_s": settle_s,
                "discard_frames": discard_frames,
                "camera_frames": camera_frames,
                "live_view": live_view,
                "live_every": live_every,
                "display_type": self.display_type,
                "is_simulative": self.isSimulative,
                "background_image": background_image,
                "reference_image": img_ref,
                "coms": [],
                "jacobians": [],
                "convergence": [],
                "amplitude_power_raw": amplitude_power_raw,
                "amplitude_exposure_us": amplitude_exposure_us,
                "amplitude_peak_fraction": amplitude_peak_fraction,
                "amplitude_saturation_fraction": amplitude_saturation_fraction,
                "amplitude_measurement_valid": amplitude_measurement_valid,
                "gradient_exposure_us": gradient_exposure_us,
                "gradient_peak_fraction": gradient_peak_fraction,
                "gradient_saturation_fraction": gradient_saturation_fraction,
                "gradient_exposure_valid": gradient_exposure_valid,
            }

            # ------------------------------------------------------------
            # Helper: Patch messen
            # ------------------------------------------------------------
            def measure_patch(m, n, alpha_tilde, beta_tilde, title_suffix="", k=None):
                phase, amp = calcUtils.make_patch_ramp(
                    P, Q,
                    (Sx, Sy),
                    m, n,
                    u0 - alpha_tilde,
                    v0 - beta_tilde,
                    slm_pitch,
                )
                if useCorrection:
                    phase = (phase +correctionPhase)%(2*np.pi)
                # phase = server.mosaic * amp
                self.showHologram(phase, amp)

                if adaptive_gradient_exposure:
                    patch_exposure = gradient_exposure_us[n, m]
                    exposure_spot_valid = True
                    if not np.isfinite(patch_exposure):
                        previous_exposures = gradient_exposure_us[
                            np.isfinite(gradient_exposure_us)
                        ]
                        initial_patch_exposure = (
                            previous_exposures[-1]
                            if previous_exposures.size
                            else reference_exposure_us
                        )
                        gradient_measurement = _capture_gradient_image(
                            f"patch ({m},{n}) initial",
                            initial_patch_exposure,
                        )
                        img = gradient_measurement["image"]
                        measurement_exposure_us = gradient_measurement[
                            "exposure_us"
                        ]
                        gradient_exposure_us[n, m] = measurement_exposure_us
                        gradient_peak_fraction[n, m] = gradient_measurement[
                            "signal_peak_fraction"
                        ]
                        gradient_saturation_fraction[n, m] = gradient_measurement[
                            "saturation_fraction"
                        ]
                        gradient_exposure_valid[n, m] = gradient_measurement[
                            "valid"
                        ]
                        exposure_spot_valid = gradient_measurement["valid"]
                    else:
                        measurement_exposure_us = _set_camera_exposure(
                            patch_exposure
                        )
                        img = np.asarray(
                            self.getSettledCamImg(settle_s, discard_frames),
                            dtype=float,
                        )
                else:
                    img = self.getSettledCamImg(settle_s, discard_frames)
                    measurement_exposure_us = old_integration_time
                    exposure_spot_valid = True
                img -= thresh_zerothorder
                img = np.clip(img, 0, None)
                if exposure_spot_valid:
                    com, power_raw = calcUtils.spot_com_and_power(img, roi)
                else:
                    print(
                        f"Patch ({m},{n}): no reliably exposed spot found in ROI."
                    )
                    com, power_raw = None, 0.0
                power = (
                    power_raw / measurement_exposure_us
                    if adaptive_gradient_exposure
                    else power_raw
                )

                if live is not None:
                    do_update = True
                    if k is not None:
                        do_update = (k % live_every == 0)

                    if do_update:
                        live.update(
                            phase,
                            img,
                            f"Wavefront calibration - patch ({m}, {n}) {title_suffix}",
                            cam_com=com,
                            patch_center=((m + 0.5) * Sx, (n + 0.5) * Sy),
                            roi=roi,
                        )
                delta_px = None if com is None else com - com_ref
                return delta_px, power, com

            amplitude_background_cache = {}

            def _set_amplitude_exposure(exposure_us):
                return _set_camera_exposure(exposure_us)

            def _get_amplitude_background(exposure_us):
                actual_exposure = _set_amplitude_exposure(exposure_us)
                cache_key = round(actual_exposure, 6)
                if cache_key not in amplitude_background_cache:
                    self.SLM.showStackedField(field=[black, black])
                    background = np.asarray(
                        self.getSettledCamImg(settle_s, discard_frames),
                        dtype=float,
                    )
                    if background.ndim != 2:
                        raise ValueError(
                            "Expected a 2D amplitude-background image, got "
                            f"shape {background.shape}."
                        )
                    amplitude_background_cache[cache_key] = background
                return actual_exposure, amplitude_background_cache[cache_key]

            def _measure_final_amplitude(m, n, alpha_tilde, beta_tilde):
                """Measure one final patch with adaptive, exposure-normalized power."""
                if old_integration_time is None:
                    raise RuntimeError(
                        "The camera exposure cannot be read. Separate amplitude "
                        "measurement cannot restore the position-calibration exposure."
                    )

                phase, amp = calcUtils.make_patch_ramp(
                    P, Q,
                    (Sx, Sy),
                    m, n,
                    u0 - alpha_tilde,
                    v0 - beta_tilde,
                    slm_pitch,
                )
                if useCorrection:
                    phase = (phase + correctionPhase) % (2 * np.pi)

                patch_gradient_exposure = gradient_exposure_us[n, m]
                initial_amplitude_exposure = (
                    patch_gradient_exposure
                    if np.isfinite(patch_gradient_exposure)
                    else old_integration_time
                )
                exposure_us = float(
                    np.clip(
                        initial_amplitude_exposure,
                        amplitude_exposure_min_us,
                        amplitude_exposure_max_us,
                    )
                )
                best_unsaturated = None
                selected = None
                dark_bound_us = None
                bright_bound_us = None
                target_peak_fraction = 0.5 * (
                    amplitude_target_low + amplitude_target_high
                )

                try:
                    for attempt in range(amplitude_max_attempts):
                        actual_exposure, amplitude_background = _get_amplitude_background(
                            exposure_us
                        )
                        self.showHologram(phase, amp)
                        raw_image = np.asarray(
                            self.getSettledCamImg(settle_s, discard_frames),
                            dtype=float,
                        )
                        if raw_image.ndim != 2:
                            raise ValueError(
                                "Expected a 2D amplitude image, got "
                                f"shape {raw_image.shape}."
                            )
                        if raw_image.shape != amplitude_background.shape:
                            raise ValueError(
                                "Amplitude image and background image have different "
                                f"shapes: {raw_image.shape} and "
                                f"{amplitude_background.shape}."
                            )

                        corrected_image = np.clip(
                            raw_image - amplitude_background,
                            0,
                            None,
                        )
                        y0, y1, x0, x1 = roi
                        raw_crop = raw_image[y0:y1, x0:x1]
                        corrected_crop = corrected_image[y0:y1, x0:x1]

                        (
                            signal_peak_fraction,
                            raw_peak_fraction,
                            saturation_fraction,
                        ) = _smoothed_spot_metrics(
                            raw_crop,
                            corrected_crop,
                        )
                        _, raw_power = calcUtils.spot_com_and_power(
                            corrected_image,
                            roi,
                            power_in_mask=True,
                        )

                        candidate = {
                            "power_raw": float(raw_power),
                            "power_normalized": float(raw_power / actual_exposure),
                            "exposure_us": actual_exposure,
                            "peak_fraction": signal_peak_fraction,
                            "raw_peak_fraction": raw_peak_fraction,
                            "signal_peak_fraction": signal_peak_fraction,
                            "saturation_fraction": saturation_fraction,
                            "image": corrected_image,
                            "phase": phase,
                        }
                        is_unsaturated = saturation_fraction == 0.0
                        candidate_distance = abs(
                            signal_peak_fraction - target_peak_fraction
                        )
                        if is_unsaturated and (
                            best_unsaturated is None
                            or candidate_distance < best_unsaturated["distance"]
                        ):
                            best_unsaturated = {
                                **candidate,
                                "distance": candidate_distance,
                            }

                        print(
                            f"Amplitude patch ({m},{n}) attempt {attempt + 1}: "
                            f"exposure={actual_exposure:.1f} us, "
                            f"smoothed spot peak={100 * signal_peak_fraction:.1f}%, "
                            f"saturated pixels={100 * saturation_fraction:.4f}%"
                        )

                        if (
                            is_unsaturated
                            and amplitude_target_low <= signal_peak_fraction
                            <= amplitude_target_high
                        ):
                            candidate["valid"] = True
                            selected = candidate
                            break

                        (
                            next_exposure,
                            dark_bound_us,
                            bright_bound_us,
                        ) = _next_adaptive_exposure(
                            actual_exposure,
                            signal_peak_fraction,
                            saturation_fraction,
                            dark_bound_us,
                            bright_bound_us,
                        )

                        if np.isclose(next_exposure, actual_exposure):
                            break
                        exposure_us = next_exposure

                    if selected is None and best_unsaturated is not None:
                        best_unsaturated.pop("distance", None)
                        best_unsaturated["valid"] = False
                        selected = best_unsaturated
                    return selected
                finally:
                    _set_amplitude_exposure(old_integration_time)

            def _store_patch_amplitude(
                m,
                n,
                alpha_tilde,
                beta_tilde,
                fallback_power,
            ):
                if not separate_amplitude_measurement:
                    power_patch[n, m] = fallback_power
                    A_patch[n, m] = np.sqrt(max(fallback_power, 0.0))
                    amplitude_power_raw[n, m] = fallback_power
                    amplitude_exposure_us[n, m] = (
                        np.nan
                        if old_integration_time is None
                        else old_integration_time
                    )
                    amplitude_measurement_valid[n, m] = np.isfinite(fallback_power)
                    return

                result = _measure_final_amplitude(
                    m,
                    n,
                    alpha_tilde,
                    beta_tilde,
                )
                if result is None:
                    print(
                        f"Amplitude patch ({m},{n}) invalid: no unsaturated "
                        "measurement was available."
                    )
                    power_patch[n, m] = 0.0
                    A_patch[n, m] = 0.0
                    return

                power_patch[n, m] = result["power_normalized"]
                A_patch[n, m] = np.sqrt(max(result["power_normalized"], 0.0))
                amplitude_power_raw[n, m] = result["power_raw"]
                amplitude_exposure_us[n, m] = result["exposure_us"]
                amplitude_peak_fraction[n, m] = result["peak_fraction"]
                amplitude_saturation_fraction[n, m] = result[
                    "saturation_fraction"
                ]
                amplitude_measurement_valid[n, m] = result["valid"]

                if live is not None:
                    live.update(
                        result["phase"],
                        result["image"],
                        f"Wavefront calibration - patch ({m}, {n}) amplitude",
                        patch_center=((m + 0.5) * Sx, (n + 0.5) * Sy),
                        roi=roi,
                    )

            # ------------------------------------------------------------
            # 2. Patch scanning
            # ------------------------------------------------------------
            for n in range(N):
                for m in range(M):

                    # input("next patch?")

                    print(f"\n--- Patch ({m}, {n}) ---")

                    # ----------------------------------------------------
                    # Optional: ohne iterative Suche
                    # ----------------------------------------------------
                    if skip_gradient_search:
                        delta_px, power, com = measure_patch(
                            m, n,
                            0.0, 0.0,
                            title_suffix="direct",
                        )

                        if com is not None:
                            com_patch[n, m] = com
                            data["convergence"].append(
                                (m, n, 0, np.linalg.norm(delta_px))
                            )

                            alpha_tilde = delta_px[0] * px_to_freq
                            beta_tilde = delta_px[1] * px_to_freq

                            gradients[n, m, 0] = -2 * np.pi * alpha_tilde
                            gradients[n, m, 1] = -2 * np.pi * beta_tilde
                            best_alphas[n, m] = alpha_tilde
                            best_betas[n, m] = beta_tilde
                            _store_patch_amplitude(
                                m,
                                n,
                                alpha_tilde,
                                beta_tilde,
                                power,
                            )

                        continue

                    # ----------------------------------------------------
                    # Iterative Suche mit lokaler Jacobi-Matrix
                    # ----------------------------------------------------
                    p = np.array([0.0, 0.0], dtype=float)

                    delta0, power0, com0 = measure_patch(
                        m, n,
                        p[0], p[1],
                        title_suffix="initial",
                        k=0,
                    )

                    if delta0 is None:
                        print("No spot found at initial measurement; skipping patch.")
                        continue
                    if power0 / power_ref < 1e-4:
                        print(f"pow/pow_ref {power0/power_ref}: No spot in ROI, skipping patch")
                        continue

                    com_patch[n, m] = com0
                    power_patch[n, m] = power0
                    data["coms"].append(com0.copy())

                    best_err = np.linalg.norm(delta0)
                    best_p = p.copy()
                    best_power = power0
                    best_com = com0.copy()
                    data["convergence"].append((m, n, 0, best_err))

                    print(f"initial err: {best_err:.3f}px, delta={delta0}")

                    if best_err <= eps_px:
                        alpha_tilde, beta_tilde = best_p
                        gradients[n, m, 0] = -2 * np.pi * alpha_tilde
                        gradients[n, m, 1] = -2 * np.pi * beta_tilde
                        best_alphas[n, m] = alpha_tilde
                        best_betas[n, m] = beta_tilde
                        _store_patch_amplitude(
                            m,
                            n,
                            alpha_tilde,
                            beta_tilde,
                            best_power,
                        )
                        continue

                    # ----------------------------------------------------
                    # Probe-Schritte: Wirkung von alpha und beta messen
                    # ----------------------------------------------------
                    delta_a, _, com_a = measure_patch(
                        m, n,
                        p[0] + probe_freq,
                        p[1],
                        title_suffix="alpha probe",
                        k=0,
                    )

                    delta_b, _, com_b = measure_patch(
                        m, n,
                        p[0],
                        p[1] + probe_freq,
                        title_suffix="beta probe",
                        k=0,
                    )

                    if delta_a is None or delta_b is None:
                        print("Probe measurement failed; skipping patch.")
                        continue

                    J = np.column_stack([
                        (delta_a - delta0) / probe_freq,
                        (delta_b - delta0) / probe_freq,
                    ])

                    data["jacobians"].append(J.copy())

                    print("Measured local Jacobian:")
                    print(J)

                    # Falls Probe in falsche Richtung geht, erkennt J das automatisch.
                    # Der Newton-Schritt nutzt J @ step = -delta_px.
                    delta_px = delta0.copy()
                    power_current = power0

                    worsening_steps = 0
                    last_err = best_err

                    for k in range(max_iter):
                        err = np.linalg.norm(delta_px)

                        print(
                            f"Patch ({m},{n}) iter {k}: "
                            f"err={err:.3f}px, "
                            f"alpha={p[0]:.4e}, beta={p[1]:.4e}, "
                            f"delta={delta_px}"
                        )

                        if err <= eps_px:
                            break

                        step = _safe_newton_step(J, delta_px, max_step)
                        p_new = p + damping * step

                        delta_new, power_new, com_new = measure_patch(
                            m, n,
                            p_new[0], p_new[1],
                            title_suffix=f"iter {k + 1}",
                            k=k,
                        )

                        if delta_new is None:
                            print("Spot lost during iteration; using best value.")
                            break

                        err_new = np.linalg.norm(delta_new)
                        data["convergence"].append((m, n, k + 1, err_new))

                        if com_new is not None:
                            com_patch[n, m] = com_new
                            power_patch[n, m] = power_new
                            data["coms"].append(com_new.copy())

                        if err_new < best_err:
                            best_err = err_new
                            best_p = p_new.copy()
                            best_power = power_new
                            best_com = com_new.copy()

                        if err_new > last_err:
                            worsening_steps += 1
                        else:
                            worsening_steps = 0

                        # Broyden-Update der Jacobi-Matrix
                        s_vec = p_new - p
                        y_vec = delta_new - delta_px

                        denom = np.dot(s_vec, s_vec)
                        if denom > 0:
                            J = J + np.outer((y_vec - J @ s_vec), s_vec) / denom

                        p = p_new
                        delta_px = delta_new
                        power_current = power_new
                        last_err = err_new

                        if worsening_steps >= 2:
                            print(
                                f"Error increased twice; using best err={best_err:.3f}px."
                            )
                            break

                    alpha_tilde, beta_tilde = best_p
                    best_alphas[n, m] = alpha_tilde
                    best_betas[n, m] = beta_tilde
                    com_patch[n, m] = best_com

                    gradients[n, m, 0] = -2 * np.pi * alpha_tilde
                    gradients[n, m, 1] = -2 * np.pi * beta_tilde

                    _store_patch_amplitude(
                        m,
                        n,
                        alpha_tilde,
                        beta_tilde,
                        best_power,
                    )

                    print(
                        f"best patch ({m},{n}): "
                        f"err={best_err:.3f}px, "
                        f"alpha={alpha_tilde:.4e}, beta={beta_tilde:.4e}"
                    )

            # ------------------------------------------------------------
            # 3. Rekonstruktion
            # ------------------------------------------------------------
            data["best_alphas"] = best_alphas
            data["best_betas"] = best_betas
            data["gradients"] = gradients
            data["com_patch"] = com_patch
            data["power_patch"] = power_patch

            A_patch /= np.max(A_patch) + 1e-12

            # Alte Variante: Gradienten auf SLM-Gitter interpolieren
            gx_map = calcUtils.interpolate_patch_values(
                gradients[..., 0],
                P, Q,
                (Sx, Sy),
                kind="linear",
            )
            gy_map = calcUtils.interpolate_patch_values(
                gradients[..., 1],
                P, Q,
                (Sx, Sy),
                kind="linear",
            )

            gx_map -= gx_map.mean()
            gy_map -= gy_map.mean()

            phi_map = calcUtils.poisson_reconstruct_phase(
                gx_map,
                gy_map,
                slm_pitch,
            )

            plotUtils.plot_wavefront(gx_map, title="gx")
            plotUtils.plot_wavefront(gy_map, title="gy")

            data["gx"] = gx_map
            data["gy"] = gy_map
            data["phase"] = phi_map
            data["A_patch"] = A_patch

            _save_calibration_outputs(data, log_directory)

            return phi_map, gradients, A_patch

        finally:
            if old_camera_frames is not None:
                self.CAM.nFrames = old_camera_frames
            if old_integration_time is not None:
                self.CAM.set_integration_time(old_integration_time)

    def testU0_v0(self):

        Q, P = self.SLM.resolution

        S = gcd(Q, P)

                
        slm_pitch = self.SLM.pitch
        if np.isscalar(slm_pitch):
            x_pitch = y_pitch = float(slm_pitch)
        else:
            x_pitch, y_pitch = slm_pitch

        u0 = 0.1 / x_pitch
        v0 = 0.1 / y_pitch

        M = P // S
        N = Q // S

        # ------------------------------------------------------------
        # 0. Zeroth-order / Hintergrund aufnehmen
        # ------------------------------------------------------------
        black = np.zeros((Q, P)) + 1e-16
        self.SLM.showStackedField(field=[black, black])

        

        m_ref, n_ref = M // 2, N // 2

        for i in np.linspace(0.1, 0.3 , 11):
            u0, v0 =2000, 0
            print(f"u0: {u0}, v0: {v0}")
            phase_ref, amp_ref = calcUtils.make_patch_ramp(P, Q, S, m_ref, n_ref, u0, v0, slm_pitch)
    
            self.showHologram(phase_ref, amp_ref)
            input("press to continue")


class Calibrators:
    def __init__(self): 
        pass
    

def save_zernike_phase_from_npz(
    input_path=None,
    output_phase_path="log/zernike_phase.npy",
    output_data_path="log/zernike_fit.npz",
    slm_pitch=(25e-6, 75e-6),
    zernike_indices=range(2, 16),
    aperture_mode="full_slm_diagonal",
    amplitude_threshold=0.05,
    regularization=0.0,
    plot=True,
):
    """Load one calibration .npz file, fit Zernikes, and save the fitted phase."""
    data = calcUtils.loadNpz(input_path)

    if "gradients" not in data:
        raise KeyError('The selected .npz file must contain data["gradients"].')

    if "phase" in data:
        resolution = data["phase"].shape
    elif "gx" in data:
        resolution = data["gx"].shape
    else:
        raise KeyError('The selected .npz file must contain data["phase"] or data["gx"].')

    if "S" not in data:
        raise KeyError('The selected .npz file must contain data["S"].')
    S_data = np.asarray(data["S"]).squeeze()
    if S_data.shape == ():
        S = int(S_data.item())
    elif S_data.size == 2:
        S = tuple(np.ravel(S_data).astype(int))
    else:
        raise ValueError('data["S"] must be a scalar or contain two patch-size values.')

    if "slm_pitch" in data:
        slm_pitch = tuple(np.asarray(data["slm_pitch"], dtype=float).ravel())
        if len(slm_pitch) == 1:
            slm_pitch = float(slm_pitch[0])
        elif len(slm_pitch) != 2:
            raise ValueError('data["slm_pitch"] must contain one or two values.')

    amplitudes = None
    if "A_patch" in data:
        amplitudes = data["A_patch"]
    elif "power_patch" in data:
        amplitudes = np.sqrt(np.maximum(data["power_patch"], 0))

    zernike_fit = calcUtils.fit_zernike_from_gradients(
        data["gradients"],
        resolution=resolution,
        patch_size=S,
        slm_pitch=slm_pitch,
        zernike_indices=zernike_indices,
        amplitudes=amplitudes,
        aperture_mode=aperture_mode,
        amplitude_threshold=amplitude_threshold if amplitudes is not None else None,
        regularization=regularization,
        return_details=True,
    )

    phase_zernike = zernike_fit["phase"]
    Path(output_phase_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_phase_path, phase_zernike)

    if output_data_path is not None:
        calcUtils.saveNpz(
            {
                "phase_zernike": phase_zernike,
                "residuals_zernike": zernike_fit["residuals"],
                "gx_fit_zernike": zernike_fit["gx_fit"],
                "gy_fit_zernike": zernike_fit["gy_fit"],
                "valid_mask_zernike": zernike_fit["valid_mask"],
                "zernike_indices": np.asarray(zernike_fit["zernike_indices"]),
                "zernike_coefficients": np.asarray(
                    [zernike_fit["coefficients"][idx] for idx in zernike_fit["zernike_indices"]]
                ),
                "zernike_coefficients_waves": np.asarray(
                    [zernike_fit["coefficients_waves"][idx] for idx in zernike_fit["zernike_indices"]]
                ),
                "residual_rms": np.asarray(zernike_fit["residual_rms"]),
                "slm_pitch": np.asarray(zernike_fit["aperture"]["slm_pitch"]),
                "aperture_radius": np.asarray(zernike_fit["aperture"]["radius"]),
                "aperture_center": np.asarray(zernike_fit["aperture"]["center"]),
            },
            output_data_path,
        )

    print(f"Saved Zernike phase to {output_phase_path}")
    if output_data_path is not None:
        print(f"Saved Zernike fit data to {output_data_path}")
    print(f"Zernike gradient residual RMS: {zernike_fit['residual_rms']:.6e} rad/m")
    print("Zernike coefficients:")
    for fringe_index in zernike_fit["zernike_indices"]:
        coeff = zernike_fit["coefficients"][fringe_index]
        coeff_waves = zernike_fit["coefficients_waves"][fringe_index]
        print(f"  fringe {fringe_index:2d}: {coeff: .6e} rad ({coeff_waves: .6e} waves)")

    if plot:
        gx_measured_map = calcUtils.interpolate_patch_values(
            data["gradients"][..., 0],
            resolution[1],
            resolution[0],
            S,
            kind="linear",
        )
        gy_measured_map = calcUtils.interpolate_patch_values(
            data["gradients"][..., 1],
            resolution[1],
            resolution[0],
            S,
            kind="linear",
        )
        residual_mag = np.sqrt(
            zernike_fit["residuals"][..., 0] ** 2
            + zernike_fit["residuals"][..., 1] ** 2
        )

        plotUtils.plot_wavefront(phase_zernike, title="Zernike phase")
        plotUtils.plot_phase_gradient(
            phase_zernike,
            gx_measured_map,
            gy_measured_map,
            title="Zernike phase with measured gradients",
        )
        plotUtils.plot_camImg(residual_mag, title="Zernike gradient residual magnitude")
        plt.show()

    return phase_zernike, zernike_fit


def playground():  
    # wf = np.load("log/wavefrontmap.npy")
    # gx = np.load("log/gx_map.npy")
    # gy = np.load("log/gy_map.npy")

    data = calcUtils.loadNpz()
    plotUtils.plot_phase_gradient(data["phase"],data["gx"], data["gy"], title="Phase")

    
    Q, P = data["gx"].shape
    S = data["S"]
    slm_pitch = data.get("slm_pitch", 8e-6)
    gx_raw, gy_raw = calcUtils.patch_gradients_to_maps(data["gradients"],P,Q,S)

    Sx, Sy = calcUtils._normalize_patch_size(S, "S")
    x_pitch, y_pitch = calcUtils._normalize_pixel_pitch(slm_pitch, "slm_pitch")
    patch_dx = (Sx * x_pitch, Sy * y_pitch)
    curlRaw = calcUtils.curl_2d(data["gradients"][...,0], data["gradients"][...,1], patch_dx)
    plotUtils.plot_camImg(curlRaw, title="Curl patch Gradients")
    
    phi_interp, _ = calcUtils.reconstruct_phase_from_patch_gradients(data["gradients"], P, Q, S, slm_pitch, "linear")
    np.save("log/wavefrontmap_pt3.npy", phi_interp)
    #phi_raw = calcUtils.poisson_reconstruct_phase_direct_Fourier_Integration(gx_raw, gy_raw, S*8e-6)
    plotUtils.plot_phase_gradient(phi_interp,gx_raw, gy_raw, title="Phase from Patches")
    shape_x, shape_y = data["gx"].shape
    plt.show()
    denom = 8
    
    gx, gy = data["gx"][shape_x//2-shape_x//denom:shape_x//2+shape_x//denom, shape_y//2-shape_y//denom:shape_y//2+shape_y//denom], data["gy"][shape_x//2-shape_x//denom:shape_x//2+shape_x//denom, shape_y//2-shape_y//denom:shape_y//2+shape_y//denom]
    plotUtils.plot_com_positions(data["com_patch"], data["com_ref"])
    plotUtils.plot_gradients_quiver(data["gx"], data["gy"])
    resolution = data["phase"].shape
    
    # phaseDFI = calcUtils.poisson_reconstruct_phase_direct_Fourier_Integration(gx, gy, 8e-6)
    phaseDFI = calcUtils.poisson_reconstruct_phase(gx,gy, slm_pitch, 300)
    gxDFI, gyDFI= calcUtils.phaseMaskToGradients(phaseDFI, slm_pitch)
    g = np.sqrt(gx**2+gy**2)
    gDFI = np.sqrt(gxDFI**2+gyDFI**2)
    mse = np.mean((g-gDFI)**2)
    
    print(f"Curl gy gy: {calcUtils.curl_2d(gx,gy,slm_pitch)} ")
    print(f"Curl gyDFI gyDFI: {calcUtils.curl_2d(gxDFI,gyDFI,slm_pitch)} ")
    
     
    print(f"MSE DFI: {mse}")
    
    plotUtils.plot_camImg(g-gDFI, title="Gradient Error")
    plotUtils.plot_phase_gradient(phaseDFI, gxDFI, gyDFI, title= "Gradients DFI")
    # S = np.gcd(resolution[0], resolution[1])
    # mosaic_mask = calcUtils.make_patch_mosaic_mask(data["best_alphas"], data["best_betas"], S, 8e-6, data["u0"], data["v0"], resolution=resolution)
    
    plotUtils.plot_phase_gradient(data["phase"], data["gx"], data["gy"], title= "Poisson")
    plotUtils.plot_phase_gradient(phaseDFI, gx, gy, title= "direct Fourier")
    
    
    plt.show()
    #plotUtils.plot_wavefront(data["phase"])
    # plotUtils.plot_wavefront(gx, title="gx")
    # plotUtils.plot_wavefront(gy, title="gy")
    # plt.imshow(calcUtils.wrap_phase(a/a.max()*np.pi*2))
    # plt.show()
    server = Server(False, True)
    server.SLM.setCorrectionPhaseMap(-data["phase"])
    spots = ((-1,-1), (1,1), (-1,1), (1,-1))

    #singleRamp = calcUtils.make_full_display_patch(resolution, data["u0"], data["v0"], 8e-6)
    # _, phase, _ = calcUtils.periodic_gs_multispot_slm(server.SLM.resolution, spots=spots, iterations=int(1e3))

    # phase = calcUtils.ifta(numX=4, numY=4)
    # while True:
    #     server.showHologram(singleRamp)
    #     f = input("change hologram to single with correction?")
    #     server.showHologram(calcUtils.wrap_phase(singleRamp-data["phase"]))
    #     f = input("change hologram to +correc?")
    #     server.showHologram(calcUtils.wrap_phase(singleRamp+data["phase"]))
    #     f = input("change hologram to corr flip 0?")
    #     server.showHologram(calcUtils.wrap_phase(singleRamp+np.flip(data["phase"], axis=0)))
    #     f = input("change hologram to corr flip 1?")
    #     server.showHologram(calcUtils.wrap_phase(singleRamp+np.flip(data["phase"], axis=1)))
        
    #     f = input("change hologram to mosaic?")
    #     server.showHologram(mosaic_mask)
    #     f = input("change hologram to singleRamp?")
    # phaseOnly = server.getCamImg()
    # plt.imsave("log/onlyPhase.png", phaseOnly)
    # server.showHologram(calcUtils.wrap_phase(phase+wf))
    # corrected = server.getCamImg()
    # plt.imsave("log/correctedPlus.png", corrected)

    # server.showHologram(calcUtils.wrap_phase(phase-wf))
    # corrected = server.getCamImg()
    # plt.imsave("log/correctedminus.png", corrected)


    # plt.imsave("log/Wavefront.png", calcUtils.wrap_phase(wf))
    # # server.simulatePropagation()
    #f = input()

    #server.showCamImg()
    # server.showStandardPatches()


def save_zernike_phase_from_npz(
    input_path=None,
    output_phase_path="log/zernike_phase.npy",
    output_data_path="log/zernike_fit.npz",
    slm_pitch=(25e-6, 75e-6),
    zernike_indices=range(2, 16),
    aperture_mode="full_slm_diagonal",
    amplitude_threshold=0.05,
    regularization=0.0,
    plot=True,
):
    """Load one calibration .npz file, fit Zernikes, and save the fitted phase."""
    data = calcUtils.loadNpz(input_path)

    if "gradients" not in data:
        raise KeyError('The selected .npz file must contain data["gradients"].')

    if "phase" in data:
        resolution = data["phase"].shape
    elif "gx" in data:
        resolution = data["gx"].shape
    else:
        raise KeyError('The selected .npz file must contain data["phase"] or data["gx"].')

    if "S" not in data:
        raise KeyError('The selected .npz file must contain data["S"].')
    S_data = np.asarray(data["S"]).squeeze()
    if S_data.shape == ():
        S = int(S_data.item())
    elif S_data.size == 2:
        S = tuple(np.ravel(S_data).astype(int))
    else:
        raise ValueError('data["S"] must be a scalar or contain two patch-size values.')

    if "slm_pitch" in data:
        slm_pitch = tuple(np.asarray(data["slm_pitch"], dtype=float).ravel())
        if len(slm_pitch) == 1:
            slm_pitch = float(slm_pitch[0])
        elif len(slm_pitch) != 2:
            raise ValueError('data["slm_pitch"] must contain one or two values.')

    amplitudes = None
    if "A_patch" in data:
        amplitudes = data["A_patch"]
    elif "power_patch" in data:
        amplitudes = np.sqrt(np.maximum(data["power_patch"], 0))

    zernike_fit = calcUtils.fit_zernike_from_gradients(
        data["gradients"],
        resolution=resolution,
        patch_size=S,
        slm_pitch=slm_pitch,
        zernike_indices=zernike_indices,
        amplitudes=amplitudes,
        aperture_mode=aperture_mode,
        amplitude_threshold=amplitude_threshold if amplitudes is not None else None,
        regularization=regularization,
        return_details=True,
    )

    phase_zernike = zernike_fit["phase"]
    Path(output_phase_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_phase_path, phase_zernike)

    if output_data_path is not None:
        calcUtils.saveNpz(
            {
                "phase_zernike": phase_zernike,
                "residuals_zernike": zernike_fit["residuals"],
                "gx_fit_zernike": zernike_fit["gx_fit"],
                "gy_fit_zernike": zernike_fit["gy_fit"],
                "valid_mask_zernike": zernike_fit["valid_mask"],
                "zernike_indices": np.asarray(zernike_fit["zernike_indices"]),
                "zernike_coefficients": np.asarray(
                    [zernike_fit["coefficients"][idx] for idx in zernike_fit["zernike_indices"]]
                ),
                "zernike_coefficients_waves": np.asarray(
                    [zernike_fit["coefficients_waves"][idx] for idx in zernike_fit["zernike_indices"]]
                ),
                "residual_rms": np.asarray(zernike_fit["residual_rms"]),
                "slm_pitch": np.asarray(zernike_fit["aperture"]["slm_pitch"]),
                "aperture_radius": np.asarray(zernike_fit["aperture"]["radius"]),
                "aperture_center": np.asarray(zernike_fit["aperture"]["center"]),
            },
            output_data_path,
        )

    print(f"Saved Zernike phase to {output_phase_path}")
    if output_data_path is not None:
        print(f"Saved Zernike fit data to {output_data_path}")
    print(f"Zernike gradient residual RMS: {zernike_fit['residual_rms']:.6e} rad/m")
    print("Zernike coefficients:")
    for fringe_index in zernike_fit["zernike_indices"]:
        coeff = zernike_fit["coefficients"][fringe_index]
        coeff_waves = zernike_fit["coefficients_waves"][fringe_index]
        print(f"  fringe {fringe_index:2d}: {coeff: .6e} rad ({coeff_waves: .6e} waves)")

    if plot:
        gx_measured_map = calcUtils.interpolate_patch_values(
            data["gradients"][..., 0],
            resolution[1],
            resolution[0],
            S,
            kind="linear",
        )
        gy_measured_map = calcUtils.interpolate_patch_values(
            data["gradients"][..., 1],
            resolution[1],
            resolution[0],
            S,
            kind="linear",
        )
        residual_mag = np.sqrt(
            zernike_fit["residuals"][..., 0] ** 2
            + zernike_fit["residuals"][..., 1] ** 2
        )

        plotUtils.plot_wavefront(phase_zernike, title="Zernike phase")
        plotUtils.plot_phase_gradient(
            phase_zernike,
            gx_measured_map,
            gy_measured_map,
            title="Zernike phase with measured gradients",
        )
        plotUtils.plot_camImg(residual_mag, title="Zernike gradient residual magnitude")
        plt.show()

    return phase_zernike, zernike_fit


def testCorrection():
    server = Server("pt3", False, False)

    correctionPhase = np.load("log/zernike_phase.npy")
    normal_phase = np.load("log/5Spots_phase.npy") #*2*np.pi
    shifted_phase = np.load("log/3x3_extrawide_shifted_phase.npy") #*2*np.pi

    plt.imshow(correctionPhase%(2*np.pi))
    plt.show()

    black, white = np.zeros(server.SLM.resolution)+1e-15, np.ones(server.SLM.resolution)

    normal_corrected = (normal_phase + correctionPhase)%(np.pi * 2)
    shifted_corrected = (shifted_phase + correctionPhase)%(np.pi * 2)
   
   
    while True:

        print("normal")

        server.showHologram(normal_corrected, white, False)

        input("press for next")
        # print("shifted")
        # server.showHologram(shifted_corrected, white, False)
        # input("show next")
    
if __name__ == "__main__":
    
    #save_zernike_phase_from_npz()
    # testCorrection()
    
    # data = calcUtils.loadNpz()
    
    server = Server("pt3",False, True)

    # server.mosaic = calcUtils.make_patch_mosaic_mask(data["best_alphas"], data["best_betas"],
    #                                                  data["S"], server.SLM.pitch, data["u0"], data["v0"], server.SLM.resolution)

    # save_zernike_phase_from_npz(output_phase_path="log/zernike_phase.npy")
    # playground()
    # plt.show()

    # server.testU0_v0()

    server.CAM.set_integration_time(160e3)
    server.CAM.nFrames = 1
    server.calibrate_wavefront_pt3(S=None,focal_length=0.8, live_view=True,live_every=4, skip_gradient_search=False, max_iter=10)
    
    # plotUtils.plot_wavefront(phase)


    #TODO:   Phasemaske einfach als Mosaik von patches. ist dann zwar shifted aber sollte trotzdem bessers sein

    
    
