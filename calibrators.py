import hardware
import calcUtils
import plotUtils
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from math import gcd


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

        if initial_roi is not None:
            self.set_roi(initial_roi)

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
        self.fig.suptitle("Wavefront calibration - draw ROI on reference spot, then press Enter")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        input("Draw ROI in the live view, then press Enter to continue.")

        selector.set_active(False)
        selector.disconnect_events()
        self.fig.suptitle(previous_title)

        if selected_roi["value"] is None:
            if initial_roi is None:
                raise RuntimeError("No ROI was selected.")
            selected_roi["value"] = initial_roi

        self.set_roi(selected_roi["value"])
        return selected_roi["value"]

    def update(self, holo=None, cam_img=None, title=None, cam_com=None, patch_center=None, cam_ref_com=None, roi = None):
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
            if self.cam_image is None:
                self.cam_image = self.ax_cam.imshow(
                    cam_preview,
                    cmap="nipy_spectral",
                    origin="lower",
                    extent=self.cam_extent,
                )
                self.fig.colorbar(self.cam_image, ax=self.ax_cam, fraction=0.046, pad=0.04)
            else:
                self.cam_image.set_data(cam_preview)
                self.cam_image.set_clim(float(np.nanmin(cam_preview)), float(np.nanmax(cam_preview)))

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

        #0. Ordung abziehen
        # black = np.zeros((Q,P)) +1e-16
        # self.SLM.showStackedField(field = [black, black]  )
        
        # thresh_zerothorder = self.getSettledCamImg(settle_s, discard_frames)
        

        # if live is not None:
        #     live.update(
        #         black,
        #         thresh_zerothorder,
        #         "Wavefront calibration - reference patch",
               
        #     )
        # roi = live.select_roi(thresh_zerothorder.shape, roi)


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
            "S": S,
            "u0": u0,
            "v0": v0,
            "roi": np.asarray(roi),
            "com_ref": com_ref,
            "power_ref": power_ref,
            "skip_gradient_search": skip_gradient_search,
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

                    if skip_gradient_search:
                        A_patch[n, m] = np.sqrt(power)
                        break

                    delta_px = com - com_ref
                    err = np.linalg.norm(delta_px)

                    hist.append((alpha_tilde, beta_tilde, delta_px.copy()))

                    print(f"Pixel:({m},{n}) iteration {k}, err: {err}, eps_px: {eps_px}")

                    if err < best_err:
                        best_err = err
                        best_alpha_tilde = alpha_tilde
                        best_beta_tilde = beta_tilde
                        best_power = power
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
            calcUtils.saveNpz(data, "log/data_" + calcUtils.getTimestamp() + ".npz")
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
        
        calcUtils.saveNpz(data, "log/data_" + calcUtils.getTimestamp() + ".npz")

        # np.save("log/gx_map.npy", gx_map)
        # np.save("log/gy_map.npy", gy_map)
        # plt.imsave("log/Wavefront.png", calcUtils.wrap_phase(phi_map))
        # np.save("log/wavefrontmap.npy", phi_map)

        if old_camera_frames is not None:
            self.CAM.nFrames = old_camera_frames

        return phi_map, gradients, A_patch  #A_map, 
    

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
        useCorrection = False
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

        if hasattr(self, "CAM"):
            old_camera_frames = self.CAM.nFrames
            self.CAM.nFrames = camera_frames

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

            thresh_zerothorder = self.getSettledCamImg(settle_s, discard_frames)

            

            if live is not None:
                live.update(
                    black,
                    thresh_zerothorder,
                    "Wavefront calibration - background",
                )
                roi = live.select_roi(thresh_zerothorder.shape, roi)
            thresh_zerothorder = 0
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

            while True:
                img_ref = self.getSettledCamImg(settle_s, discard_frames) - thresh_zerothorder
                img_ref = np.clip(img_ref, 0, None)
                
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

                com_ref, power_ref = calcUtils.spot_com_and_power(img_ref, roi)
                print(f"ref power: {power_ref}")

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

            px_to_freq = self.CAM.pitch / (focal_length * self.waveLength)

            probe_freq = probe_px * px_to_freq
            max_step = max_step_px * px_to_freq

            data = {
                "S": (Sx, Sy),
                "slm_pitch": slm_pitch,
                "u0": u0,
                "v0": v0,
                "roi": np.asarray(roi),
                "com_ref": com_ref,
                "power_ref": power_ref,
                "skip_gradient_search": skip_gradient_search,
                "probe_px": probe_px,
                "probe_freq": probe_freq,
                "max_step_px": max_step_px,
                "damping": damping,
                "coms": [],
                "jacobians": [],
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
                
                img = self.getSettledCamImg(settle_s, discard_frames) 
                img -= thresh_zerothorder
                img = np.clip(img, 0, None)
                com, power = calcUtils.spot_com_and_power(img, roi)

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
                        )
                if com is None: 
                    delta_px = 0
                else:
                    delta_px = com - com_ref
                return delta_px, power, com

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
                            power_patch[n, m] = power
                            A_patch[n, m] = np.sqrt(power)

                            alpha_tilde = delta_px[0] * px_to_freq
                            beta_tilde = delta_px[1] * px_to_freq

                            gradients[n, m, 0] = -2 * np.pi * alpha_tilde
                            gradients[n, m, 1] = -2 * np.pi * beta_tilde

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

                    print(f"initial err: {best_err:.3f}px, delta={delta0}")

                    if best_err <= eps_px:
                        alpha_tilde, beta_tilde = best_p
                        A_patch[n, m] = np.sqrt(best_power)
                        gradients[n, m, 0] = -2 * np.pi * alpha_tilde
                        gradients[n, m, 1] = -2 * np.pi * beta_tilde
                        best_alphas[n, m] = alpha_tilde
                        best_betas[n, m] = beta_tilde
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

                        if com_new is not None:
                            com_patch[n, m] = com_new
                            power_patch[n, m] = power_new
                            data["coms"].append(com_new.copy())

                        if err_new < best_err:
                            best_err = err_new
                            best_p = p_new.copy()
                            best_power = power_new

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
                    A_patch[n, m] = np.sqrt(best_power)

                    gradients[n, m, 0] = -2 * np.pi * alpha_tilde
                    gradients[n, m, 1] = -2 * np.pi * beta_tilde

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

            calcUtils.saveNpz(data, "log/data_" + calcUtils.getTimestamp() + ".npz")

            return phi_map, gradients, A_patch

        finally:
            if old_camera_frames is not None:
                self.CAM.nFrames = old_camera_frames

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
    testCorrection()
    
    # data = calcUtils.loadNpz()
    
    server = Server("pt3",False, True)

    # server.mosaic = calcUtils.make_patch_mosaic_mask(data["best_alphas"], data["best_betas"],
    #                                                  data["S"], server.SLM.pitch, data["u0"], data["v0"], server.SLM.resolution)

    playground()
    # plt.show()

    # server.testU0_v0()

    server.CAM.set_integration_time(200e3)
    server.CAM.nFrames = 1
    server.calibrate_wavefront_pt3(S=None,focal_length=0.8, live_view=True,live_every=4, skip_gradient_search=False, max_iter=10)
    
    # plotUtils.plot_wavefront(phase)


    #TODO:   Phasemaske einfach als Mosaik von patches. ist dann zwar shifted aber sollte trotzdem bessers sein

    
    
