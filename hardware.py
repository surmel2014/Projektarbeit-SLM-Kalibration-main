import os
import calcUtils
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".matplotlib"))

try:
    import cv2
except Exception:
    cv2 = None

try:
    import screeninfo
except Exception:
    screeninfo = None

try:
    import ximea.xiapi as xiapi
except Exception:
    xiapi = None

import matplotlib.pyplot as plt

from pt3_communication.pt3_comm_client import Communication_Class


class SLM:
    def __init__(
        self,
        resolution = (1024, 1024),
        waveLength = 633e-9,
        isSimulative = True,
        screen_id = 2,
        display_type = "holoeye_1920",
        window_name = "slm",
        correction_phase_map = None,
    ):
        self.display_width = None
        self.display_height = None

        if display_type == "holoeye_1920":
            self.resolution = (1080, 1920)
            self.pitch = 8e-6

        elif display_type == "pt3":
            self.resolution = (1440, 7680)
            self.display_width = self.resolution[1]
            self.display_height = self.resolution[0]
            self.pitch = (25e-6, 75e-6)  # x=25µm, y=75µm
            self.cluster = Communication_Class(ip_port = "192.168.240.131:3456")#localhost:50051")

        else:
            self.resolution = resolution
            self.pitch = pitch

        if not np.isscalar(self.pitch):
            self.pitch = tuple(self.pitch)
            if len(self.pitch) != 2:
                raise ValueError("pitch must be a scalar or a two-value tuple.")
            self.pitch = (float(self.pitch[0]), float(self.pitch[1]))
        self.isSimulative = isSimulative

        self.laserWavelength = waveLength
        self.screen_id = screen_id
        self.display_type = display_type
        self.window_name = window_name
        self.flag_show = False
        self.screen = None
        
        
        self.currentHolo = None
        self.requestedHolo = None
        self.currentUseCorrection = True
        self.correctionPhaseMap = None
        if correction_phase_map is not None:
            self.setCorrectionPhaseMap(correction_phase_map)

        if not self.isSimulative and not display_type == "pt3":
            self._initDisplay()
        
        
    def showHologram(self, holo, use_correction=True):
        self.requestedHolo = np.asarray(holo)
        self.currentUseCorrection = use_correction
        self.currentHolo = self._applyCorrectionPhaseMap(self.requestedHolo, use_correction)

        if self.isSimulative:
            self._SimShowHolo(self.currentHolo)
            
        else: 
            self._RealShowHolo(self.currentHolo)
    
    def showStackedField(self, amp = None, phase = None, field = None):
        if self.display_type is not "pt3" or self.screen_id is True:
            raise SystemError("Wrong Display type for stacked array or Simulative Mode is turned on" )

        if amp is not None and phase is not None:
            phase = phase / np.max(phase) * 255 
            amp = amp / np.max(amp) * 255 
            
            stack =  np.array([amp,phase], dtype=np.uint8)
            self.cluster.send_and_display_image(stack)
        elif field is not None:
            field = np.array(field)
            field[0] = field[0] / np.max(field[0]) *255 
            field[1] = field[1] / np.max(field[1]) *255 
            self.cluster.send_and_display_image(field.astype(np.uint8))

    def setCorrectionPhaseMap(self, correction_phase_map):
        correction_phase_map = np.asarray(correction_phase_map, dtype=float)
        if correction_phase_map.shape != self.resolution:
            raise ValueError(
                f"Expected correction phase map shape {self.resolution}, got {correction_phase_map.shape}."
            )
        if not np.all(np.isfinite(correction_phase_map)):
            raise ValueError("Correction phase map contains NaN or infinite values.")

        self.correctionPhaseMap = np.mod(correction_phase_map, 2 * np.pi)


    def clearCorrectionPhaseMap(self):
        self.correctionPhaseMap = None


    def _applyCorrectionPhaseMap(self, holo, use_correction=True):
        holo = np.asarray(holo)
        if holo.shape[:2] != self.resolution:
            raise ValueError(f"Expected hologram shape {self.resolution}, got {holo.shape}.")

        if not use_correction or self.correctionPhaseMap is None:
            return holo

        if np.iscomplexobj(holo):
            return holo * np.exp(1j * self.correctionPhaseMap)

        phase = holo.astype(float, copy=False)
        if not np.all(np.isfinite(phase)):
            raise ValueError("Hologram contains NaN or infinite values.")

        return np.mod(phase + self.correctionPhaseMap, 2 * np.pi)
            
    def _SimShowHolo(self, holo):
        self.currentHolo = holo
        
            
    def _RealShowHolo(self, holo):
        image = self._phaseToDisplayImage(holo)
        image = self._formatDisplayImage(image)

        if self.display_type == "pt3":
            self.cluster.send_and_display_image(image)
        else:
            if not self.flag_show:
                cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
                cv2.moveWindow(self.window_name, self.screen.x, self.screen.y)
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                self.flag_show = True

            cv2.imshow(self.window_name, image)
            cv2.waitKey(1)


    def _initDisplay(self):
        if cv2 is None:
            raise RuntimeError("OpenCV is not available. Install opencv-python or use SLM(isSimulative=True).")
        if screeninfo is None:
            raise RuntimeError("screeninfo is not available. Install screeninfo or use SLM(isSimulative=True).")

        screens = screeninfo.get_monitors()
        if self.screen_id < 1 or self.screen_id > len(screens):
            raise ValueError(f"screen_id must be between 1 and {len(screens)}.")

        self.screen = screens[self.screen_id - 1]
        self.display_width = self.screen.width
        self.display_height = self.screen.height


    def _phaseToDisplayImage(self, holo):
        holo = np.asarray(holo)
        if holo.shape[:2] != self.resolution:
            raise ValueError(f"Expected hologram shape {self.resolution}, got {holo.shape}.")

        if np.iscomplexobj(holo):
            phase = np.angle(holo)
            return np.mod(phase, 2 * np.pi) / (2 * np.pi)

        holo = holo.astype(float, copy=False)
        if not np.all(np.isfinite(holo)):
            raise ValueError("Hologram contains NaN or infinite values.")

        if np.min(holo) < 0 or np.max(holo) > 1:
            return np.mod(holo, 2 * np.pi) / (2 * np.pi)

        return holo


    def _formatDisplayImage(self, image):
        if self.display_type == "4k_RGB":
            image = np.flip(image, (0, 1))
            if image.ndim == 2:
                image = image[:, :, None]
            width_end = int(image.shape[1] / 3) * 3
            red = image[:, 0:width_end:3, 0]
            green = image[:, 1:width_end:3, 0]
            blue = image[:, 2:width_end:3, 0]
            image = np.stack((blue, green, red), axis=2)
            return self._centerCropPad(image)

        if self.display_type == "holoeye_1920":
            return self._centerCropPad(image)

        if self.display_type == "holoeye_1920_no_cover":
            return self._centerCropPad(np.flip(image, (0, 1)))

        if self.display_type == "crl_opto":
            return self._centerCropPad(image)

        if self.display_type == "5_5_zoll":
            if image.ndim == 2:
                image = image[:, :, None]
            image = np.transpose(image, (1, 0, 2))
            image = np.flip(image, 0)
            return self._centerCropPad(image)
        
        if self.display_type == "pt3":
            return self._centerCropPad(image)

        raise ValueError(f"Unknown display_type: {self.display_type}")


    def _centerCropPad(self, image):
        target_height = self.display_height
        target_width = self.display_width

        img_height = image.shape[0]
        img_width = image.shape[1]
        img_y_center = img_height // 2
        img_x_center = img_width // 2

        if img_height > target_height:
            start = img_y_center - target_height // 2
            end = start + target_height
            image = image[start:end, ...]
        elif img_height < target_height:
            pad_before = (target_height - img_height) // 2
            pad_after = target_height - img_height - pad_before
            image = np.pad(image, self._padSpec(image.ndim, 0, pad_before, pad_after))

        if img_width > target_width:
            start = img_x_center - target_width // 2
            end = start + target_width
            image = image[:, start:end, ...]
        elif img_width < target_width:
            pad_before = (target_width - img_width) // 2
            pad_after = target_width - img_width - pad_before
            image = np.pad(image, self._padSpec(image.ndim, 1, pad_before, pad_after))

        return image


    def _padSpec(self, ndim, axis, pad_before, pad_after):
        pad_spec = [(0, 0)] * ndim
        pad_spec[axis] = (pad_before, pad_after)
        return tuple(pad_spec)


    def scale(self, scale):
        if self.requestedHolo is None:
            raise RuntimeError("No hologram has been shown yet.")

        self.showHologram(self.requestedHolo * scale, use_correction=self.currentUseCorrection)


    def stop(self):
        self.flag_show = False
        if cv2 is not None:
            cv2.destroyWindow(self.window_name)
            
            
    def showAxicon(self, angle=0.001):
        holo = calcUtils.axiconPhase(self.resolution, axiconAngle=angle, wavelength=self.laserWavelength, pixelPitch=self.pitch)
        self.showHologram(holo)
        
    def showBlack(self):
        self.showHologram(np.zeros(shape=self.resolution))

    def __del__(self):
        if self.flag_show:
            try:
                self.stop()
            except Exception:
                pass

class CAM:
    def __init__(self, resolution = (1024, 1024), isSimulative = True):
        self.resolution = resolution
        self.pitch = 3.45e-6  #meters
        self.isSimulative = isSimulative
        self.cam = None
        self.img_buffer = None

        self.nFrames = 5
        self._integration_time_micros = 500.0
        self._last_frame_saturation_value = 255.0
        
        self.simImg = None
        self.simExtent = None
        if not self.isSimulative:
            if xiapi is None:
                raise RuntimeError("Ximea xiapi is not available. Use CAM(isSimulative=True) or install the Ximea camera driver/API.")
            print("test")
            try:
                self.cam = xiapi.Camera()
            except Exception as exc:
                raise RuntimeError("Could not initialize the Ximea camera. Check that the camera is connected and the Ximea driver/API is installed.") from exc
            self.cam.set_param("debug_level", "XI_DL_DISABLED")
            self.cam.open_device()
            self.cam.set_exposure(500)
            self.cam.set_gain(0)
            self.img_buffer = xiapi.Image()
            
            self.resolution = self._getCamResolution()
            
       
    
    def _getCamResolution(self):
        if self.isSimulative:
            return self.resolution
        else:
            self.cam.start_acquisition()
            self.cam.get_image(self.img_buffer)
            image = self.img_buffer.get_image_data_numpy()
            #image_flip = np.flip( np.flip( image, 0), 1)
            self.cam.stop_acquisition()
            return image.shape

    def set_integration_time(self, time_micros):
        """Set and remember the camera exposure time in microseconds."""
        time_micros = float(time_micros)
        if time_micros <= 0:
            raise ValueError("The integration time must be greater than zero.")
        if not self.isSimulative:
            self.cam.set_exposure(int(round(time_micros)))
            try:
                time_micros = float(self.cam.get_exposure())
            except (AttributeError, TypeError, ValueError):
                pass
        self._integration_time_micros = time_micros

    def get_integration_time(self):
        """Return the current camera exposure time in microseconds."""
        if not self.isSimulative:
            try:
                self._integration_time_micros = float(self.cam.get_exposure())
            except (AttributeError, TypeError, ValueError):
                pass
        return self._integration_time_micros

    def get_saturation_value(self):
        """Return the digital full-scale value of the most recent raw frame."""
        return self._last_frame_saturation_value
        
    
    def __del__(self):

        #self.module_logger.info('stop camera')
        if self.cam is not None:
            try:
                self.cam.stop_acquisition()
                self.cam.close_device()
            except Exception:
                pass
    
    def getCamImg(self):
        if self.isSimulative:
            if self.simImg is not None:
                sim_dtype = np.asarray(self.simImg).dtype
                if np.issubdtype(sim_dtype, np.integer):
                    self._last_frame_saturation_value = float(np.iinfo(sim_dtype).max)
                elif np.nanmax(self.simImg) <= 1.0:
                    self._last_frame_saturation_value = 1.0
            return self.simImg
        
        img_ls = []
        for i in range(self.nFrames):
            self.cam.start_acquisition()
            
            self.cam.get_image(self.img_buffer)
            image = self.img_buffer.get_image_data_numpy()
            if np.issubdtype(image.dtype, np.integer):
                self._last_frame_saturation_value = float(np.iinfo(image.dtype).max)
            #image_flip = np.flip( np.flip( image, 0), 1)
            self.cam.stop_acquisition()
            img_ls.append(image)
        image = np.mean(np.asarray(img_ls, dtype = np.float32),0)
        #self.cam.close_device()

        return image

    
    
    def getSpotPosition(self):
        """
        Wrapper function that directly outputs spot position

        Returns:
            _type_: _description_
        """
        
        img = self.getCamImg()
        spot = self.analyzeSpot(img)
        
        spotPosMeter = spot["centroid_m"]
        return spotPosMeter
    
    def analyzeSpot(self, image, subtract_background = True, threshold = None):
        """Calculate centroid and D4sigma spot diameter from a camera image."""
        intensity = np.asarray(image)
        if intensity.ndim != 2:
            raise ValueError(f"Expected a 2D camera image, got shape {intensity.shape}.")

        if np.iscomplexobj(intensity):
            intensity = np.abs(intensity) ** 2
        else:
            intensity = intensity.astype(float, copy = False)

        if not np.all(np.isfinite(intensity)):
            raise ValueError("Camera image contains NaN or infinite values.")

        if subtract_background:
            intensity = intensity - np.min(intensity)

        if threshold is not None:
            intensity = np.where(intensity >= threshold, intensity, 0.0)

        total_intensity = np.sum(intensity)
        if total_intensity <= 0:
            raise ValueError("Cannot analyze an empty or fully thresholded image.")

        y, x = np.indices(intensity.shape, dtype = float)
        centroid_x = np.sum(x * intensity) / total_intensity
        centroid_y = np.sum(y * intensity) / total_intensity

        sigma_x = np.sqrt(np.sum(((x - centroid_x) ** 2) * intensity) / total_intensity)
        sigma_y = np.sqrt(np.sum(((y - centroid_y) ** 2) * intensity) / total_intensity)

        d4sigma_x = 4.0 * sigma_x
        d4sigma_y = 4.0 * sigma_y

        return {
            "centroid_px": (float(centroid_x), float(centroid_y)),
            "centroid_m": (float(centroid_x * self.pitch), float(centroid_y * self.pitch)),
            "d4sigma_px": (float(d4sigma_x), float(d4sigma_y)),
            "d4sigma_m": (float(d4sigma_x * self.pitch), float(d4sigma_y * self.pitch)),
        }


if __name__ == "__main__":
    
    print("test 1")
    
    cam = CAM()
    
    img = cam.getCamImg()
    plt.imshow(img)
    plt.show()
    
