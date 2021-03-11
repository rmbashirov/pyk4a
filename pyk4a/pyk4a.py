import numpy as np
from typing import Tuple
import k4a_module
from enum import Enum
import time
from collections import defaultdict
import json
import operator
import threading
import queue

from pyk4a.config import Config, ColorControlMode, ColorControlCommand


# k4a_wait_result_t
class Result(Enum):
    Success = 0
    Failed = 1
    Timeout = 2


class K4AException(Exception):
    pass


class K4ATimeoutException(K4AException):
    pass


class PyK4A:
    TIMEOUT_WAIT_INFINITE = -1

    def __init__(self, config=Config(), device_id=0):
        self._device_id = device_id
        self._config = config
        self.is_running = False
        self.timings = defaultdict(list)
        self.skip_counts = []

    def __del__(self):
        if self.is_running:
            self.disconnect()

    def connect(self, lut=False):
        print('connecting...')
        self._device_open()
        self._start_cameras(lut)
        self.is_running = True
        self.counter = -1
        print('connected')

    def disconnect(self):
        self._stop_cameras()
        self._device_close()
        self.is_running = False

    def _device_open(self):
        res = k4a_module.device_open(self._device_id)
        self._verify_error(res)

    def _device_close(self):
        res = k4a_module.device_close()
        self._verify_error(res)

    def _start_cameras(self, lut=False):
        if lut:
            res = k4a_module.device_start_cameras_with_lut(*self._config.unpack())    
        else:
            res = k4a_module.device_start_cameras(*self._config.unpack())
        self._verify_error(res)

    def _stop_cameras(self):
        res = k4a_module.device_stop_cameras()
        self._verify_error(res)

    def get_capture(self, timeout=TIMEOUT_WAIT_INFINITE, color_only=False, transform_depth_to_color=True):
        """Get next capture
            Parameters:
                timeout: Timeout of capture
                color_only: If True, only color image will be returned
                transform_depth_to_color: If True, depth image will be transformed to the color image space
            Returns:
                color: Color image
                deoth: Depth image
                pose: Pose data of shape (num_bodies, num_joints, 10). Last dimension means:
                      0:2 - 2d keypoints projected to color image
                      2:5 - 3d keypoints in world depth image coordinates
                      5:9 - orientation
                      9 - confidence
                body_index_map: Body Index map is the body instance segmentation map. Each pixel maps to the corresponding pixel in the depth image or the ir image.
                                The value for each pixel represents which body the pixel belongs to. It can be either background (value 255)
                                or the index of a detected k4abt_body_t.
        """

        res = k4a_module.device_get_capture(timeout)
        self._verify_error(res)

        color = k4a_module.device_get_color_image()
        if color_only:
            return color

        # depth = k4a_module.device_get_depth_image(transform_depth_to_color)
        # pose, body_index_map = k4a_module.device_get_pose_data()

        return color, depth, pose, body_index_map

    def get_cam_params(self):
        res = k4a_module.device_get_cam_params()
        res = json.loads(res)
        return res

    def get_serial_number(self):
        get_serial_number = k4a_module.device_get_serial_number()
        return get_serial_number

    def get_capture2(
        self,
        skip_old_atol_ms=None, parallel_bt=True,
        get_bt=True, get_depth=True, get_color=True,
        get_color_timestamp=True, get_depth_timestamp=True,
        undistort_color=True, undistort_depth=True, undistort_bt=True,
        transformed_depth=True,
        timeout=TIMEOUT_WAIT_INFINITE, verbose=0
    ):
        self.counter += 1
        skip_count = 0
        while True:
            t0 = time.time()
            res = k4a_module.device_get_capture(timeout)
            self._verify_error(res)
            t1 = time.time()

            if skip_old_atol_ms is None or (t1 - t0) * 1000 > skip_old_atol_ms:
                break
            else:
                skip_count += 1

        result = dict()
        result['skip_count'] = skip_count
        if get_color:
            if undistort_color:
                k4a_module.device_get_color_image_undistorted_start()
            else:
                k4a_module.device_get_color_image_start()
        if get_depth:
            if undistort_depth:
                if transformed_depth:
                    k4a_module.device_get_transformed_depth_image_undistorted_start()
                else:
                    k4a_module.device_get_depth_image_undistorted_start()
            else:
                if transformed_depth:
                    k4a_module.device_get_transformed_depth_image_start()
                else:
                    raise Exception('get depth_image not implemented')
        if parallel_bt:
            if get_bt:
                if undistort_bt:
                    k4a_module.device_get_pose_data_undistorted_start()
                else:
                    k4a_module.device_get_pose_data_start()
            t2 = time.time()

        suff = lambda x: '_undistorted' if x else ''
        if get_depth:
            if undistort_depth:
                if transformed_depth:
                    depth_result = k4a_module.device_get_transformed_depth_image_undistorted_join()
                else:
                    depth_result = k4a_module.device_get_depth_image_undistorted_join()
            else:
                if transformed_depth:
                    depth_result = k4a_module.device_get_transformed_depth_image_join()
                else:
                    raise Exception('get depth_image not implemented')
            if depth_result is not None:
                pref = 'tranfromed_' if transformed_depth else ''
                result[f'{pref}depth{suff(undistort_depth)}'] = depth_result
                if get_depth_timestamp:
                    result['depth_timestamp'] = k4a_module.device_get_depth_image_device_timestamp_usec()
            else:
                print('depth_result None')
        if get_color:
            color_result = k4a_module.device_get_color_image_undistorted_join() \
                if undistort_color else k4a_module.device_get_color_image_join()
            if color_result is not None:
                result[f'color{suff(undistort_color)}'] = color_result
                if get_color_timestamp:
                    result['color_timestamp'] = k4a_module.device_get_color_image_device_timestamp_usec()
        if not parallel_bt:
            t2 = time.time()
        if get_bt:
            if not parallel_bt:
                if undistort_bt:
                    k4a_module.device_get_pose_data_undistorted_start()
                else:
                    k4a_module.device_get_pose_data_start()
            pose, body_index_map_result = k4a_module.device_get_pose_data_undistorted_join() \
                if undistort_bt else k4a_module.device_get_pose_data_join()
            if not (pose is None or body_index_map_result is None):
                result['pose'] = pose
                result[f'body_index_map{suff(undistort_bt)}'] = body_index_map_result
        t3 = time.time()

        self.timings['get_capture2_0'].append(t1 - t0)
        self.timings['get_capture2_1'].append(t2 - t1)
        self.timings['get_capture2_2'].append(t3 - t2)
        self.skip_counts.append(skip_count)

        if verbose > 0 and self.counter % verbose == 0:
            timings_s = 'k4a timings:\n'
            min_frames = 10
            durations = dict()
            for k, v in sorted(self.timings.items(), key=operator.itemgetter(0)):
                if len(v) > 2 * min_frames:
                    start = int(len(v) / 2)
                    duration = np.array(v)[start:].mean()
                    timings_s += f'\t{k}: {duration * 1000:.3f}ms\n'
                    durations[k] = duration
            if skip_old_atol_ms is not None and len(self.skip_counts) > 2 * min_frames:
                start = int(len(self.skip_counts) / 2)
                timings_s += f'\tskip_count mean: {np.mean(self.skip_counts[start:]):.2f}\n'
            if len(durations) > 0:
                fps = 1 / (durations['get_capture2_0'] + durations['get_capture2_1'] + durations['get_capture2_2'])
                timings_s += f'\tfps: {fps:.1f}\n'

            if len(durations) > 0:
                print(timings_s)
        
        return result

    @property
    def sync_jack_status(self) -> Tuple[bool, bool]:
        res, jack_in, jack_out = k4a_module.device_get_sync_jack()
        self._verify_error(res)
        return jack_in == 1, jack_out == 1

    def _get_color_control(self, cmd: ColorControlCommand) -> Tuple[int, ColorControlMode]:
        res, mode, value = k4a_module.device_get_color_control(cmd)
        self._verify_error(res)
        return value, ColorControlMode(mode)

    def _set_color_control(self, cmd: ColorControlCommand, value: int, mode=ColorControlMode.MANUAL):
        res = k4a_module.device_set_color_control(cmd, mode, value)
        self._verify_error(res)

    @property
    def brightness(self) -> int:
        return self._get_color_control(ColorControlCommand.BRIGHTNESS)[0]

    @property
    def contrast(self) -> int:
        return self._get_color_control(ColorControlCommand.CONTRAST)[0]

    @property
    def saturation(self) -> int:
        return self._get_color_control(ColorControlCommand.SATURATION)[0]

    @property
    def sharpness(self) -> int:
        return self._get_color_control(ColorControlCommand.SHARPNESS)[0]

    @property
    def backlight_compensation(self) -> int:
        return self._get_color_control(ColorControlCommand.BACKLIGHT_COMPENSATION)[0]

    @property
    def gain(self) -> int:
        return self._get_color_control(ColorControlCommand.GAIN)[0]

    @property
    def powerline_frequency(self) -> int:
        return self._get_color_control(ColorControlCommand.POWERLINE_FREQUENCY)[0]

    @property
    def exposure(self) -> int:
        # sets mode to manual
        return self._get_color_control(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE)[0]

    @property
    def exposure_mode_auto(self) -> bool:
        return self._get_color_control(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE)[1] == ColorControlMode.AUTO

    @property
    def whitebalance(self) -> int:
        # sets mode to manual
        return self._get_color_control(ColorControlCommand.WHITEBALANCE)[0]

    @property
    def whitebalance_mode_auto(self) -> bool:
        return self._get_color_control(ColorControlCommand.WHITEBALANCE)[1] == ColorControlMode.AUTO

    @brightness.setter
    def brightness(self, value: int):
        self._set_color_control(ColorControlCommand.BRIGHTNESS, value)

    @contrast.setter
    def contrast(self, value: int):
        self._set_color_control(ColorControlCommand.CONTRAST, value)

    @saturation.setter
    def saturation(self, value: int):
        self._set_color_control(ColorControlCommand.SATURATION, value)

    @sharpness.setter
    def sharpness(self, value: int):
        self._set_color_control(ColorControlCommand.SHARPNESS, value)

    @backlight_compensation.setter
    def backlight_compensation(self, value: int):
        self._set_color_control(ColorControlCommand.BACKLIGHT_COMPENSATION, value)

    @gain.setter
    def gain(self, value: int):
        self._set_color_control(ColorControlCommand.GAIN, value)

    @powerline_frequency.setter
    def powerline_frequency(self, value: int):
        self._set_color_control(ColorControlCommand.POWERLINE_FREQUENCY, value)

    @exposure.setter
    def exposure(self, value: int):
        self._set_color_control(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE, value)

    @exposure_mode_auto.setter
    def exposure_mode_auto(self, mode_auto: bool, value=2500):
        mode = ColorControlMode.AUTO if mode_auto else ColorControlMode.MANUAL
        self._set_color_control(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE, value=value, mode=mode)

    @whitebalance.setter
    def whitebalance(self, value: int, mode=ColorControlMode.MANUAL):
        self._set_color_control(ColorControlCommand.WHITEBALANCE, value)

    @whitebalance_mode_auto.setter
    def whitebalance_mode_auto(self, mode_auto: bool, value=2500):
        mode = ColorControlMode.AUTO if mode_auto else ColorControlMode.MANUAL
        self._set_color_control(ColorControlCommand.WHITEBALANCE, value=value, mode=mode)

    @staticmethod
    def _verify_error(res):
        res = Result(res)
        if res == Result.Failed:
            raise K4AException()
        elif res == Result.Timeout:
            raise K4ATimeoutException()


if __name__ == "__main__":
    k4a = PyK4A(Config())
    k4a.connect()
    print("Connected")
    jack_in, jack_out = k4a.get_sync_jack()
    print("Jack status : in -> {} , out -> {}".format(jack_in, jack_out))
    for _ in range(10):
        color, depth = k4a.device_get_capture(color_only=False)
        print(color.shape, depth.shape)
    k4a.disconnect()
    print("Disconnected")
