import pyk4a
from pyk4a import Config, PyK4A, ColorResolution
import argparse
import os
import os.path as osp
import json


def main(dp, device_id=0):
    fps_config = pyk4a.FPS.FPS_30
    k4a = PyK4A(Config(
        color_resolution=ColorResolution.RES_1536P,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        camera_fps=fps_config
    ), device_id=device_id)
    k4a.connect(lut=True)
    cam_params = k4a.get_cam_params()
    serial_number = k4a.get_serial_number()
    assert serial_number is not None
    os.makedirs(dp, exist_ok=True)
    with open(osp.join(dp, f'{serial_number}.json'), 'w') as f:
        json.dump(cam_params, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dirpath', required=True)
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()
    main(args.output_dirpath, args.device_id)
