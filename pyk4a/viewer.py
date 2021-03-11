import time
import argparse
import numpy as np
import cv2
import pickle
import pyk4a
from pyk4a import Config, PyK4A, ColorResolution


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump_filepath')
    parser.add_argument('--dump_frames', type=int, default=100)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--undistort_color', action='store_true')
    parser.add_argument('--undistort_depth', action='store_true')
    parser.add_argument('--transformed_depth', action='store_true')
    parser.add_argument('--undistort_bt', action='store_true')
    parser.add_argument('--parallel_bt', action='store_true')
    parser.add_argument('--vis_color', action='store_true')
    parser.add_argument('--vis_depth', action='store_true')
    parser.add_argument('--vis_bt', action='store_true')
    parser.add_argument('--no_color', action='store_false')
    parser.add_argument('--no_depth', action='store_false')
    parser.add_argument('--no_bt', action='store_false')
    parsed_args = parser.parse_args()
    return parsed_args


def main():
    parsed_args = parse_args()

    if parsed_args.fps == 30:
        fps_config = pyk4a.FPS.FPS_30
    elif parsed_args.fps == 15:
        fps_config = pyk4a.FPS.FPS_15
    else:
        raise Exception(f'fps {parsed_args.fps} not found')

    k4a = PyK4A(Config(
        color_resolution=ColorResolution.RES_1536P,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        camera_fps=fps_config
    ))
    k4a.connect(lut=True)

    frame_num = -1

    is_dump = parsed_args.dump_filepath is not None and parsed_args.dump_frames is not None
    if is_dump:
        dump_data = {
            'frames': [],
            'cam_params': k4a.get_cam_params()
        }
    times = []
    verbose = parsed_args.fps

    get_color = parsed_args.no_color
    get_depth = parsed_args.no_depth
    get_bt = parsed_args.no_bt

    # num_results = get_bt * 2 + get_depth + get_color
    while True:
        frame_num += 1
        start = time.time()
        result = k4a.get_capture2(
            parallel_bt=parsed_args.parallel_bt,
            get_bt=get_bt, get_depth=get_depth, get_color=get_color,
            get_color_timestamp=get_color, get_depth_timestamp=get_depth,
            undistort_color=parsed_args.undistort_color,
            undistort_depth=parsed_args.undistort_depth,
            transformed_depth=parsed_args.transformed_depth,
            undistort_bt=parsed_args.undistort_bt,
            verbose=verbose
        )

        # if len(result.keys()) < num_results:
        #     print(f'frame_num: {frame_num}, result_items: {len(result.keys())}')
        times.append(time.time() - start)

        if verbose > 0:
            if len(times) > 10 and len(times) % verbose == 0:
                print(f'py fps: {1 / np.array(times[10:]).mean()}')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if is_dump:
            dump_data['frames'].append(result)
            if frame_num > parsed_args.dump_frames:
                break

        suff = lambda x: '_undistorted' if x else ''
        if parsed_args.vis_color:
            k = f'color{suff(parsed_args.undistort_color)}'
            if k in result:
                result_img = result[k][::4, ::4, :3]
                cv2.imshow(k, result_img)

        if parsed_args.vis_depth:
            pref = 'transformed_' if parsed_args.transformed_depth else ''
            k = f'{pref}depth{suff(parsed_args.undistort_depth)}'
            if k in result:
                result_img = result[k]
                result_img = result_img.astype(np.float32) / 1000
                result_img = (255 * result_img / 4).clip(0, 255).astype(np.uint8)
                cv2.imshow(k, result_img)

        if parsed_args.vis_bt:
            k = f'body_index_map{suff(parsed_args.undistort_bt)}'
            if k in result and 'pose' in result:
                result_img = result[k]
                cv2.imshow(k, result_img)

    k4a.disconnect()
    
    if is_dump:
        print('dumping...')
        with open(parsed_args.dump_filepath, 'wb') as f:
            pickle.dump(dump_data, f)
    else:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()










# source_H, source_W, _ = frame.shape
# frame = cv2.resize(frame, (256 * source_W // source_H, 256))
# target_H, target_W, _ = frame.shape

# if pose is not None and pose.shape[0] > 0:
#     pts = pose[0, :, :2].reshape(-1, 2)[kinect2coco]
#     pts[:, 0] = pts[:, 0]*(target_H/source_H)
#     pts[:, 1] = pts[:, 1]*(target_W/source_W)

#     for i in range(pts.shape[0]):
#         cv2.circle(frame, (int(pts[i, 0]), int(pts[i, 1])), 1, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)