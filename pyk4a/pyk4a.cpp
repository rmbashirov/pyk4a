#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <future>

#include <k4a/k4a.h>
#include <k4abt.h>

#include "turbojpeg.h"
// #include "NvPipe.h"
// #include "libgpujpeg/gpujpeg.h"

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
// #include <opencv2/highgui.hpp>
// #include <opencv2/opencv.hpp>
// #include <opencv2/rgbd.hpp>

#include <nlohmann/json.hpp>
using namespace nlohmann;

#include <Python.h>
#include <numpy/arrayobject.h>

using namespace std::chrono;
using namespace std;
//#ifdef HAVE_OPENCV
using namespace cv;
//#endif



// ######################## UNDISTORT ########################

#define INVALID INT32_MIN
typedef struct _pinhole_t
{
    float px;
    float py;
    float fx;
    float fy;

    int width;
    int height;
} pinhole_t;

typedef struct _coordinate_t
{
    int x;
    int y;
    float weight[4];
} coordinate_t;

typedef enum
{
    INTERPOLATION_NEARESTNEIGHBOR, /**< Nearest neighbor interpolation */
    INTERPOLATION_BILINEAR,        /**< Bilinear interpolation */
    INTERPOLATION_BILINEAR_DEPTH   /**< Bilinear interpolation with invalidation when neighbor contain invalid
                                                data with value 0 */
} interpolation_t;


#include <tgmath.h>

void compute_xy_range(const k4a_calibration_t* calibration,
                             const k4a_calibration_type_t camera,
                             const int width,
                             const int height,
                             float& x_min,
                             float& x_max,
                             float& y_min,
                             float& y_max)
{
    // Step outward from the centre point until we find the bounds of valid projection
    const float step_u = 0.25f;
    const float step_v = 0.25f;
    const float min_u = 0;
    const float min_v = 0;
    const float max_u = (float)width - 1;
    const float max_v = (float)height - 1;
    const float center_u = 0.5f * width;
    const float center_v = 0.5f * height;

    int valid;
    k4a_float2_t p;
    k4a_float3_t ray;

    // search x_min
    for (float uv[2] = { center_u, center_v }; uv[0] >= min_u; uv[0] -= step_u)
    {
        p.xy.x = uv[0];
        p.xy.y = uv[1];
        k4a_calibration_2d_to_3d(calibration, &p, 1.f, camera, camera, &ray, &valid);

        if (!valid)
        {
            break;
        }
        x_min = ray.xyz.x;
    }

    // search x_max
    for (float uv[2] = { center_u, center_v }; uv[0] <= max_u; uv[0] += step_u)
    {
        p.xy.x = uv[0];
        p.xy.y = uv[1];
        k4a_calibration_2d_to_3d(calibration, &p, 1.f, camera, camera, &ray, &valid);

        if (!valid)
        {
            break;
        }
        x_max = ray.xyz.x;
    }

    // search y_min
    for (float uv[2] = { center_u, center_v }; uv[1] >= min_v; uv[1] -= step_v)
    {
        p.xy.x = uv[0];
        p.xy.y = uv[1];
        k4a_calibration_2d_to_3d(calibration, &p, 1.f, camera, camera, &ray, &valid);

        if (!valid)
        {
            break;
        }
        y_min = ray.xyz.y;
    }

    // search y_max
    for (float uv[2] = { center_u, center_v }; uv[1] <= max_v; uv[1] += step_v)
    {
        p.xy.x = uv[0];
        p.xy.y = uv[1];
        k4a_calibration_2d_to_3d(calibration, &p, 1.f, camera, camera, &ray, &valid);

        if (!valid)
        {
            break;
        }
        y_max = ray.xyz.y;
    }
}

pinhole_t create_pinhole_from_xy_range(const k4a_calibration_t* calibration, const k4a_calibration_type_t camera)
{
    int width = calibration->depth_camera_calibration.resolution_width;
    int height = calibration->depth_camera_calibration.resolution_height;
    if (camera == K4A_CALIBRATION_TYPE_COLOR)
    {
        width = calibration->color_camera_calibration.resolution_width;
        height = calibration->color_camera_calibration.resolution_height;
    }

    float x_min = 0, x_max = 0, y_min = 0, y_max = 0;
    compute_xy_range(calibration, camera, width, height, x_min, x_max, y_min, y_max);

    pinhole_t pinhole;

    float fx = 1.f / (x_max - x_min);
    float fy = 1.f / (y_max - y_min);
    float px = -x_min * fx;
    float py = -y_min * fy;

    pinhole.fx = fx * width;
    pinhole.fy = fy * height;
    pinhole.px = px * width;
    pinhole.py = py * height;
    pinhole.width = width;
    pinhole.height = height;

    return pinhole;
}


void create_undistortion_lut(const k4a_calibration_t* calibration,
                                    const k4a_calibration_type_t camera,
                                    const pinhole_t* pinhole,
                                    k4a_image_t lut,
                                    interpolation_t type)
{
    coordinate_t* lut_data = (coordinate_t*)(void*)k4a_image_get_buffer(lut);

    k4a_float3_t ray;
    ray.xyz.z = 1.f;

    int src_width = calibration->depth_camera_calibration.resolution_width;
    int src_height = calibration->depth_camera_calibration.resolution_height;
    if (camera == K4A_CALIBRATION_TYPE_COLOR)
    {
        src_width = calibration->color_camera_calibration.resolution_width;
        src_height = calibration->color_camera_calibration.resolution_height;
    }

    for (int y = 0, idx = 0; y < pinhole->height; y++)
    {
        ray.xyz.y = ((float)y - pinhole->py) / pinhole->fy;

        for (int x = 0; x < pinhole->width; x++, idx++)
        {
            ray.xyz.x = ((float)x - pinhole->px) / pinhole->fx;

            k4a_float2_t distorted;
            int valid;
            k4a_calibration_3d_to_2d(calibration, &ray, camera, camera, &distorted, &valid);

            coordinate_t src;
            if (type == INTERPOLATION_NEARESTNEIGHBOR)
            {
                // Remapping via nearest neighbor interpolation
                src.x = (int)floor(distorted.xy.x + 0.5f);
                src.y = (int)floor(distorted.xy.y + 0.5f);
            }
            else if (type == INTERPOLATION_BILINEAR || type == INTERPOLATION_BILINEAR_DEPTH)
            {
                // Remapping via bilinear interpolation
                src.x = (int)floor(distorted.xy.x);
                src.y = (int)floor(distorted.xy.y);
            }
            else
            {
                printf("Unexpected interpolation type!\n");
                exit(-1);
            }

            if (valid && src.x >= 0 && src.x < src_width && src.y >= 0 && src.y < src_height)
            {
                lut_data[idx] = src;

                if (type == INTERPOLATION_BILINEAR || type == INTERPOLATION_BILINEAR_DEPTH)
                {
                    // Compute the floating point weights, using the distance from projected point src to the
                    // image coordinate of the upper left neighbor
                    float w_x = distorted.xy.x - src.x;
                    float w_y = distorted.xy.y - src.y;
                    float w0 = (1.f - w_x) * (1.f - w_y);
                    float w1 = w_x * (1.f - w_y);
                    float w2 = (1.f - w_x) * w_y;
                    float w3 = w_x * w_y;

                    // Fill into lut
                    lut_data[idx].weight[0] = w0;
                    lut_data[idx].weight[1] = w1;
                    lut_data[idx].weight[2] = w2;
                    lut_data[idx].weight[3] = w3;
                }
            }
            else
            {
                lut_data[idx].x = INVALID;
                lut_data[idx].y = INVALID;
            }
        }
    }
}

template <typename T>
void k4a_remap(const k4a_image_t src, const k4a_image_t lut, k4a_image_t dst, interpolation_t type)
{
    int src_width = k4a_image_get_width_pixels(src);
    int dst_width = k4a_image_get_width_pixels(dst);
    int dst_height = k4a_image_get_height_pixels(dst);

    T* src_data = (T*)(void*)k4a_image_get_buffer(src);
    T* dst_data = (T*)(void*)k4a_image_get_buffer(dst);
    coordinate_t* lut_data = (coordinate_t*)(void*)k4a_image_get_buffer(lut);

    memset(dst_data, 0, (size_t)dst_width * (size_t)dst_height * sizeof(T));

    for (int i = 0; i < dst_width * dst_height; i++)
    {
        if (lut_data[i].x != INVALID && lut_data[i].y != INVALID)
        {
            if (type == INTERPOLATION_NEARESTNEIGHBOR)
            {
                dst_data[i] = src_data[lut_data[i].y * src_width + lut_data[i].x];
            }
            else if (type == INTERPOLATION_BILINEAR || type == INTERPOLATION_BILINEAR_DEPTH)
            {
                const T neighbors[4]{ src_data[lut_data[i].y * src_width + lut_data[i].x],
                                             src_data[lut_data[i].y * src_width + lut_data[i].x + 1],
                                             src_data[(lut_data[i].y + 1) * src_width + lut_data[i].x],
                                             src_data[(lut_data[i].y + 1) * src_width + lut_data[i].x + 1] };

                // If the image contains invalid data, e.g. depth image contains value 0, ignore the bilinear
                // interpolation for current target pixel if one of the neighbors contains invalid data to avoid
                // introduce noise on the edge. If the image is color or ir images, user should use
                // INTERPOLATION_BILINEAR
                if (type == INTERPOLATION_BILINEAR_DEPTH)
                {
                    // If the image contains invalid data, e.g. depth image contains value 0, ignore the bilinear
                    // interpolation for current target pixel if one of the neighbors contains invalid data to avoid
                    // introduce noise on the edge. If the image is color or ir images, user should use
                    // INTERPOLATION_BILINEAR
                    if (neighbors[0] == 0 || neighbors[1] == 0 || neighbors[2] == 0 || neighbors[3] == 0)
                    {
                        continue;
                    }

                    // Ignore interpolation at large depth discontinuity without disrupting slanted surface
                    // Skip interpolation threshold is estimated based on the following logic:
                    // - angle between two pixels is: theta = 0.234375 degree (120 degree / 512) in binning resolution
                    // mode
                    // - distance between two pixels at same depth approximately is: A ~= sin(theta) * depth
                    // - distance between two pixels at highly slanted surface (e.g. alpha = 85 degree) is: B = A /
                    // cos(alpha)
                    // - skip_interpolation_ratio ~= sin(theta) / cos(alpha)
                    // We use B as the threshold that to skip interpolation if the depth difference in the triangle is
                    // larger than B. This is a conservative threshold to estimate largest distance on a highly slanted
                    // surface at given depth, in reality, given distortion, distance, resolution difference, B can be
                    // smaller
                    const float skip_interpolation_ratio = 0.04693441759f;
                    float depth_min = min(min(neighbors[0], neighbors[1]), min(neighbors[2], neighbors[3]));
                    float depth_max = max(max(neighbors[0], neighbors[1]), max(neighbors[2], neighbors[3]));
                    float depth_delta = depth_max - depth_min;
                    float skip_interpolation_threshold = skip_interpolation_ratio * depth_min;
                    if (depth_delta > skip_interpolation_threshold)
                    {
                        continue;
                    }
                }

                dst_data[i] = (T)(neighbors[0] * lut_data[i].weight[0] + neighbors[1] * lut_data[i].weight[1] +
                                         neighbors[2] * lut_data[i].weight[2] + neighbors[3] * lut_data[i].weight[3] +
                                         0.5f);
            }
            else
            {
                printf("Unexpected interpolation type!\n");
                exit(-1);
            }
        }
    }
}

template void k4a_remap<uint16_t>(const k4a_image_t src, const k4a_image_t lut, k4a_image_t dst, interpolation_t type);
template void k4a_remap<uint8_t>(const k4a_image_t src, const k4a_image_t lut, k4a_image_t dst, interpolation_t type);

template<typename T> 
Mat create_mat_from_buffer(T *data, int width, int height, int channels = 1) {
    // printf("%d %d\n", CV_MAKETYPE(DataType<T>::type, channels), CV_16UC1);
    Mat mat(height, width, CV_MAKETYPE(DataType<T>::type, channels));
    memcpy(mat.data, data, width * height * channels * sizeof(T));
    return mat;
}
template Mat create_mat_from_buffer<uint16_t>(uint16_t *data, int width, int height, int channels);
template Mat create_mat_from_buffer<uint8_t>(uint8_t *data, int width, int height, int channels);

// template<typename T> cv::Mat create_mat_from_buffer(T *data, int width, int height, int channels = 1);

// ######################## UNDISTORT ########################


class JPEGDecoder {
public:
    tjhandle tjHandle = NULL;
    JPEGDecoder() {
        // tjHandle = tjInitDecompress();
    }

    bool decode(const k4a_image_t color_image, uint8_t* buffer, int w, int h) {
        if (tjHandle == NULL) {
            tjHandle = tjInitDecompress();
        }

        k4a_image_format_t format;
        format = k4a_image_get_format(color_image);
        if (format != K4A_IMAGE_FORMAT_COLOR_MJPG) {
            printf("color format not supported\n");
            return false;
        }

        uint8_t* src_buffer = k4a_image_get_buffer(color_image);
        if (src_buffer == NULL) {
            printf("k4a_image_get_buffer NULL\n");
            return false;
        }
        int _width, _height, _jpegsubsamp;
        tjDecompressHeader2(
            tjHandle, 
            src_buffer, 
            static_cast<unsigned long>(k4a_image_get_size(color_image)), 
            &_width, 
            &_height, 
            &_jpegsubsamp
        );

        // printf("s, %d %d %d %d\n", _width, _height, _jpegsubsamp, k4a_image_get_size(color_image));
        // printf("e, %d %d\n", w, h);

        if (_width != w || _height != h) {
            printf("_width != w || _height != h\n");
            return false;
        }

        int error = tjDecompress2(
            tjHandle,
            (uint8_t*)k4a_image_get_buffer(color_image),
            static_cast<unsigned long>(k4a_image_get_size(color_image)),
            buffer,
            w,
            0, // pitch
            h,
            TJPF_BGRA,
            TJFLAG_FASTDCT | TJFLAG_FASTUPSAMPLE
        );
        if (error != 0) {
            printf("tjDecompress2 error %d\n", error);
            if (tjDestroy(tjHandle) != 0) {
                printf("Failed to destroy turboJPEG handle\n");
            } else {
                tjHandle = tjInitDecompress();
                printf("created new turboJPEG handle\n");
            }
            return false;
        }

        return true;
    }

    ~JPEGDecoder() {
        if (tjDestroy(tjHandle)) {
            std::cout << "tjDestroy error" << std::endl;
        }
    }
};


// class NvJPEGDecoder {
// public:
//     struct gpujpeg_decoder* decoder;
//     struct gpujpeg_decoder_output decoder_output;

//     NvJPEGDecoder() {}
    
//     bool decode(const k4a_image_t color_image) {
//         if (decoder == NULL) {
//             gpujpeg_init_device(0, 0);
//             printf("gpujpeg_init_device %d\n", 0);

//             decoder = gpujpeg_decoder_create(NULL);
//             if (decoder == NULL ) {
//                 printf("cannot create decoder\n");
//                 return false;
//             }

//             // struct gpujpeg_image_parameters param_image;
//             // gpujpeg_image_set_default_parameters(&param_image);
//             // param_image.width = w;
//             // param_image.height = h;
//             // param_image.comp_count = 3;

//             // decoder = gpujpeg_decoder_create(&param_image);
            

//             // struct gpujpeg_parameters param;
//             // gpujpeg_set_default_parameters(&param);
//             // param.restart_interval = 16; 
//             // param.interleaved = 1;

//             // gpujpeg_decoder_init(decoder, &param, &param_image);
//             // decoder->coder.param_image.color_space = GPUJPEG_RGB;

//             gpujpeg_decoder_output_set_default(&decoder_output);

//             printf("decoder initialized\n");
//         }

//         k4a_image_format_t format;
//         format = k4a_image_get_format(color_image);
//         if (format != K4A_IMAGE_FORMAT_COLOR_MJPG) {
//             printf("color format not supported\n");
//             return false;
//         }

//         int image_decompressed_size = 0;
//         int decoder_error = gpujpeg_decoder_decode(
//             decoder, 
//             k4a_image_get_buffer(color_image), 
//             static_cast<unsigned long>(k4a_image_get_size(color_image)), 
//             &decoder_output
//         );
//         if (decoder_error != 0) {
//             printf("gpujpeg_decoder_decode error\n");
//             return false;
//         }

//         return true;
//     }

//     ~NvJPEGDecoder() {
//     }
// };

json get_intrinsics(k4a_calibration_intrinsic_parameters_t * intrinsics) {
    json result;
    result["fx"] = intrinsics->param.fx;
    result["fy"] = intrinsics->param.fy;
    result["cx"] = intrinsics->param.cx;
    result["cy"] = intrinsics->param.cy;

    result["k1"] = intrinsics->param.k1;
    result["k2"] = intrinsics->param.k2;
    result["p1"] = intrinsics->param.p1;
    result["p2"] = intrinsics->param.p2;
    result["k3"] = intrinsics->param.k3;
    result["k4"] = intrinsics->param.k4;
    result["k5"] = intrinsics->param.k5;
    result["k6"] = intrinsics->param.k6;
    return result;
}


void output_intrinsics(
    int w_color, int h_color,
    int w_depth, int h_depth,
    k4a_calibration_intrinsic_parameters_t * depth_intrinsics,
    k4a_calibration_intrinsic_parameters_t * rgb_intrinsics,
    Mat t_vec, Mat r_vec,
    json& json_output
) {
    json color_resolution;
    color_resolution["w"] = w_color;
    color_resolution["h"] = h_color;
    json_output["color_resolution"] = color_resolution;

    json depth_resolution;
    depth_resolution["w"] = w_depth;
    depth_resolution["h"] = h_depth;
    json_output["depth_resolution"] = depth_resolution;

    json_output["depth_to_rgb"] = json::array();

    json depth_to_rgb_json, r_json, t_json;
    for (int i = 0; i < 3; i++) {
        r_json.push_back(r_vec.at<float>(i));
    }
    depth_to_rgb_json["r"] = r_json;

    for (int i = 0; i < 3; i++) {
        t_json.push_back(t_vec.at<float>(i));
    }
    depth_to_rgb_json["t"] = t_json;
    json_output["depth_to_rgb"] = depth_to_rgb_json;

    json_output["depth_intrinsics"] = get_intrinsics(depth_intrinsics);
    json_output["rgb_intrinsics"] = get_intrinsics(rgb_intrinsics);
}

void output_pinhole(const std::string& label, pinhole_t& pinhole, json& json_output) {
    json tmp;
    tmp["fx"] = pinhole.fx;
    tmp["fy"] = pinhole.fy;
    tmp["px"] = pinhole.px;
    tmp["py"] = pinhole.py;
    json_output[label] = tmp;
}


void output_cam_mat(const std::string& label, Mat& cam_mat, json& json_output) {
    json tmp;
    tmp["fx"] = cam_mat.at<float>(0,0);
    tmp["fy"] = cam_mat.at<float>(1,1);
    tmp["cx"] = cam_mat.at<float>(0,2);
    tmp["cy"] = cam_mat.at<float>(1,2);
    json_output[label] = tmp;
}


#ifdef __cplusplus
extern "C" {
#endif
    k4a_capture_t capture;
    k4a_transformation_t transformation_handle;
    k4a_device_t device;
    k4abt_tracker_t tracker;
    k4a_calibration_t calibration;
    k4a_image_t depth_lut;
    int COLOR_CHANNELS = 4;
    int NUM_JOINTS = 32;
    int NUM_DATA = 10;
    int counter = 0;
    int w_color, h_color, w_depth, h_depth;
    vector<float> _camera_matrix_rgb;
    vector<float> _dist_coeffs_rgb;
    Mat camera_matrix_rgb;
    Mat dist_coeffs_rgb;
    Mat new_camera_matrix_rgb;
    Mat color_undistort_map_1, color_undistort_map_2;
    // NvJPEGDecoder decoder;
    JPEGDecoder decoder;
    json cam_params;

    std::future<void> device_get_pose_data_future;
    bool device_get_pose_data_result_0;
    int device_get_pose_data_result_1;
    double_t* device_get_pose_data_result_2;
    uint8_t* device_get_pose_data_result_3;

    std::future<void> device_get_pose_data_undistorted_future;
    bool device_get_pose_data_undistorted_result_0;
    int device_get_pose_data_undistorted_result_1;
    double_t* device_get_pose_data_undistorted_result_2;
    uint8_t* device_get_pose_data_undistorted_result_3;


    std::future<void> device_get_color_image_future;
    bool device_get_color_image_result_0;
    uint8_t* device_get_color_image_result_1;

    std::future<void> device_get_color_image_undistorted_future;
    bool device_get_color_image_undistorted_result_0;
    uint8_t* device_get_color_image_undistorted_result_1;
    uint64_t color_image_device_timestamp_usec;


    std::future<void> device_get_depth_image_undistorted_future;
    bool device_get_depth_image_undistorted_result_0;
    uint16_t* device_get_depth_image_undistorted_result_1;

    std::future<void> device_get_transformed_depth_image_future;
    bool device_get_transformed_depth_image_result_0;
    uint16_t* device_get_transformed_depth_image_result_1;

    std::future<void> device_get_transformed_depth_image_undistorted_future;
    bool device_get_transformed_depth_image_undistorted_result_0;
    uint16_t* device_get_transformed_depth_image_undistorted_result_1;
    uint64_t depth_image_device_timestamp_usec;


    static PyObject* device_open(PyObject* self, PyObject* args){
        int device_id;
        PyArg_ParseTuple(args, "I", &device_id);
        k4a_result_t result = k4a_device_open(device_id, &device);
        return Py_BuildValue("I", result);
    }

    static PyObject* device_close(PyObject* self, PyObject* args){
        k4a_device_close(device);
        return Py_BuildValue("I", K4A_RESULT_SUCCEEDED);
    }

    static PyObject* device_get_sync_jack(PyObject* self, PyObject* args){
        bool in_jack = 0;
        bool out_jack = 0;
        k4a_result_t result = k4a_device_get_sync_jack(device, &in_jack, &out_jack);

        return Py_BuildValue("III", result, in_jack, out_jack);
    }

    static PyObject* device_get_color_control(PyObject* self, PyObject* args){
        k4a_color_control_command_t command;
        k4a_color_control_mode_t mode;
        int32_t value = 0;
        PyArg_ParseTuple(args, "I", &command);

        k4a_result_t result = k4a_device_get_color_control(device, command, &mode, &value);
        if (result == K4A_RESULT_FAILED) {
            return Py_BuildValue("III", 0, 0, 0);
        }
        return Py_BuildValue("III", result, mode, value);
    }

    static PyObject* device_set_color_control(PyObject* self, PyObject* args){
        k4a_color_control_command_t command = K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE;
        k4a_color_control_mode_t mode = K4A_COLOR_CONTROL_MODE_MANUAL;
        int32_t value = 0;
        PyArg_ParseTuple(args, "III", &command, &mode, &value);

        k4a_result_t result = k4a_device_set_color_control(device, command, mode, value);
        if (result == K4A_RESULT_FAILED) {
            return Py_BuildValue("I", K4A_RESULT_FAILED);
        }
        return Py_BuildValue("I", result);
    }

    static PyObject* device_start_cameras(PyObject* self, PyObject* args){
        k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
        PyArg_ParseTuple(args, "IIIIIIIII", &config.color_format,
                &config.color_resolution, &config.depth_mode,
                &config.camera_fps, &config.synchronized_images_only,
                &config.depth_delay_off_color_usec, &config.wired_sync_mode,
                &config.subordinate_delay_off_master_usec,
                &config.disable_streaming_indicator);

        k4a_result_t result;
      	k4abt_tracker_configuration_t tracker_calibration = K4ABT_TRACKER_CONFIG_DEFAULT;
        result = k4a_device_get_calibration(device, config.depth_mode,
                config.color_resolution, &calibration);
        if (result == K4A_RESULT_FAILED) {
            return Py_BuildValue("I", K4A_RESULT_FAILED);
        }
        transformation_handle = k4a_transformation_create(&calibration);
        if (transformation_handle == NULL) {
            return Py_BuildValue("I", K4A_RESULT_FAILED);
        }
        result = k4a_device_start_cameras(device, &config);
        result = k4abt_tracker_create(&calibration, tracker_calibration, &tracker);
        if (result == K4A_RESULT_FAILED) {
            return Py_BuildValue("I", K4A_RESULT_FAILED);
        }
        return Py_BuildValue("I", result);
    }

    static PyObject* device_start_cameras_with_lut(PyObject* self, PyObject* args){
        printf("k4abt version: %s\n", K4ABT_VERSION_STR);
        printf("k4a version: %s\n", K4A_VERSION_STR);

        k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
        int gpu_id;
        PyArg_ParseTuple(args, "IIIIIIIIII", &config.color_format,
                &config.color_resolution, &config.depth_mode,
                &config.camera_fps, &config.synchronized_images_only,
                &config.depth_delay_off_color_usec, &config.wired_sync_mode,
                &config.subordinate_delay_off_master_usec,
                &config.disable_streaming_indicator, &gpu_id);

        k4a_result_t result;
        k4abt_tracker_configuration_t tracker_calibration = K4ABT_TRACKER_CONFIG_DEFAULT;
        tracker_calibration.gpu_device_id = gpu_id;
        printf("start\n");
        result = k4a_device_get_calibration(device, config.depth_mode, config.color_resolution, &calibration);
        if (result == K4A_RESULT_FAILED) {
            return Py_BuildValue("I", K4A_RESULT_FAILED);
        }
        printf("k4a_device_get_calibration done\n");
        transformation_handle = k4a_transformation_create(&calibration);
        if (transformation_handle == NULL) {
            return Py_BuildValue("I", K4A_RESULT_FAILED);
        }
        printf("k4a_transformation_create done\n");

        w_color = calibration.color_camera_calibration.resolution_width;
        h_color = calibration.color_camera_calibration.resolution_height;

        w_depth = calibration.depth_camera_calibration.resolution_width;
        h_depth = calibration.depth_camera_calibration.resolution_height;
        printf("%d %d %d %d\n", w_color, h_color, w_depth, h_depth);

        pinhole_t depth_pinhole = create_pinhole_from_xy_range(&calibration, K4A_CALIBRATION_TYPE_DEPTH);
        printf("create_pinhole_from_xy_range done\n");
        interpolation_t interpolation_type = INTERPOLATION_BILINEAR_DEPTH;

        k4a_image_create(
            K4A_IMAGE_FORMAT_CUSTOM,
            depth_pinhole.width,
            depth_pinhole.height,
            depth_pinhole.width * (int)sizeof(coordinate_t),
            &depth_lut
        );
        create_undistortion_lut(&calibration, K4A_CALIBRATION_TYPE_DEPTH, &depth_pinhole, depth_lut, interpolation_type);
        printf("create_undistortion_lut done\n");

        result = k4a_device_start_cameras(device, &config);
        if (result == K4A_RESULT_FAILED) {
            return Py_BuildValue("I", K4A_RESULT_FAILED);
        }
        printf("k4a_device_start_cameras done\n");
        result = k4abt_tracker_create(&calibration, tracker_calibration, &tracker);
        printf("k4abt_tracker_create\n");
        if (result == K4A_RESULT_FAILED) {
            return Py_BuildValue("I", K4A_RESULT_FAILED);
        }
        printf("k4abt_tracker_create done\n");

        // rgb intrinsics
        k4a_calibration_intrinsic_parameters_t *rgb_intrinsics = &calibration.color_camera_calibration.intrinsics.parameters;
        _camera_matrix_rgb = {
            rgb_intrinsics->param.fx, 0.f, rgb_intrinsics->param.cx, 
            0.f, rgb_intrinsics->param.fy, rgb_intrinsics->param.cy, 
            0.f, 0.f, 1.f
        };
        camera_matrix_rgb = Mat(3, 3, CV_32F, &_camera_matrix_rgb[0]);
        _dist_coeffs_rgb = { 
            rgb_intrinsics->param.k1, rgb_intrinsics->param.k2, 
            rgb_intrinsics->param.p1, rgb_intrinsics->param.p2, 
            rgb_intrinsics->param.k3, rgb_intrinsics->param.k4,
            rgb_intrinsics->param.k5, rgb_intrinsics->param.k6
        };
        dist_coeffs_rgb = Mat(8, 1, CV_32F, &_dist_coeffs_rgb[0]);
        new_camera_matrix_rgb = camera_matrix_rgb.clone();
        new_camera_matrix_rgb.at<float>(1,1) = new_camera_matrix_rgb.at<float>(0,0);

        initUndistortRectifyMap(
            camera_matrix_rgb, 
            dist_coeffs_rgb,
            Mat(), 
            new_camera_matrix_rgb, 
            Size(w_color, h_color),
            CV_32FC1, 
            color_undistort_map_1, color_undistort_map_2
        );
        // printf("rmap size: %d %d\n", color_undistort_map_1.rows, color_undistort_map_1.cols);

        // decoder = NvJPEGDecoder();
        decoder = JPEGDecoder();


//        cam_params = new json();
        cam_params["k4abt_sdk_version"] = K4ABT_VERSION_STR;

        Mat se3 = Mat(
            3, 3,
            CV_32FC1,
            calibration.extrinsics[K4A_CALIBRATION_TYPE_DEPTH][K4A_CALIBRATION_TYPE_COLOR].rotation
        );
        Mat r_vec = Mat(3, 1, CV_32FC1);
        Rodrigues(se3, r_vec);
        Mat t_vec = Mat(
            3, 1,
            CV_32F,
            calibration.extrinsics[K4A_CALIBRATION_TYPE_DEPTH][K4A_CALIBRATION_TYPE_COLOR].translation
        );

        k4a_calibration_intrinsic_parameters_t *depth_intrinsics = &calibration.depth_camera_calibration.intrinsics.parameters;
//        k4a_calibration_intrinsic_parameters_t *rgb_intrinsics = &calibration.color_camera_calibration.intrinsics.parameters;

        output_intrinsics(
            w_color, h_color,
            w_depth, h_depth,
            depth_intrinsics, rgb_intrinsics,
            t_vec, r_vec,
            cam_params
        );

        output_pinhole("depth_undistorted_intrinsics", depth_pinhole, cam_params);
        output_cam_mat("rgb_undistorted_intrinsics", new_camera_matrix_rgb, cam_params);

//        k4a_buffer_result_t buffer_result;
//        size_t serial_number_size;
//        buffer_result = k4a_device_get_serialnum(device, NULL, &serial_number_size);
//        char* serial_number = (char*)malloc(sizeof(char) * serial_number_size);
//        buffer_result = k4a_device_get_serialnum(device, serial_number, &serial_number_size);
//        std::cout << serial_number << std::endl;

        return Py_BuildValue("I", result);
    }

    static PyObject* device_get_cam_params(PyObject* self, PyObject* args){
        std::string s = cam_params.dump();
        return Py_BuildValue("s", s.c_str());
    }

    static PyObject* device_get_serial_number(PyObject* self, PyObject* args){
        k4a_buffer_result_t buffer_result;
        size_t serial_number_size = 0;
        buffer_result = k4a_device_get_serialnum(device, NULL, &serial_number_size);
        char* serial_number = (char*)malloc(sizeof(char) * serial_number_size);
        buffer_result = k4a_device_get_serialnum(device, serial_number, &serial_number_size);
        if (buffer_result != K4A_BUFFER_RESULT_SUCCEEDED) {
            return Py_BuildValue("s", NULL);
        } else {
            return Py_BuildValue("s", serial_number);
        }
    }

    static PyObject* device_stop_cameras(PyObject* self, PyObject* args){
        if (transformation_handle) k4a_transformation_destroy(transformation_handle);
        if (capture) k4a_capture_release(capture);
        if (tracker) k4abt_tracker_destroy(tracker);
        k4a_device_stop_cameras(device);
        return Py_BuildValue("I", K4A_RESULT_SUCCEEDED);
    }

    static PyObject* device_get_capture(PyObject* self, PyObject* args){
        int32_t timeout;
        PyArg_ParseTuple(args, "I", &timeout);
        if (capture) k4a_capture_release(capture);
        k4a_capture_create(&capture);
        k4a_wait_result_t result = k4a_device_get_capture(device, &capture, timeout);
        return Py_BuildValue("I", result);
    }

    static void k4a_image_t_capsule_cleanup(PyObject *capsule) {
        k4a_image_t *image = (k4a_image_t*)PyCapsule_GetContext(capsule);
        k4a_image_release(*image);
        // free(image);
    }

    static void double_t_buffer_capsule_cleanup(PyObject *capsule) {
        double_t *buffer = (double_t*)PyCapsule_GetContext(capsule);
        delete buffer;
    }

    static void uint8_t_buffer_capsule_cleanup(PyObject *capsule) {
        uint8_t *buffer = (uint8_t*)PyCapsule_GetContext(capsule);
        delete buffer;
    }

    static void uint16_t_buffer_capsule_cleanup(PyObject *capsule) {
        uint16_t *buffer = (uint16_t*)PyCapsule_GetContext(capsule);
        delete buffer;
    }

    static PyObject* device_get_pose_data(PyObject* self, PyObject* args){
        k4abt_tracker_enqueue_capture(tracker, capture, 0);
        k4abt_frame_t body_frame = NULL;
        k4a_wait_result_t pop_frame_result = k4abt_tracker_pop_result(tracker, &body_frame, 0);

        if (pop_frame_result == K4A_WAIT_RESULT_SUCCEEDED)
        {
            size_t num_bodies = k4abt_frame_get_num_bodies(body_frame);
            double_t* buffer = new double_t[num_bodies*NUM_JOINTS*NUM_DATA];

            for (size_t i = 0; i < num_bodies; i++)
            {
                k4abt_skeleton_t skeleton;
                k4abt_frame_get_body_skeleton(body_frame, i, &skeleton);
                for (int j = 0; j < (int)K4ABT_JOINT_COUNT; j++)
                {
                    k4a_float3_t position = skeleton.joints[j].position;
                    k4a_float2_t position_image;
                    int valid;
                    //Convert 3d points in mm to image coordinates
                    k4a_calibration_3d_to_2d(&calibration,
                                             &position,
                                             K4A_CALIBRATION_TYPE_DEPTH,
                                             K4A_CALIBRATION_TYPE_COLOR,
                                             &position_image, &valid);

                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 0] = position_image.v[0];
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 1] = position_image.v[1];

                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 2] = position.v[0];
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 3] = position.v[1];
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 4] = position.v[2];

                    k4a_quaternion_t orientation = skeleton.joints[j].orientation;
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 5] = orientation.v[0];
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 6] = orientation.v[1];
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 7] = orientation.v[2];
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 8] = orientation.v[3];

                    k4abt_joint_confidence_level_t confidence_level = skeleton.joints[j].confidence_level;
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 9] = confidence_level;
                }
            }

            // body index mask
            k4a_image_t* body_index_map = (k4a_image_t*) malloc(sizeof(k4a_image_t));
            *body_index_map = k4abt_frame_get_body_index_map(body_frame);

            k4abt_frame_release(body_frame);

            // pose
            npy_intp dims[3];
            dims[0] = num_bodies;
            dims[1] = NUM_JOINTS;
            dims[2] = NUM_DATA;

            PyArrayObject* np_pose_data = (PyArrayObject*) PyArray_SimpleNewFromData(3, dims, NPY_DOUBLE, buffer);
            
            PyObject *capsule = PyCapsule_New(buffer, NULL, double_t_buffer_capsule_cleanup);
            PyCapsule_SetContext(capsule, buffer);
            PyArray_SetBaseObject((PyArrayObject *) np_pose_data, capsule);

            // body index mask
            uint8_t* buffer_body_index_mask = k4a_image_get_buffer(*body_index_map);
            npy_intp dims_body_index_mask[2];
            dims_body_index_mask[0] = k4a_image_get_height_pixels(*body_index_map);
            dims_body_index_mask[1] = k4a_image_get_width_pixels(*body_index_map);
            PyArrayObject* np_body_index_mask = (PyArrayObject*) PyArray_SimpleNewFromData(2, dims_body_index_mask, NPY_UINT8, buffer_body_index_mask);

            PyObject *capsule_body_index_mask = PyCapsule_New(buffer_body_index_mask, NULL, k4a_image_t_capsule_cleanup);
            PyCapsule_SetContext(capsule_body_index_mask, body_index_map);
            PyArray_SetBaseObject((PyArrayObject*) np_body_index_mask, capsule_body_index_mask);

            return Py_BuildValue("NN", PyArray_Return(np_pose_data), PyArray_Return(np_body_index_mask));
        }
        else {
            return Py_BuildValue("ss", NULL, NULL);
        }
    }

    static bool check_depth_image_exists() {
        k4a_image_t depth = k4a_capture_get_depth_image(capture);
        if (depth != nullptr)
        {
            k4a_image_release(depth);
            return true;
        }
        else
        {
            return false;
        }
    }


    static void device_get_pose_data_wrapper() {
        if (!check_depth_image_exists()) {
            printf("!check_depth_image_exists\n");
            device_get_pose_data_result_0 = false;
            return;
        }

        k4a_wait_result_t enqueue_frame_result = k4abt_tracker_enqueue_capture(tracker, capture, K4A_WAIT_INFINITE);
        if (enqueue_frame_result != K4A_WAIT_RESULT_SUCCEEDED) {
            printf("k4abt_tracker_enqueue_capture error\n");
            device_get_pose_data_result_0 = false;
            return;
        }

        k4abt_frame_t body_frame = NULL;
        k4a_wait_result_t pop_frame_result = k4abt_tracker_pop_result(tracker, &body_frame, K4A_WAIT_INFINITE);

        if (pop_frame_result == K4A_WAIT_RESULT_SUCCEEDED)
        {
            size_t num_bodies = k4abt_frame_get_num_bodies(body_frame);
            double_t* buffer = new double_t[num_bodies*NUM_JOINTS*NUM_DATA];

            for (size_t i = 0; i < num_bodies; i++)
            {
                k4abt_skeleton_t skeleton;
                k4abt_frame_get_body_skeleton(body_frame, i, &skeleton);
                for (int j = 0; j < (int)K4ABT_JOINT_COUNT; j++)
                {
                    k4a_float3_t position = skeleton.joints[j].position;
                    k4a_float2_t position_image;
                    int valid;
                    //Convert 3d points in mm to image coordinates
                    k4a_calibration_3d_to_2d(&calibration,
                                             &position,
                                             K4A_CALIBRATION_TYPE_DEPTH,
                                             K4A_CALIBRATION_TYPE_COLOR,
                                             &position_image, &valid);

                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 0] = position_image.v[0];
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 1] = position_image.v[1];

                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 2] = position.v[0];
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 3] = position.v[1];
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 4] = position.v[2];

                    k4a_quaternion_t orientation = skeleton.joints[j].orientation;
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 5] = orientation.v[0];
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 6] = orientation.v[1];
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 7] = orientation.v[2];
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 8] = orientation.v[3];

                    k4abt_joint_confidence_level_t confidence_level = skeleton.joints[j].confidence_level;
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 9] = confidence_level;
                }
            }

            k4a_image_t body_index_map = k4abt_frame_get_body_index_map(body_frame);
            uint8_t* buffer_body_index_map = (uint8_t*)malloc(w_depth * h_depth * sizeof(uint8_t));
            memcpy(buffer_body_index_map, k4a_image_get_buffer(body_index_map), w_depth * h_depth * sizeof(uint8_t));
            k4a_image_release(body_index_map);

            k4abt_frame_release(body_frame);

            device_get_pose_data_result_0 = true;
            device_get_pose_data_result_1 = num_bodies;
            device_get_pose_data_result_2 = buffer;
            device_get_pose_data_result_3 = buffer_body_index_map;
        }
        else {
            printf("k4abt_tracker_pop_result error\n");
            device_get_pose_data_result_0 = false;
        }
    }

    static PyObject* device_get_pose_data_start(PyObject* self, PyObject* args) {
        device_get_pose_data_future = std::async(
            std::launch::async | std::launch::deferred,
            [](){device_get_pose_data_wrapper();}
        );
        return Py_BuildValue("");
    }

    static PyObject* device_get_pose_data_join(PyObject* self, PyObject* args) {
        device_get_pose_data_future.get();
        if (device_get_pose_data_result_0) {
            int num_bodies = device_get_pose_data_result_1;
            double_t* buffer = new double_t[num_bodies * NUM_JOINTS * NUM_DATA];
            memcpy(buffer, device_get_pose_data_result_2, num_bodies * NUM_JOINTS * NUM_DATA * sizeof(double_t));
            delete device_get_pose_data_result_2;

            npy_intp dims[3];
            dims[0] = num_bodies;
            dims[1] = NUM_JOINTS;
            dims[2] = NUM_DATA;

            PyArrayObject* np_pose_data = (PyArrayObject*) PyArray_SimpleNewFromData(3, dims, NPY_DOUBLE, buffer);

            PyObject *capsule = PyCapsule_New(buffer, NULL, double_t_buffer_capsule_cleanup);
            PyCapsule_SetContext(capsule, buffer);
            PyArray_SetBaseObject((PyArrayObject *) np_pose_data, capsule);

            uint8_t* buffer_body_index_map = (uint8_t*)malloc(w_depth * h_depth * sizeof(uint8_t));
            memcpy(buffer_body_index_map, device_get_pose_data_result_3, w_depth * h_depth * sizeof(uint8_t));
            delete device_get_pose_data_result_3;

            npy_intp dims_body_index[2];
            dims_body_index[0] = h_depth;
            dims_body_index[1] = w_depth;
            PyArrayObject* np_body_index_mask = (PyArrayObject*) PyArray_SimpleNewFromData(2, dims_body_index, NPY_UINT8, buffer_body_index_map);

            PyObject *capsule_body_index_mask = PyCapsule_New(buffer_body_index_map, NULL, uint8_t_buffer_capsule_cleanup);
            PyCapsule_SetContext(capsule_body_index_mask, buffer_body_index_map);
            PyArray_SetBaseObject((PyArrayObject*) np_body_index_mask, capsule_body_index_mask);


            return Py_BuildValue(
                "NN",
                PyArray_Return(np_pose_data),
                PyArray_Return(np_body_index_mask)
            );
        } else {
            return Py_BuildValue("ss", NULL, NULL);
        }
    }

    static void device_get_pose_data_undistorted_wrapper() {
        if (!check_depth_image_exists()) {
            printf("!check_depth_image_exists\n");
            device_get_pose_data_undistorted_result_0 = false;
            return;
        }

        k4a_wait_result_t enqueue_frame_result = k4abt_tracker_enqueue_capture(tracker, capture, K4A_WAIT_INFINITE);
        if (enqueue_frame_result != K4A_WAIT_RESULT_SUCCEEDED) {
            printf("k4abt_tracker_enqueue_capture error\n");
            device_get_pose_data_undistorted_result_0 = false;
            return;
        }

        k4abt_frame_t body_frame = NULL;
        k4a_wait_result_t pop_frame_result = k4abt_tracker_pop_result(tracker, &body_frame, K4A_WAIT_INFINITE);

        if (pop_frame_result == K4A_WAIT_RESULT_SUCCEEDED)
        {
            size_t num_bodies = k4abt_frame_get_num_bodies(body_frame);
            double_t* buffer = new double_t[num_bodies*NUM_JOINTS*NUM_DATA];

            for (size_t i = 0; i < num_bodies; i++)
            {
                k4abt_skeleton_t skeleton;
                k4abt_frame_get_body_skeleton(body_frame, i, &skeleton);
                for (int j = 0; j < (int)K4ABT_JOINT_COUNT; j++)
                {
                    k4a_float3_t position = skeleton.joints[j].position;
                    k4a_float2_t position_image;
                    int valid;
                    //Convert 3d points in mm to image coordinates
                    k4a_calibration_3d_to_2d(&calibration,
                                             &position,
                                             K4A_CALIBRATION_TYPE_DEPTH,
                                             K4A_CALIBRATION_TYPE_COLOR,
                                             &position_image, &valid);

                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 0] = position_image.v[0];
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 1] = position_image.v[1];

                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 2] = position.v[0];
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 3] = position.v[1];
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 4] = position.v[2];

                    k4a_quaternion_t orientation = skeleton.joints[j].orientation;
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 5] = orientation.v[0];
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 6] = orientation.v[1];
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 7] = orientation.v[2];
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 8] = orientation.v[3];

                    k4abt_joint_confidence_level_t confidence_level = skeleton.joints[j].confidence_level;
                    buffer[(i * NUM_JOINTS * NUM_DATA) + (j * NUM_DATA) + 9] = confidence_level;
                }
            }

            // body_index mask
            k4a_image_t body_index_map = k4abt_frame_get_body_index_map(body_frame);
            k4a_image_t body_index_map_undistorted = NULL;
            k4a_image_create(
                K4A_IMAGE_FORMAT_CUSTOM8,
                w_depth, h_depth,
                w_depth * (int)sizeof(uint8_t),
                &body_index_map_undistorted
            );
            k4a_remap<uint8_t>(body_index_map, depth_lut, body_index_map_undistorted, INTERPOLATION_NEARESTNEIGHBOR);
            k4a_image_release(body_index_map);

            uint8_t* buffer_body_index_map_undistorted = (uint8_t*)malloc(w_depth * h_depth * sizeof(uint8_t));
            memcpy(buffer_body_index_map_undistorted, k4a_image_get_buffer(body_index_map_undistorted), w_depth * h_depth * sizeof(uint8_t));
            k4a_image_release(body_index_map_undistorted);

            k4abt_frame_release(body_frame);

            device_get_pose_data_undistorted_result_0 = true;
            device_get_pose_data_undistorted_result_1 = num_bodies;
            device_get_pose_data_undistorted_result_2 = buffer;
            device_get_pose_data_undistorted_result_3 = buffer_body_index_map_undistorted;
        }
        else {
            printf("k4abt_tracker_pop_result error\n");
            device_get_pose_data_undistorted_result_0 = false;
        }
    }

    static PyObject* device_get_pose_data_undistorted_start(PyObject* self, PyObject* args) {
        device_get_pose_data_undistorted_future = std::async(
            std::launch::async | std::launch::deferred, 
            [](){device_get_pose_data_undistorted_wrapper();}
        );
        return Py_BuildValue("");
    }

    static PyObject* device_get_pose_data_undistorted_join(PyObject* self, PyObject* args) {
        device_get_pose_data_undistorted_future.get();
        if (device_get_pose_data_undistorted_result_0) {
            int num_bodies = device_get_pose_data_undistorted_result_1;
            double_t* buffer = new double_t[num_bodies * NUM_JOINTS * NUM_DATA];
            memcpy(buffer, device_get_pose_data_undistorted_result_2, num_bodies * NUM_JOINTS * NUM_DATA * sizeof(double_t));
            delete device_get_pose_data_undistorted_result_2;

            npy_intp dims[3];
            dims[0] = num_bodies;
            dims[1] = NUM_JOINTS;
            dims[2] = NUM_DATA;

            PyArrayObject* np_pose_data = (PyArrayObject*) PyArray_SimpleNewFromData(3, dims, NPY_DOUBLE, buffer);
            
            PyObject *capsule = PyCapsule_New(buffer, NULL, double_t_buffer_capsule_cleanup);
            PyCapsule_SetContext(capsule, buffer);
            PyArray_SetBaseObject((PyArrayObject *) np_pose_data, capsule);

            uint8_t* buffer_body_index_map_undistorted = (uint8_t*)malloc(w_depth * h_depth * sizeof(uint8_t));
            memcpy(buffer_body_index_map_undistorted, device_get_pose_data_undistorted_result_3, w_depth * h_depth * sizeof(uint8_t));
            delete device_get_pose_data_undistorted_result_3;
            
            npy_intp dims_body_index[2];
            dims_body_index[0] = h_depth;
            dims_body_index[1] = w_depth;
            PyArrayObject* np_body_index_mask_undistorted = (PyArrayObject*) PyArray_SimpleNewFromData(2, dims_body_index, NPY_UINT8, buffer_body_index_map_undistorted);
            
            PyObject *capsule_body_index_mask_undistorted = PyCapsule_New(buffer_body_index_map_undistorted, NULL, uint8_t_buffer_capsule_cleanup);
            PyCapsule_SetContext(capsule_body_index_mask_undistorted, buffer_body_index_map_undistorted);
            PyArray_SetBaseObject((PyArrayObject*) np_body_index_mask_undistorted, capsule_body_index_mask_undistorted);


            return Py_BuildValue(
                "NN", 
                PyArray_Return(np_pose_data), 
                PyArray_Return(np_body_index_mask_undistorted)
            );
        } else {
            return Py_BuildValue("ss", NULL, NULL);
        }
    }

    static PyObject* device_get_color_image(PyObject* self, PyObject* args){
        k4a_image_t* color_image = (k4a_image_t*) malloc(sizeof(k4a_image_t));
        *color_image = k4a_capture_get_color_image(capture);
        if (color_image) {
            uint8_t* buffer = k4a_image_get_buffer(*color_image);
            npy_intp dims[3];
            dims[0] = k4a_image_get_height_pixels(*color_image);
            dims[1] = k4a_image_get_width_pixels(*color_image);
            dims[2] = 4;

            PyArrayObject* np_color_image = (PyArrayObject*) PyArray_SimpleNewFromData(3, dims, NPY_UINT8, buffer);
            PyObject *capsule = PyCapsule_New(buffer, NULL, k4a_image_t_capsule_cleanup);
            PyCapsule_SetContext(capsule, color_image);
            PyArray_SetBaseObject((PyArrayObject *) np_color_image, capsule);
            return PyArray_Return(np_color_image);
        }
        else {
            free(color_image);
            return Py_BuildValue("");
        }
    }

    static void device_get_color_image_wrapper(){
        k4a_image_t color_image = k4a_capture_get_color_image(capture);

        counter += 1;

        if (color_image != NULL && counter >= 10) {
            k4a_image_format_t format;
            format = k4a_image_get_format(color_image);
            if (format != K4A_IMAGE_FORMAT_COLOR_MJPG) {
                printf("%d color format not supported. Please use MJPEG\n", format);
                k4a_image_release(color_image);
                device_get_color_image_result_0 = false;
                return;
            }

            uint8_t* buffer = (uint8_t*)malloc(w_color * h_color * COLOR_CHANNELS * sizeof(uint8_t));
            if (!decoder.decode(color_image, buffer, w_color, h_color)) {
                device_get_color_image_result_0 = false;
                return;
            }

            color_image_device_timestamp_usec = k4a_image_get_device_timestamp_usec(color_image);
            k4a_image_release(color_image);

            device_get_color_image_result_0 = true;
            device_get_color_image_result_1 = buffer;
        } else {
            k4a_image_release(color_image);
            device_get_color_image_result_0 = false;
        }
    }

    static PyObject* device_get_color_image_start(PyObject* self, PyObject* args){
        device_get_color_image_future = std::async(
            std::launch::async | std::launch::deferred,
            [](){device_get_color_image_wrapper();}
        );
        return Py_BuildValue("");
    }

    static PyObject* device_get_color_image_join(PyObject* self, PyObject* args){
        device_get_color_image_future.get();
        if (device_get_color_image_result_0) {
            uint8_t* buffer = (uint8_t*)malloc(w_color * h_color * COLOR_CHANNELS * sizeof(uint8_t));
            memcpy(buffer, device_get_color_image_result_1, w_color * h_color * 4 * sizeof(uint8_t));
            delete device_get_color_image_result_1;

            npy_intp dims[3];
            dims[0] = h_color;
            dims[1] = w_color;
            dims[2] = COLOR_CHANNELS;

            PyArrayObject* np_color_image = (PyArrayObject*) PyArray_SimpleNewFromData(3, dims, NPY_UINT8, buffer);

            PyObject *capsule = PyCapsule_New(buffer, NULL, uint8_t_buffer_capsule_cleanup);
            PyCapsule_SetContext(capsule, buffer);
            PyArray_SetBaseObject((PyArrayObject*)np_color_image, capsule);

            return PyArray_Return(np_color_image);
        } else {
            return Py_BuildValue("");
        }
    }

    static void device_get_color_image_undistorted_wrapper(){
        // auto t0 = high_resolution_clock::now(); 
        k4a_image_t color_image = k4a_capture_get_color_image(capture);
        // auto t1 = high_resolution_clock::now(); 
        // auto time_wait_frame = duration_cast<microseconds>(t1 - t0).count() / 1e3;
        // printf("%.3fms wait frame\n", time_wait_frame);

        counter += 1;

        if (color_image != NULL && counter >= 10) {
            k4a_image_format_t format;
            format = k4a_image_get_format(color_image);
            if (format != K4A_IMAGE_FORMAT_COLOR_MJPG) {
                printf("%d color format not supported. Please use MJPEG\n", format);
                k4a_image_release(color_image);
                device_get_color_image_undistorted_result_0 = false;
                return;    
            }
            
            // if (counter == 50) {
            //     uint8_t* buffer = k4a_image_get_buffer(color_image);
            //     unsigned long bsize = static_cast<unsigned long>(k4a_image_get_size(color_image));

            //     auto myfile = std::fstream("./tmp.bin", std::ios::out | std::ios::binary);
            //     myfile.write((char*)&buffer[0], bsize);
            //     myfile.close();

            //     // ofstream myFile ("./tmp.bin", ios::out | ios::binary);
            //     // myFile.write (buffer, 100);

            //     // fstream file;
            //     // file.open("./tmp.bin", std::ios_base::binary);
            //     // assert(file.is_open());
            //     // for (int i = 0; i < bsize / sizeof(uint8_t); ++i) {
            //     //     file.write((uint8_t)(buffer[i]), sizeof(uint8_t));
            //     // }
            //     // file.close();
            // }

            // if (counter == 50) {
            //     uint8_t* buffer_new = reinterpret_cast<uint8_t*>(reinterpret_cast<int*>(buffer));
            //     ofstream myfile;
            //     myfile.open("example.txt");
            //     // for (int i = 0; i < w_color * h_color * COLOR_CHANNELS; i++) {
            //     for (int i = 0; i < 100; i++) {
            //         // printf("%d ", buffer[i]);
            //         myfile << buffer_new[i] << " ";
            //     }
            //     myfile.close();
            //     printf("###############################################\n");
            // }

            
            // auto t1 = high_resolution_clock::now(); 
            // printf("color size: %d\n", static_cast<unsigned long>(k4a_image_get_size(color_image)));
            uint8_t* buffer = (uint8_t*)malloc(w_color * h_color * COLOR_CHANNELS * sizeof(uint8_t));
            if (!decoder.decode(color_image, buffer, w_color, h_color)) {
                // k4a_image_release(color_image);
                // delete buffer;
                device_get_color_image_undistorted_result_0 = false;
                return;    
            } else {
                // uint8_t* buffer = (uint8_t*)malloc(w_color * h_color * COLOR_CHANNELS * sizeof(uint8_t));
                // printf("decode successfull\n");
            }
            // auto t2 = high_resolution_clock::now(); 
            // auto time_decode = duration_cast<microseconds>(t2 - t1).count() / 1e3;
            // printf("%.3fms decode\n", time_decode);
            

            // if (!decoder.decode(color_image)) {
            //     k4a_image_release(color_image);
            //     return Py_BuildValue("");    
            // } else {
            //     // uint8_t* buffer = (uint8_t*)malloc(w_color * h_color * COLOR_CHANNELS * sizeof(uint8_t));
            //     printf("decoded size: %d\n", decoder.decoder_output.data_size);
            //     // printf("decode successfull");
            // }
            // if (counter == 50) {
            //     uint8_t* buffer_new = reinterpret_cast<uint8_t*>(reinterpret_cast<int*>(buffer));
            //     ofstream myfile;
            //     myfile.open("example.txt");
            //     // for (int i = 0; i < w_color * h_color * COLOR_CHANNELS; i++) {
            //     for (int i = 0; i < 100; i++) {
            //         // printf("%d ", buffer[i]);
            //         myfile << buffer_new[i] << " ";
            //     }
            //     myfile.close();
            //     printf("###############################################\n");
            // }
            
            
            Mat color_mat(h_color, w_color, CV_8UC4, (void*)buffer, Mat::AUTO_STEP);
            Mat color_mat_undistorted;
            remap(color_mat, color_mat_undistorted, color_undistort_map_1, color_undistort_map_2, INTER_LINEAR);
            
            // auto t3 = high_resolution_clock::now(); 
            // auto time_undistort = duration_cast<microseconds>(t3 - t2).count() / 1e3;
            // printf("%.3fms undistort\n", time_undistort);

            uint8_t* buffer_undistorted = (uint8_t*)malloc(w_color * h_color * COLOR_CHANNELS * sizeof(uint8_t));
            memcpy(buffer_undistorted, color_mat_undistorted.data, w_color * h_color * COLOR_CHANNELS * sizeof(uint8_t));

            delete buffer;
            color_mat.release();
            color_mat_undistorted.release();
            color_image_device_timestamp_usec = k4a_image_get_device_timestamp_usec(color_image);
            k4a_image_release(color_image);

            device_get_color_image_undistorted_result_0 = true;
            device_get_color_image_undistorted_result_1 = buffer_undistorted;
        } else {
            k4a_image_release(color_image);
            device_get_color_image_undistorted_result_0 = false;
        }
    }

    static PyObject* device_get_color_image_undistorted_start(PyObject* self, PyObject* args){
        device_get_color_image_undistorted_future = std::async(
            std::launch::async | std::launch::deferred, 
            [](){device_get_color_image_undistorted_wrapper();}
        );
        return Py_BuildValue("");
    }

    static PyObject* device_get_color_image_undistorted_join(PyObject* self, PyObject* args){
        device_get_color_image_undistorted_future.get();
        if (device_get_color_image_undistorted_result_0) {
            uint8_t* buffer_undistorted = (uint8_t*)malloc(w_color * h_color * COLOR_CHANNELS * sizeof(uint8_t));
            memcpy(buffer_undistorted, device_get_color_image_undistorted_result_1, w_color * h_color * 4 * sizeof(uint8_t));
            delete device_get_color_image_undistorted_result_1;

            npy_intp dims[3];
            dims[0] = h_color;
            dims[1] = w_color;
            dims[2] = COLOR_CHANNELS;

            PyArrayObject* np_color_image_undistorted = (PyArrayObject*) PyArray_SimpleNewFromData(3, dims, NPY_UINT8, buffer_undistorted);

            PyObject *capsule = PyCapsule_New(buffer_undistorted, NULL, uint8_t_buffer_capsule_cleanup);
            PyCapsule_SetContext(capsule, buffer_undistorted);
            PyArray_SetBaseObject((PyArrayObject*)np_color_image_undistorted, capsule);

            return PyArray_Return(np_color_image_undistorted);
        } else {
            return Py_BuildValue("");
        }
    }

    static PyObject* device_get_depth_image(PyObject* self, PyObject* args){
        int is_transform_enabled;
        PyArg_ParseTuple(args, "p", &is_transform_enabled);

        k4a_image_t* depth_image = (k4a_image_t*) malloc(sizeof(k4a_image_t));
        *depth_image = k4a_capture_get_depth_image(capture);
        if (is_transform_enabled && *depth_image) {
            k4a_image_t color_image = k4a_capture_get_color_image(capture);
            if (color_image) {
                k4a_image_t depth_image_transformed;
                k4a_image_create(
                    k4a_image_get_format(*depth_image),
                    k4a_image_get_width_pixels(color_image),
                    k4a_image_get_height_pixels(color_image),
                    k4a_image_get_width_pixels(color_image) * (int)sizeof(uint16_t),
                    &depth_image_transformed);
                k4a_result_t res = k4a_transformation_depth_image_to_color_camera(
                    transformation_handle,
                    *depth_image, depth_image_transformed);
                if (res == K4A_RESULT_FAILED){
                    free(depth_image);
                    return Py_BuildValue("");
                }

                k4a_image_release(color_image);
                k4a_image_release(*depth_image);
                *depth_image = depth_image_transformed;
            }
        }

        if (*depth_image) {
            uint8_t* buffer = k4a_image_get_buffer(*depth_image);
            npy_intp dims[2];
            dims[0] = k4a_image_get_height_pixels(*depth_image);
            dims[1] = k4a_image_get_width_pixels(*depth_image);
            PyArrayObject* np_depth_image = (PyArrayObject*) PyArray_SimpleNewFromData(2, dims, NPY_UINT16, buffer);
            PyObject *capsule = PyCapsule_New(buffer, NULL, uint8_t_buffer_capsule_cleanup);
            PyCapsule_SetContext(capsule, depth_image);
            PyArray_SetBaseObject((PyArrayObject *) np_depth_image, capsule);
            return PyArray_Return(np_depth_image);
        }
        else {
            free(depth_image);
            return Py_BuildValue("");
        }
    }

    static void device_get_depth_image_undistorted_wrapper(){
        bool error = false;

        k4a_image_t depth_image = NULL;
        depth_image = k4a_capture_get_depth_image(capture);
        k4a_image_t depth_image_undistorted = NULL;
        if (depth_image != NULL) {
            k4a_image_create(
                K4A_IMAGE_FORMAT_DEPTH16,
                w_depth,
                h_depth,
                w_depth * (int)sizeof(uint16_t),
                &depth_image_undistorted
            );
            k4a_remap<uint16_t>(depth_image, depth_lut, depth_image_undistorted, INTERPOLATION_BILINEAR_DEPTH);
            depth_image_device_timestamp_usec = k4a_image_get_device_timestamp_usec(depth_image);
        } else {
            error = true;
        }
        k4a_image_release(depth_image);
        if (error) {
            if (depth_image_undistorted != NULL) {
                k4a_image_release(depth_image_undistorted);
            }
            device_get_depth_image_undistorted_result_0 = false;
            return;
        }

        uint8_t* buffer = k4a_image_get_buffer(depth_image_undistorted);
        uint16_t* depth_buffer = (uint16_t*)malloc(w_depth * h_depth * sizeof(uint16_t));
        memcpy(depth_buffer, reinterpret_cast<uint16_t*>(buffer), w_depth * h_depth * sizeof(uint16_t));
        k4a_image_release(depth_image_undistorted);

        device_get_depth_image_undistorted_result_0 = true;
        device_get_depth_image_undistorted_result_1 = depth_buffer;
    }

    static PyObject* device_get_depth_image_undistorted_start(PyObject* self, PyObject* args){
        device_get_depth_image_undistorted_future = std::async(
            std::launch::async | std::launch::deferred,
            [](){device_get_depth_image_undistorted_wrapper();}
        );
        return Py_BuildValue("");
    }

    static PyObject* device_get_depth_image_undistorted_join(PyObject* self, PyObject* args){
        device_get_depth_image_undistorted_future.get();
        if (device_get_depth_image_undistorted_result_0) {
            uint16_t* depth_buffer_new = (uint16_t*)malloc(w_depth * h_depth * sizeof(uint16_t));
            memcpy(
                depth_buffer_new,
                device_get_depth_image_undistorted_result_1,
                w_depth * h_depth * sizeof(uint16_t)
            );
            delete device_get_depth_image_undistorted_result_1;

            npy_intp dims[2];
            dims[0] = h_depth;
            dims[1] = w_depth;
            PyArrayObject* np_depth_image = (PyArrayObject*) PyArray_SimpleNewFromData(2, dims, NPY_UINT16, depth_buffer_new);
            PyObject *capsule = PyCapsule_New(depth_buffer_new, NULL, uint16_t_buffer_capsule_cleanup);
            PyCapsule_SetContext(capsule, depth_buffer_new);
            PyArray_SetBaseObject((PyArrayObject *) np_depth_image, capsule);

            return PyArray_Return(np_depth_image);
        } else {
            return Py_BuildValue("");
        }
    }

    static void device_get_transformed_depth_image_wrapper(){
        bool error = false;

        k4a_image_t depth_image = NULL;
        k4a_image_t depth_image_transformed = NULL;
        depth_image = k4a_capture_get_depth_image(capture);
        if (depth_image != NULL) {
            k4a_image_create(
                k4a_image_get_format(depth_image),
                w_color, h_color,
                w_color * (int)sizeof(uint16_t),
                &depth_image_transformed
            );
            k4a_result_t res = k4a_transformation_depth_image_to_color_camera(
                transformation_handle,
                depth_image, depth_image_transformed
            );
            if (res == K4A_RESULT_FAILED) {
                error = true;
            }
        } else {
            error = true;
        }
        depth_image_device_timestamp_usec = k4a_image_get_device_timestamp_usec(depth_image);
        k4a_image_release(depth_image);
        if (error) {
            if (depth_image_transformed != NULL) {
                k4a_image_release(depth_image_transformed);
            }
            device_get_transformed_depth_image_result_0 = false;
            return;
        }

        uint8_t* buffer = k4a_image_get_buffer(depth_image_transformed);
        uint16_t* depth_buffer = (uint16_t*)malloc(w_color * h_color * sizeof(uint16_t));
        memcpy(depth_buffer, reinterpret_cast<uint16_t*>(buffer), w_color * h_color * sizeof(uint16_t));
        k4a_image_release(depth_image_transformed);

        device_get_transformed_depth_image_result_0 = true;
        device_get_transformed_depth_image_result_1 = depth_buffer;
    }

    static PyObject* device_get_transformed_depth_image_start(PyObject* self, PyObject* args){
        device_get_transformed_depth_image_future = std::async(
            std::launch::async | std::launch::deferred,
            [](){device_get_transformed_depth_image_wrapper();}
        );
        return Py_BuildValue("");
    }

    static PyObject* device_get_transformed_depth_image_join(PyObject* self, PyObject* args){
        device_get_transformed_depth_image_future.get();
        if (device_get_transformed_depth_image_result_0) {
            uint16_t* depth_buffer_new = (uint16_t*)malloc(w_color * h_color * sizeof(uint16_t));
            memcpy(depth_buffer_new, device_get_transformed_depth_image_result_1, w_color * h_color * sizeof(uint16_t));
            delete device_get_transformed_depth_image_result_1;

            npy_intp dims[2];
            dims[0] = h_color;
            dims[1] = w_color;
            PyArrayObject* np_depth_image = (PyArrayObject*) PyArray_SimpleNewFromData(2, dims, NPY_UINT16, depth_buffer_new);
            PyObject *capsule = PyCapsule_New(depth_buffer_new, NULL, uint16_t_buffer_capsule_cleanup);
            PyCapsule_SetContext(capsule, depth_buffer_new);
            PyArray_SetBaseObject((PyArrayObject *) np_depth_image, capsule);

            return PyArray_Return(np_depth_image);
        } else {
            return Py_BuildValue("");
        }
    }

    static void device_get_transformed_depth_image_undistorted_wrapper(){
        bool error = false;

        k4a_image_t depth_image = NULL;
        k4a_image_t depth_image_transformed = NULL;
        depth_image = k4a_capture_get_depth_image(capture);
        if (depth_image != NULL) {
            k4a_image_create(
                k4a_image_get_format(depth_image),
                w_color, h_color,
                w_color * (int)sizeof(uint16_t),
                &depth_image_transformed
            );
            k4a_result_t res = k4a_transformation_depth_image_to_color_camera(
                transformation_handle,
                depth_image, depth_image_transformed
            );
            if (res == K4A_RESULT_FAILED) {
                error = true;
            }
        } else {
            error = true;
        }
        depth_image_device_timestamp_usec = k4a_image_get_device_timestamp_usec(depth_image);
        k4a_image_release(depth_image);
        if (error) {
            if (depth_image_transformed != NULL) {
                k4a_image_release(depth_image_transformed);
            }
            device_get_transformed_depth_image_undistorted_result_0 = false;
            return;
        }

        uint8_t* buffer = k4a_image_get_buffer(depth_image_transformed);
        uint16_t* depth_buffer = reinterpret_cast<uint16_t*>(buffer);

        UMat umat, umat3;
        create_mat_from_buffer<uint16_t>(depth_buffer, w_color, h_color).copyTo(umat);
        remap(umat, umat3, color_undistort_map_1, color_undistort_map_2, INTER_LINEAR);

        uint16_t* depth_buffer_new = (uint16_t*)malloc(w_color * h_color * sizeof(uint16_t));
        auto ptr = umat3.getMat(ACCESS_READ).ptr<uint16_t>();
        memcpy(depth_buffer_new, ptr, w_color * h_color * sizeof(uint16_t));

        // Mat m = umat3.getMat(ACCESS_READ);
        // for (int i = 0; i < w_color * h_color; i++) {
        //     int i_h = i / w_color;
        //     int i_w = i % w_color;
        //     depth_buffer_new[i] = (uint16_t)m.at<uint16_t>(i_h, i_w);
        // }
        
        // counter += 1;
        // if (counter == 50) {
        //     auto ptr = mat;
        //     ofstream myfile;
        //     myfile.open("example.txt");
        //     for (int i = 0; i < w_color * h_color; i++) {
        //         // int i_h = i / w_color;
        //         // int i_w = i % w_color;
        //         // myfile << (uint16_t)mat.at<uint16_t>(i_h, i_w) << " ";
        //         // myfile << depth_buffer_new[i] << " ";
        //         myfile << ptr[i] << " ";
        //     }
        //     myfile.close();
        //     printf("done\n");
        // }

        // m.release();
        k4a_image_release(depth_image_transformed);
        umat.release();
        umat3.release();

        device_get_transformed_depth_image_undistorted_result_0 = true;
        device_get_transformed_depth_image_undistorted_result_1 = depth_buffer_new;
    }

    static PyObject* device_get_transformed_depth_image_undistorted_start(PyObject* self, PyObject* args){
        device_get_transformed_depth_image_undistorted_future = std::async(
            std::launch::async | std::launch::deferred, 
            [](){device_get_transformed_depth_image_undistorted_wrapper();}
        );
        return Py_BuildValue("");
    }

    static PyObject* device_get_transformed_depth_image_undistorted_join(PyObject* self, PyObject* args){
        device_get_transformed_depth_image_undistorted_future.get();
        if (device_get_transformed_depth_image_undistorted_result_0) {
            uint16_t* depth_buffer_new = (uint16_t*)malloc(w_color * h_color * sizeof(uint16_t));
            memcpy(depth_buffer_new, device_get_transformed_depth_image_undistorted_result_1, w_color * h_color * sizeof(uint16_t));
            delete device_get_transformed_depth_image_undistorted_result_1;

            npy_intp dims[2];
            dims[0] = h_color;
            dims[1] = w_color;
            PyArrayObject* np_depth_image = (PyArrayObject*) PyArray_SimpleNewFromData(2, dims, NPY_UINT16, depth_buffer_new);
            PyObject *capsule = PyCapsule_New(depth_buffer_new, NULL, uint16_t_buffer_capsule_cleanup);
            PyCapsule_SetContext(capsule, depth_buffer_new);
            PyArray_SetBaseObject((PyArrayObject *) np_depth_image, capsule);

            return PyArray_Return(np_depth_image);
        } else {
            return Py_BuildValue("");
        }
    }


    static PyObject* device_get_color_image_device_timestamp_usec(PyObject* self, PyObject* args) {
        return Py_BuildValue("I", color_image_device_timestamp_usec);
    }

    static PyObject* device_get_depth_image_device_timestamp_usec(PyObject* self, PyObject* args) {
        return Py_BuildValue("I", depth_image_device_timestamp_usec);
    }

    // Source : https://github.com/MathGaron/pyvicon/blob/master/pyvicon/pyvicon.cpp
    //###################
    //Module initialisation
    //###################

    struct module_state
    {
        PyObject *error;
    };

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))


    //#####################
    // Methods
    //#####################
    static PyMethodDef Pyk4aMethods[] = {
        {"device_open", device_open, METH_VARARGS, "Open an Azure Kinect device"},
        {"device_start_cameras", device_start_cameras, METH_VARARGS, "create tracker and calib"},
        {"device_start_cameras_with_lut", device_start_cameras_with_lut, METH_VARARGS, "create tracker and calib and with depth undistort lut"},
        {"device_stop_cameras", device_stop_cameras, METH_VARARGS, "Stops the color and depth camera capture"},
        {"device_get_capture", device_get_capture, METH_VARARGS, "Reads a sensor capture"},
        {"device_get_pose_data", device_get_pose_data, METH_VARARGS, "Get the body pose estimates associated with the given capture"},
        {"device_get_pose_data_start", device_get_pose_data_start, METH_VARARGS, "get pose_data start"},
        {"device_get_pose_data_join", device_get_pose_data_join, METH_VARARGS, "get pose_data join"},
        {"device_get_pose_data_undistorted_start", device_get_pose_data_undistorted_start, METH_VARARGS, "get pose_data_undistorted start"},
        {"device_get_pose_data_undistorted_join", device_get_pose_data_undistorted_join, METH_VARARGS, "get pose_data_undistorted join"},
        {"device_get_color_image", device_get_color_image, METH_VARARGS, "Get the color image associated with the given capture"},
        {"device_get_color_image_start", device_get_color_image_start, METH_VARARGS, "get color_image start"},
        {"device_get_color_image_join", device_get_color_image_join, METH_VARARGS, "get color_image join"},
        {"device_get_color_image_undistorted_start", device_get_color_image_undistorted_start, METH_VARARGS, "get color_image_undistorted start"},
        {"device_get_color_image_undistorted_join", device_get_color_image_undistorted_join, METH_VARARGS, "get color_image_undistorted join"},
        {"device_get_depth_image", device_get_depth_image, METH_VARARGS, "Set or add a depth image to the associated capture"},

        {"device_get_cam_params", device_get_cam_params, METH_VARARGS, "return intrinsincs and extrinsics for color and depth"},
        {"device_get_serial_number", device_get_serial_number, METH_VARARGS, "return serial number"},

        {"device_get_depth_image_undistorted_start", device_get_depth_image_undistorted_start, METH_VARARGS, "get depth_image_undistorted start"},
        {"device_get_depth_image_undistorted_join", device_get_depth_image_undistorted_join, METH_VARARGS, "get depth_image_undistorted join"},
        {"device_get_transformed_depth_image_start", device_get_transformed_depth_image_start, METH_VARARGS, "get transformed_depth_image start"},
        {"device_get_transformed_depth_image_join", device_get_transformed_depth_image_join, METH_VARARGS, "get transformed_depth_image join"},
        {"device_get_transformed_depth_image_undistorted_start", device_get_transformed_depth_image_undistorted_start, METH_VARARGS, "get transformed_depth_image_undistorted start"},
        {"device_get_transformed_depth_image_undistorted_join", device_get_transformed_depth_image_undistorted_join, METH_VARARGS, "get transformed_depth_image_undistorted join"},
        {"device_close", device_close, METH_VARARGS, "Close an Azure Kinect device"},
        {"device_get_sync_jack", device_get_sync_jack, METH_VARARGS, "Get the device jack status for the synchronization in and synchronization out connectors."},
        {"device_get_color_control", device_get_color_control, METH_VARARGS, "Get device color control."},
        {"device_set_color_control", device_set_color_control, METH_VARARGS, "Set device color control."},
        {"device_get_color_image_device_timestamp_usec", device_get_color_image_device_timestamp_usec, METH_VARARGS, "get color image timestamp"},
        {"device_get_depth_image_device_timestamp_usec", device_get_depth_image_device_timestamp_usec, METH_VARARGS, "get color image timestamp"},
        {NULL, NULL, 0, NULL}
    };

    static int pyk4a_traverse(PyObject *m, visitproc visit, void *arg)
    {
        Py_VISIT(GETSTATE(m)->error);
        return 0;
    }

    static int pyk4a_clear(PyObject *m)
    {
        Py_CLEAR(GETSTATE(m)->error);
        return 0;
    }

    static struct PyModuleDef moduledef =
    {
        PyModuleDef_HEAD_INIT,
        "k4a_module",
        NULL,
        sizeof(struct module_state),
        Pyk4aMethods,
        NULL,
        pyk4a_traverse,
        pyk4a_clear,
        NULL
    };
#define INITERROR return NULL


    //########################
    // Module init function
    //########################
    PyMODINIT_FUNC PyInit_k4a_module(void) {
        import_array();
        PyObject *module = PyModule_Create(&moduledef);

        if (module == NULL)
            INITERROR;
        struct module_state *st = GETSTATE(module);

        st->error = PyErr_NewException("pyk4a_module.Error", NULL, NULL);
        if (st->error == NULL)
        {
            Py_DECREF(module);
            INITERROR;
        }
        return module;
    }

#ifdef __cplusplus
}
#endif

