## 1

```
sudo apt update
sudo apt install -y build-essential cmake pkg-config unzip yasm git checkinstall

sudo apt-get install libturbojpeg0-dev

sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev libavresample-dev
sudo apt install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt install -y libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev 
sudo apt install -y libfaac-dev libmp3lame-dev libvorbis-dev
sudo apt-get install -y libgtk-3-dev
sudo apt-get install -y libtbb-dev
sudo apt-get install -y libatlas-base-dev gfortran
sudo apt-get install -y libprotobuf-dev protobuf-compiler
sudo apt-get install -y libgoogle-glog-dev libgflags-dev
sudo apt-get install -y libgphoto2-dev libeigen3-dev libhdf5-dev doxygen
```

## 2
```
cd ./extern
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip
wget -O opencv_contrib.zip https://github.com//opencv/opencv_contrib/archive/4.2.0.zip
unzip opencv.zip
unzip opencv_contrib.zip
```

## 3
```
cd ./extern/opencv-4.2.0
mkdir -p build && cd build
cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D INSTALL_C_EXAMPLES=OFF \
    -D WITH_TBB=ON \
    -D WITH_CUDA=OFF \
    -D BUILD_opencv_cudacodec=OFF \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=0 \
    -D WITH_V4L=OFF \
    -D WITH_QT=OFF \
    -D WITH_OPENGL=OFF \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_PC_FILE_NAME=opencv.pc \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D BUILD_opencv_java=OFF \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=ON \
    -D PYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -D PYTHON_LIBRARY=$(python -c "from distutils.sysconfig import get_python_lib, get_python_version; import os.path as osp; lib_dp=osp.abspath(osp.join(get_python_lib(), '..', '..')); lib_fp=osp.join(lib_dp, f'libpython{get_python_version()}m.so'); print(lib_fp);") \
    -D PYTHON_PACKAGES_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    -D OPENCV_PYTHON_INSTALL_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    -D PYTHON_EXECUTABLE=$(which python) \
    -D OPENCV_EXTRA_MODULES_PATH="../../opencv_contrib-4.2.0/modules" \
    -D BUILD_EXAMPLES=OFF \
    ..

make -j$(nproc)
sudo make install
```

## 4
**lib dirs** are `/usr/local/lib` and `/usr/lib/x86_64-linux-gnu`, they are set in `setup.py`
* ensure you have `libturbojpeg.so` in **lib dirs**
 * ensure you have `libopencv_*.so` libs in **lib dirs**
* ensure you have `opencv2/core.hpp`, `opencv2/calib3d.hpp`, `opencv2/imgproc.hpp` in `/usr/local/include/opencv4` dir


## 5
* `cd ./extern/opencv-4.2.0`
* `sudo sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'`
* `sudo ldconfig`
* `mkdir -p /usr/local/lib/pkgconfig`
* `sudo cp ./unix-install/opencv.pc /usr/local/lib/pkgconfig/`
* add `export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig` to rc file (to `~/.bashrc` or `~/.zshrc` or whatever terminal you use)
* check: `pkg-config --modversion opencv`
