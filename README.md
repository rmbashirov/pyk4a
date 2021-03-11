# pyk4a

A fork from [this](https://github.com/etiennedub/pyk4a) repo.

Python 3 wrapper for azure kinect streaming.

## Requirements
* install azure kinect sensor and body tracking sdks, see [setup_k4a.md](setup_k4a.md)
* install opencv, see [setup_opencv.md](setup_opencv.md)
* install nlohmann json:
  - `cd` to dir you like
  - `git clone --recursive https://github.com/nlohmann/json ./nlohmann_json`
  - `cd nlohmann_json`
  - `mkdir build && cd build`
  - `cmake ..`
  - `cmake --build . -- -j$(nproc)`
  - `sudo make install`
  - ensure you have `/usr/local/include/nlohmann/json.hpp`
* add `export OMP_WAIT_POLICY=Passive` to rc file (to `~/.bashrc` or `~/.zshrc` or whatever terminal you use)

## Setup
`./setup.sh`

## Test
* `python -m pyk4a.viewer --undistort_depth --vis_depth --parallel_bt`
* `python -m pyk4a.viewer --vis_color --no_bt --no_depth`
* `python -m pyk4a.viewer --no_depth --parallel_bt --fps 30 --dump_frames 300 --dump_filepath ~/Desktop/tmp/dump.pickle`


