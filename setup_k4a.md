## 1
```
sudo apt-get update
sudo apt-get install -y \
    pkg-config \
    ninja-build \
    doxygen \
    clang \
    gcc-multilib \
    g++-multilib \
    python3 \
    git-lfs \
    nasm \
    cmake \
    libgl1-mesa-dev \
    libsoundio-dev \
    libvulkan-dev \
    libx11-dev \
    libxcursor-dev \
    libxinerama-dev \
    libxrandr-dev \
    libusb-1.0-0-dev \
    libssl-dev \
    libudev-dev \
    mesa-common-dev \
    uuid-dev
```

## 2
- `git clone --recursive https://github.com/microsoft/Azure-Kinect-Sensor-SDK.git ./extern/azure-sdk`
- `cd ./extern/azure-sdk`
- `git checkout 17b644560ce7b4ee7dd921dfff0ae811aa54ede6`
- `sudo cp ./scripts/99-k4a.rules /etc/udev/rules.d/`
- optional:
  - `mkdir build && cd build`
  - `cmake .. -GNinja`
  - `ninja`

## 3
- `curl https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -`
- `sudo apt-add-repository https://packages.microsoft.com/ubuntu/18.04/prod`
- replace `deb https://packages.microsoft.com/ubuntu/18.04/prod bionic main` with `deb [arch=amd64] https://packages.microsoft.com/ubuntu/18.04/prod bionic main` here: `sudo nano /etc/apt/sources.list`
- `sudo apt update`
- `sudo apt-get install -y libk4a1.3 libk4a1.3-dev libk4abt1.0 libk4abt1.0-dev`
