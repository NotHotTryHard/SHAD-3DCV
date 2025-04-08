set -o xtrace

setup_root() {
    apt-get install -qq -y                  \
        cmake                               \
        g++                                 \
        gcc                                 \
        git                                 \
        libatlas-base-dev                   \
        libavcodec-dev                      \
        libavformat-dev                     \
        libeigen3-dev                       \
        libgflags-dev                       \
        libgoogle-glog-dev                  \
        libgstreamer-plugins-base1.0-dev    \
        libgstreamer1.0-dev                 \
        libjpeg-dev                         \
        libopenexr-dev                      \
        libpng-dev                          \
        libsuitesparse-dev                  \
        libswscale-dev                      \
        libtiff-dev                         \
        libwebp-dev                         \
        python3-dev                         \
        python3-pip                         \
        python3-tk                          \
        ;

    python3 -m pip install -qq                      \
        imageio==2.16.1                             \
        networkx==2.7.1                             \
        numpy==1.22.2                               \
        packaging==21.3                             \
        pandas==1.4.1                               \
        Pillow==9.0.1                               \
        pyparsing==3.0.7                            \
        python-dateutil==2.8.2                      \
        pytz==2022.1                                \
        PyWavelets==1.3.0                           \
        scikit-image==0.19.2                        \
        scipy==1.8.0                                \
        six==1.16.0                                 \
        tifffile==2022.3.16                         \
        ;

    # Build and install OpenCV + Contrib from source
    git clone -q --depth 1 --branch 4.9.0 https://github.com/opencv/opencv.git
    git clone -q --depth 1 --branch 4.9.0 https://github.com/opencv/opencv_contrib.git
    mkdir build
    cd build
    cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules ../opencv
    make
    make install
}

setup_checker() {
    python3 --version # Python 3.10.12
    python3 -m pip freeze # see list above
    python3 -c 'import numpy; import skimage.io; import cv2'
}

"$@"