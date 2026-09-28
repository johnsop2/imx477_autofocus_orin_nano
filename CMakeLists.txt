cmake_minimum_required(VERSION 3.12.0)
project(camera_autofocus VERSION 0.1.0 LANGUAGES CXX CUDA)

# ------------------------------------------------------------
# C++ / CUDA
# ------------------------------------------------------------

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Jetson Orin Nano = Ampere, compute capability 8.7
set(CMAKE_CUDA_ARCHITECTURES 87)

# ------------------------------------------------------------
# Optimization
# ------------------------------------------------------------

set(CMAKE_BUILD_TYPE Release)

add_compile_options(
    -O3
)

# CUDA-specific optimization
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# ------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------

find_package(CUDAToolkit REQUIRED)

find_package(PkgConfig REQUIRED)

pkg_check_modules(
    GSTREAMER
    REQUIRED
    gstreamer-1.0
)

pkg_check_modules(
    GSTREAMER_APP
    REQUIRED
    gstreamer-app-1.0
)

pkg_check_modules(
    GSTREAMER_VIDEO
    REQUIRED
    gstreamer-video-1.0
)

# ------------------------------------------------------------
# Executable
# ------------------------------------------------------------

add_executable(
    camera_autofocus
    camera_autofocus.cpp
    laplacian.cu
)

# ------------------------------------------------------------
# Include directories
# ------------------------------------------------------------

target_include_directories(
    camera_autofocus
    PRIVATE

    ${OpenCV_INCLUDE_DIRS}

    ${GSTREAMER_INCLUDE_DIRS}
    ${GSTREAMER_APP_INCLUDE_DIRS}
    ${GSTREAMER_VIDEO_INCLUDE_DIRS}
)

# ------------------------------------------------------------
# Libraries
# ------------------------------------------------------------

target_link_libraries(
    camera_autofocus
    PRIVATE

    ${OpenCV_LIBS}

    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_APP_LIBRARIES}
    ${GSTREAMER_VIDEO_LIBRARIES}

    CUDA::cudart

    pthread
)

# ------------------------------------------------------------
# Linker flags from pkg-config
# ------------------------------------------------------------

target_link_directories(
    camera_autofocus
    PRIVATE

    ${GSTREAMER_LIBRARY_DIRS}
    ${GSTREAMER_APP_LIBRARY_DIRS}
    ${GSTREAMER_VIDEO_LIBRARY_DIRS}
)

# ------------------------------------------------------------
# GStreamer compile flags
# ------------------------------------------------------------

target_compile_options(
    camera_autofocus
    PRIVATE

    ${GSTREAMER_CFLAGS_OTHER}
    ${GSTREAMER_APP_CFLAGS_OTHER}
    ${GSTREAMER_VIDEO_CFLAGS_OTHER}
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=1217>
)

# ------------------------------------------------------------
# CUDA separable compilation
# ------------------------------------------------------------

set_target_properties(
    camera_autofocus
    PROPERTIES

    CUDA_SEPARABLE_COMPILATION ON

    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ------------------------------------------------------------
# Helpful warnings
# ------------------------------------------------------------

target_compile_options(
    camera_autofocus
    PRIVATE

    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>

    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
)
