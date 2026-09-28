#include <cstddef>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "laplacian.h"

namespace
{
constexpr int CUDA_BLOCK_X = 32;
constexpr int CUDA_BLOCK_Y = 8;
constexpr int CUDA_THREADS = CUDA_BLOCK_X * CUDA_BLOCK_Y;

// Adaptive edge/noise threshold.
// A gradient magnitude below this is treated as noise.
constexpr double GRADIENT_NOISE_THRESHOLD = 18.0;

// Prevent a few very strong pixels from dominating the metric.
constexpr double GRADIENT_CLAMP = 255.0 * 255.0 * 4.0;
constexpr double LAPLACIAN_CLAMP = 1020.0 * 1020.0;

bool cuda_check(cudaError_t result, const char* operation)
{
    if (result == cudaSuccess)
        return true;

    std::cerr
        << "CUDA error during "
        << operation
        << ": "
        << cudaGetErrorString(result)
        << '\n';

    return false;
}

__global__
void focus_metric_kernel(
    const uint8_t* __restrict__ y,
    size_t pitch,
    int frame_width,
    int frame_height,
    int roi_x,
    int roi_y,
    int roi_width,
    int roi_height,
    double* __restrict__ partial_gradient,
    double* __restrict__ partial_gradient_sq,
    double* __restrict__ partial_laplacian,
    double* __restrict__ partial_laplacian_sq,
    unsigned long long* __restrict__ partial_edges)
{
    __shared__ double shared_gradient[CUDA_THREADS];
    __shared__ double shared_gradient_sq[CUDA_THREADS];
    __shared__ double shared_laplacian[CUDA_THREADS];
    __shared__ double shared_laplacian_sq[CUDA_THREADS];
    __shared__ unsigned long long shared_edges[CUDA_THREADS];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int rx = blockIdx.x * blockDim.x + tx;
    const int ry = blockIdx.y * blockDim.y + ty;

    double gradient_value = 0.0;
    double gradient_sq_value = 0.0;
    double laplacian_value = 0.0;
    double laplacian_sq_value = 0.0;
    unsigned long long edge_value = 0;

    if (rx < roi_width && ry < roi_height)
    {
        const int x = roi_x + rx;
        const int row = roi_y + ry;

        if (x > 0 &&
            x < frame_width - 1 &&
            row > 0 &&
            row < frame_height - 1)
        {
            const uint8_t* current =
                y + static_cast<size_t>(row) * pitch;

            const uint8_t* previous = current - pitch;
            const uint8_t* next = current + pitch;

            const double center = static_cast<double>(current[x]);
            const double left = static_cast<double>(current[x - 1]);
            const double right = static_cast<double>(current[x + 1]);
            const double up = static_cast<double>(previous[x]);
            const double down = static_cast<double>(next[x]);

            // Central-difference gradients.
            const double gx = right - left;
            const double gy = down - up;

            // Tenengrad energy.
            double gradient =
                gx * gx +
                gy * gy;

            // 4-neighbour Laplacian.
            double laplacian =
                4.0 * center
                - left
                - right
                - up
                - down;

            // Suppress weak responses that are more likely to be sensor noise.
            const double gradient_magnitude =
                sqrt(gradient);

            if (gradient_magnitude >= GRADIENT_NOISE_THRESHOLD)
                edge_value = 1;

            // Robust clipping prevents isolated high-frequency/noisy pixels
            // from dominating the focus score.
            if (gradient > GRADIENT_CLAMP)
                gradient = GRADIENT_CLAMP;

            if (laplacian > LAPLACIAN_CLAMP)
                laplacian = LAPLACIAN_CLAMP;
            else if (laplacian < -LAPLACIAN_CLAMP)
                laplacian = -LAPLACIAN_CLAMP;

            gradient_value = gradient;
            gradient_sq_value = gradient * gradient;
            laplacian_value = laplacian;
            laplacian_sq_value = laplacian * laplacian;
        }
    }

    shared_gradient[tid] = gradient_value;
    shared_gradient_sq[tid] = gradient_sq_value;
    shared_laplacian[tid] = laplacian_value;
    shared_laplacian_sq[tid] = laplacian_sq_value;
    shared_edges[tid] = edge_value;

    __syncthreads();

    for (int stride = CUDA_THREADS / 2;
         stride > 0;
         stride >>= 1)
    {
        if (tid < stride)
        {
            shared_gradient[tid] +=
                shared_gradient[tid + stride];

            shared_gradient_sq[tid] +=
                shared_gradient_sq[tid + stride];

            shared_laplacian[tid] +=
                shared_laplacian[tid + stride];

            shared_laplacian_sq[tid] +=
                shared_laplacian_sq[tid + stride];

            shared_edges[tid] +=
                shared_edges[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        const size_t block_index =
            static_cast<size_t>(blockIdx.y) *
            static_cast<size_t>(gridDim.x) +
            static_cast<size_t>(blockIdx.x);

        partial_gradient[block_index] =
            shared_gradient[0];

        partial_gradient_sq[block_index] =
            shared_gradient_sq[0];

        partial_laplacian[block_index] =
            shared_laplacian[0];

        partial_laplacian_sq[block_index] =
            shared_laplacian_sq[0];

        partial_edges[block_index] =
            shared_edges[0];
    }
}

} // namespace

CudaFocusProcessor::~CudaFocusProcessor()
{
    release();
}

bool CudaFocusProcessor::initialize(int width, int height)
{
    if (width < 3 || height < 3)
    {
        std::cerr << "Focus image is too small.\n";
        return false;
    }

    if (initialized_)
        release();

    width_ = width;
    height_ = height;

    int device_count = 0;

    if (!cuda_check(
            cudaGetDeviceCount(&device_count),
            "cudaGetDeviceCount"))
    {
        return false;
    }

    if (device_count == 0)
    {
        std::cerr << "No CUDA device found.\n";
        return false;
    }

    if (!cuda_check(
            cudaSetDevice(0),
            "cudaSetDevice"))
    {
        return false;
    }

    cudaDeviceProp properties{};

    if (!cuda_check(
            cudaGetDeviceProperties(&properties, 0),
            "cudaGetDeviceProperties"))
    {
        return false;
    }

    std::cout
        << "CUDA device: "
        << properties.name
        << '\n'
        << "Compute capability: "
        << properties.major
        << '.'
        << properties.minor
        << '\n';

    // Center 25% of the frame.
    // 50% width x 50% height = 25% area.
    roi_width_ = width_ / 2;
    roi_height_ = height_ / 2;
    roi_x_ = (width_ - roi_width_) / 2;
    roi_y_ = (height_ - roi_height_) / 2;

    if (roi_width_ < 3 || roi_height_ < 3)
    {
        std::cerr << "Focus ROI is too small.\n";
        release();
        return false;
    }

    std::cout
        << "Focus ROI: "
        << roi_width_
        << " x "
        << roi_height_
        << " at ("
        << roi_x_
        << ", "
        << roi_y_
        << ")\n";

    if (!cuda_check(
            cudaMallocPitch(
                reinterpret_cast<void**>(&d_y_),
                &pitch_,
                static_cast<size_t>(width_),
                static_cast<size_t>(height_)),
            "cudaMallocPitch"))
    {
        release();
        return false;
    }

    grid_x_ =
        static_cast<unsigned int>(
            (roi_width_ + CUDA_BLOCK_X - 1) /
            CUDA_BLOCK_X);

    grid_y_ =
        static_cast<unsigned int>(
            (roi_height_ + CUDA_BLOCK_Y - 1) /
            CUDA_BLOCK_Y);

    blocks_ =
        static_cast<size_t>(grid_x_) *
        static_cast<size_t>(grid_y_);

    if (!cuda_check(
            cudaMalloc(
                reinterpret_cast<void**>(&d_sum_gradient_),
                blocks_ * sizeof(double)),
            "cudaMalloc gradient"))
    {
        release();
        return false;
    }

    if (!cuda_check(
            cudaMalloc(
                reinterpret_cast<void**>(&d_sum_gradient_sq_),
                blocks_ * sizeof(double)),
            "cudaMalloc gradient squared"))
    {
        release();
        return false;
    }

    if (!cuda_check(
            cudaMalloc(
                reinterpret_cast<void**>(&d_sum_laplacian_),
                blocks_ * sizeof(double)),
            "cudaMalloc laplacian"))
    {
        release();
        return false;
    }

    if (!cuda_check(
            cudaMalloc(
                reinterpret_cast<void**>(&d_sum_laplacian_sq_),
                blocks_ * sizeof(double)),
            "cudaMalloc laplacian squared"))
    {
        release();
        return false;
    }

    if (!cuda_check(
            cudaMalloc(
                reinterpret_cast<void**>(&d_edge_count_),
                blocks_ * sizeof(unsigned long long)),
            "cudaMalloc edge count"))
    {
        release();
        return false;
    }

    h_sum_gradient_.resize(blocks_);
    h_sum_gradient_sq_.resize(blocks_);
    h_sum_laplacian_.resize(blocks_);
    h_sum_laplacian_sq_.resize(blocks_);
    h_edge_count_.resize(blocks_);

    last_tenengrad_ = 0.0;
    last_laplacian_ = 0.0;
    last_edge_ratio_ = 0.0;

    initialized_ = true;

    std::cout
        << "CUDA focus processor initialized.\n"
        << "Focus metric: Tenengrad + Laplacian variance\n"
        << "Focus ROI area: 25% of frame\n";

    return true;
}

bool CudaFocusProcessor::calculate(
    const uint8_t* y,
    size_t stride,
    double& score)
{
    if (!initialized_ || y == nullptr)
        return false;

    if (stride < static_cast<size_t>(width_))
    {
        std::cerr << "Invalid Y-plane stride.\n";
        return false;
    }

    // The Laplacian and gradients require neighboring pixels.
    // We therefore copy the complete Y plane for this version.
    if (!cuda_check(
            cudaMemcpy2D(
                d_y_,
                pitch_,
                y,
                stride,
                static_cast<size_t>(width_),
                static_cast<size_t>(height_),
                cudaMemcpyHostToDevice),
            "cudaMemcpy2D Y"))
    {
        return false;
    }

    const dim3 block(
        CUDA_BLOCK_X,
        CUDA_BLOCK_Y);

    const dim3 grid(
        grid_x_,
        grid_y_);

    focus_metric_kernel<<<grid, block>>>(
        d_y_,
        pitch_,
        width_,
        height_,
        roi_x_,
        roi_y_,
        roi_width_,
        roi_height_,
        d_sum_gradient_,
        d_sum_gradient_sq_,
        d_sum_laplacian_,
        d_sum_laplacian_sq_,
        d_edge_count_);

    if (!cuda_check(
            cudaGetLastError(),
            "focus_metric_kernel launch"))
    {
        return false;
    }

    if (!cuda_check(
            cudaDeviceSynchronize(),
            "focus_metric_kernel execution"))
    {
        return false;
    }

    if (!cuda_check(
            cudaMemcpy(
                h_sum_gradient_.data(),
                d_sum_gradient_,
                blocks_ * sizeof(double),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy gradient"))
    {
        return false;
    }

    if (!cuda_check(
            cudaMemcpy(
                h_sum_gradient_sq_.data(),
                d_sum_gradient_sq_,
                blocks_ * sizeof(double),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy gradient squared"))
    {
        return false;
    }

    if (!cuda_check(
            cudaMemcpy(
                h_sum_laplacian_.data(),
                d_sum_laplacian_,
                blocks_ * sizeof(double),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy laplacian"))
    {
        return false;
    }

    if (!cuda_check(
            cudaMemcpy(
                h_sum_laplacian_sq_.data(),
                d_sum_laplacian_sq_,
                blocks_ * sizeof(double),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy laplacian squared"))
    {
        return false;
    }

    if (!cuda_check(
            cudaMemcpy(
                h_edge_count_.data(),
                d_edge_count_,
                blocks_ * sizeof(unsigned long long),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy edge count"))
    {
        return false;
    }

    double total_gradient = 0.0;
    double total_gradient_sq = 0.0;
    double total_laplacian = 0.0;
    double total_laplacian_sq = 0.0;
    unsigned long long total_edges = 0;

    for (size_t i = 0; i < blocks_; ++i)
    {
        total_gradient += h_sum_gradient_[i];
        total_gradient_sq += h_sum_gradient_sq_[i];

        total_laplacian += h_sum_laplacian_[i];
        total_laplacian_sq += h_sum_laplacian_sq_[i];

        total_edges += h_edge_count_[i];
    }

    const int valid_width = roi_width_ - 2;
    const int valid_height = roi_height_ - 2;

    if (valid_width <= 0 || valid_height <= 0)
        return false;

    const double pixel_count =
        static_cast<double>(valid_width) *
        static_cast<double>(valid_height);

    // Mean Tenengrad.
    const double mean_gradient =
        total_gradient / pixel_count;

    // Mean squared Laplacian minus squared mean.
    const double mean_laplacian =
        total_laplacian / pixel_count;

    const double mean_laplacian_sq =
        total_laplacian_sq / pixel_count;

    double laplacian_variance =
        mean_laplacian_sq -
        mean_laplacian * mean_laplacian;

    if (laplacian_variance < 0.0)
        laplacian_variance = 0.0;

    // Edge density is useful for detecting textureless scenes.
    const double edge_ratio =
        static_cast<double>(total_edges) /
        pixel_count;

    /*
     * Convert the two metrics to stable dimensionless ranges.
     *
     * Tenengrad is already non-negative. The square-root compression
     * reduces the influence of extremely textured scenes.
     *
     * Laplacian variance receives the same treatment.
     */
    const double normalized_tenengrad =
        sqrt(mean_gradient);

    const double normalized_laplacian =
        sqrt(laplacian_variance);

    /*
     * Combined focus score.
     *
     * Tenengrad gets slightly more weight because it tends to be more
     * stable in the presence of sensor noise, while Laplacian variance
     * supplies sensitivity to fine detail.
     */
    double combined =
        0.60 * normalized_tenengrad +
        0.40 * normalized_laplacian;

    /*
     * If almost no meaningful edges exist in the ROI, the image does
     * not provide a reliable focus measurement. Reduce the score rather
     * than allowing random noise to appear as a focus peak.
     */
    if (edge_ratio < 0.005)
        combined *= edge_ratio / 0.005;

    last_tenengrad_ = normalized_tenengrad;
    last_laplacian_ = normalized_laplacian;
    last_edge_ratio_ = edge_ratio;

    score = combined;

    if (!std::isfinite(score) || score < 0.0)
    {
        score = 0.0;
        return false;
    }

    return true;
}

void CudaFocusProcessor::release()
{
    if (d_y_)
    {
        cudaFree(d_y_);
        d_y_ = nullptr;
    }

    if (d_sum_gradient_)
    {
        cudaFree(d_sum_gradient_);
        d_sum_gradient_ = nullptr;
    }

    if (d_sum_gradient_sq_)
    {
        cudaFree(d_sum_gradient_sq_);
        d_sum_gradient_sq_ = nullptr;
    }

    if (d_sum_laplacian_)
    {
        cudaFree(d_sum_laplacian_);
        d_sum_laplacian_ = nullptr;
    }

    if (d_sum_laplacian_sq_)
    {
        cudaFree(d_sum_laplacian_sq_);
        d_sum_laplacian_sq_ = nullptr;
    }

    if (d_edge_count_)
    {
        cudaFree(d_edge_count_);
        d_edge_count_ = nullptr;
    }

    h_sum_gradient_.clear();
    h_sum_gradient_sq_.clear();
    h_sum_laplacian_.clear();
    h_sum_laplacian_sq_.clear();
    h_edge_count_.clear();

    width_ = 0;
    height_ = 0;

    roi_x_ = 0;
    roi_y_ = 0;
    roi_width_ = 0;
    roi_height_ = 0;

    grid_x_ = 0;
    grid_y_ = 0;
    blocks_ = 0;
    pitch_ = 0;

    last_tenengrad_ = 0.0;
    last_laplacian_ = 0.0;
    last_edge_ratio_ = 0.0;

    initialized_ = false;
}
