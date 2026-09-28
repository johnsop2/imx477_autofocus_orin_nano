#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

class CudaFocusProcessor
{
public:
    CudaFocusProcessor() = default;
    ~CudaFocusProcessor();

    CudaFocusProcessor(const CudaFocusProcessor&) = delete;
    CudaFocusProcessor& operator=(const CudaFocusProcessor&) = delete;

    bool initialize(int width, int height);
    bool calculate(const uint8_t* y, size_t stride, double& score);

    // Individual metrics from the most recent frame.
    double tenengrad() const { return last_tenengrad_; }
    double laplacian() const { return last_laplacian_; }
    double edge_ratio() const { return last_edge_ratio_; }

private:
    void release();

    int width_ = 0;
    int height_ = 0;

    // Center 25% of frame: 50% width x 50% height.
    int roi_x_ = 0;
    int roi_y_ = 0;
    int roi_width_ = 0;
    int roi_height_ = 0;

    unsigned int grid_x_ = 0;
    unsigned int grid_y_ = 0;
    size_t blocks_ = 0;

    uint8_t* d_y_ = nullptr;
    size_t pitch_ = 0;

    // Per-block accumulated statistics.
    double* d_sum_gradient_ = nullptr;
    double* d_sum_gradient_sq_ = nullptr;
    double* d_sum_laplacian_ = nullptr;
    double* d_sum_laplacian_sq_ = nullptr;
    unsigned long long* d_edge_count_ = nullptr;

    std::vector<double> h_sum_gradient_;
    std::vector<double> h_sum_gradient_sq_;
    std::vector<double> h_sum_laplacian_;
    std::vector<double> h_sum_laplacian_sq_;
    std::vector<unsigned long long> h_edge_count_;

    double last_tenengrad_ = 0.0;
    double last_laplacian_ = 0.0;
    double last_edge_ratio_ = 0.0;

    bool initialized_ = false;
};
