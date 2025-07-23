#include <c10/cuda/CUDAException.h>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

namespace yanufft {

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// M_PI is not always defined in C++, especially with MSVC
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Kaiser-Bessel kernel evaluation
__host__ __device__ __forceinline__ float kaiser_bessel_kernel_1d(float x, float beta) {
    if (fabs(x) > 1) {
        return 0.0f;
    }

    x = fabs(x);
    x = beta * sqrtf(1 - x * x);
    
    if (x < 3.75f) {
        auto t =  x / 3.75f;
        t = t * t;
        return (
            1.0 +
            t * (3.5156229 +
            t * (3.0899424 +
            t * (1.2067492 +
            t * (0.2659732 + 
            t * (0.360768e-1 + 
            t * 0.45813e-2)))))
        );
    } else {
        auto t =  3.75f / x;

        return (
            (expf(x) / sqrtf(x)) *
            0.39894228 +
            t * (0.1328592e-1 +
            t * (0.225319e-2 +
            t * (-0.157565e-2 +
            t * (0.916281e-2 +
            t * (-0.2057706e-1 +
            t * (0.2635537e-1 +
            t * (-0.1647633e-1 +
            t * 0.392377e-2)))))))
        );
    }
}


// Gridding Kernel: Spreads non-uniform data to a uniform oversampled grid
__global__ void gridding_kernel_3d(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> k_coords,
    const torch::PackedTensorAccessor32<c10::complex<float>, 1, torch::RestrictPtrTraits> data,
    torch::PackedTensorAccessor32<c10::complex<float>, 3, torch::RestrictPtrTraits> grid,
    const int N_k, const int Nx, const int Ny, const int Nz,
    const float half_width_x, const float half_width_y, const float half_width_z,
    const float beta, const float kernel_norm, const bool chop) {

    // Get the point
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_k) return;

    // K-space coordinates
    float kz = k_coords[tid][0];
    float ky = k_coords[tid][1];
    float kx = k_coords[tid][2];

    // Starting point in the oversampled grid
    int x0 = static_cast<int>(ceilf(kx - half_width_x));
    int y0 = static_cast<int>(ceilf(ky - half_width_y));
    int z0 = static_cast<int>(ceilf(kz - half_width_z));
    
    // Stoping point in the oversampled grid
    int x1 = static_cast<int>(floorf(kx + half_width_x));
    int y1 = static_cast<int>(floorf(ky + half_width_y));
    int z1 = static_cast<int>(floorf(kz + half_width_z));

    if( x0 < 0 || y0 < 0 || z0 < 0 || x1 >= Nx || y1 >= Ny || z1 >= Nz ) {
        return; // Out of bounds, skip this thread
    }
 
    c10::complex<float> data_val = data[tid];
    
    //
    //  Add the point to the grid
    // 
    float wb = kernel_norm;
    for( int z = z0; z <= z1; ++z ) {
        float wz = wb * kaiser_bessel_kernel_1d( ((float)z - kz) / half_width_z, beta);
        if (chop){
            wz *= ((float)(2 * (z % 2) - 1)); // Apply chop
        }

        for( int y = y0; y <= y1; ++y ) {
            float wy = wz * kaiser_bessel_kernel_1d(((float)y - ky) / half_width_y, beta);
            if (chop){
                wy *= ((float)(2 * (y % 2) - 1)); // Apply chop
            }
            
            for( int x = x0; x <= x1; ++x ) {
                
                float wx = wy * kaiser_bessel_kernel_1d(((float)x - kx) / half_width_x, beta);
                if (chop){
                    wx *= ((float)(2 * (x % 2) - 1)); // Apply chop
                }

                // ATOMIC ADD: Multiple threads might write to the same grid location
                atomicAdd(reinterpret_cast<float*>(&grid[z][y][x]), data_val.real() * wx);
                atomicAdd(reinterpret_cast<float*>(&grid[z][y][x]) + 1, data_val.imag() * wx);
            }
        }
    }
}


// Gridding Kernel: Spreads non-uniform data to a uniform oversampled grid
__global__ void interpolation_kernel_3d(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> k_coords,
    torch::PackedTensorAccessor32<c10::complex<float>, 1, torch::RestrictPtrTraits> data,
    const torch::PackedTensorAccessor32<c10::complex<float>, 3, torch::RestrictPtrTraits> grid,
    const int N_k, const int Nx, const int Ny, const int Nz,
    const float half_width_x, const float half_width_y, const float half_width_z,
    const float beta, const float kernel_norm, const bool chop) {

    // Get the point
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_k) return;

    // K-space coordinates
    float kz = k_coords[tid][0];
    float ky = k_coords[tid][1];
    float kx = k_coords[tid][2];

    // Starting point in the oversampled grid
    int x0 = static_cast<int>(ceilf(kx - half_width_x));
    int y0 = static_cast<int>(ceilf(ky - half_width_y));
    int z0 = static_cast<int>(ceilf(kz - half_width_z));
    
    // Stoping point in the oversampled grid
    int x1 = static_cast<int>(floorf(kx + half_width_x));
    int y1 = static_cast<int>(floorf(ky + half_width_y));
    int z1 = static_cast<int>(floorf(kz + half_width_z));
    
   
    if( x0 < 0 || y0 < 0 || z0 < 0 || x1 >= Nx || y1 >= Ny || z1 >= Nz ) {
        data[tid] = c10::complex<float>(0.0f, 0.0f); // Initialize data value to zero
        return; // Out of bounds, skip this thread
    }
     
    //
    //  Gather data from the grid
    // 
    c10::complex<float> accumulated_val(0.0f, 0.0f);
    float wb = kernel_norm;
    for( int z = z0; z <= z1; ++z ) {
        
        float wz = wb * kaiser_bessel_kernel_1d( ((float)z - kz) / half_width_z, beta);
        if(chop) {  
            wz *= ((float)(2 * (z % 2) - 1)); // Apply chop
        }

        for( int y = y0; y <= y1; ++y ) {
            float wy = wz * kaiser_bessel_kernel_1d(((float)y - ky) / half_width_y, beta);
            if(chop) {
                wy *= ((float)(2 * (y % 2) - 1)); // Apply chop
            }
        
            for( int x = x0; x <= x1; ++x ) {
                
                float wx = wy * kaiser_bessel_kernel_1d(((float)x - kx) / half_width_x, beta);
                if(chop) {
                    wx *= ((float)(2 * (x % 2) - 1)); // Apply chop
                }

                accumulated_val += grid[z][y][x] * wx;
            }
        }
    }
    data[tid] = accumulated_val;
}


at::Tensor nufft_gridding_cuda(
    at::Tensor kdata,
    at::Tensor coords,
    at::Tensor N,
    double width,
    double beta,
    bool chop) {

    // Check input
    TORCH_CHECK(kdata.device().is_cuda(), "NUFFT::Gridding Only GPU tensors supported - kdata is not on GPU");
    TORCH_CHECK(kdata.dtype() == torch::kComplexFloat, "NUFFT::Gridding Only complex float64 tensors supported kdata is not complex64");
    TORCH_CHECK(kdata.dim() == 1, "NUFFT::Gridding Expected 1D tensor for kdata");

    // Check coordinates
    TORCH_CHECK(coords.device().is_cuda(), "NUFFT::Gridding Only GPU tensors supported - coords is not on GPU");
    TORCH_CHECK(coords.dtype() == torch::kFloat, "NUFFT::Gridding Only float32 tensors supported");
    TORCH_CHECK(coords.dim() == 2, "NUFFT::Gridding Expected 2D tensor for coordinates");
    TORCH_CHECK(coords.size(1) == 3, "NUFFT::Gridding Expected coordinate tensor of size (N, 3)");

    // Check image size
    // TORCH_CHECK(N.device().is_cuda(), "NUFFT::Gridding Only GPU tensors supported - N is not on GPU");
    TORCH_CHECK(N.dim() == 1, "NUFFT::Gridding Expected 1D tensor for image size");
    TORCH_CHECK(N.size(0) == 3, "NUFFT::Gridding Expected size to be 3D");
      
    // Allocate grid
    int64_t Nz = N[0].item<int64_t>();
    int64_t Ny = N[1].item<int64_t>();
    int64_t Nx = N[2].item<int64_t>();
    at::TensorOptions options = torch::TensorOptions().dtype(torch::kComplexFloat).device(torch::kCUDA);
    at::Tensor grid = torch::zeros( {Nz, Ny, Nx}, kdata.options());
    
    // Half width
    const float half_width_x = ((float)width) / 2.0;
    const float half_width_y = ((float)width) / 2.0;
    const float half_width_z = ((float)width) / 2.0;
    
    // Number of points
    const int N_k = kdata.size(0);
    const int threads = 256;
    const int blocks = (N_k + threads - 1) / threads;
    
    // Integrate the kernel to normalize it
    float kernel_norm = 0.0;
    for(float k = 0; k <= 1.0; k+=0.01){
        kernel_norm += 0.02*kaiser_bessel_kernel_1d(k, beta);
    }
    kernel_norm = 1.0 / ( kernel_norm * kernel_norm * kernel_norm * half_width_z * half_width_y * half_width_x );

    // Launch the kernel    
    gridding_kernel_3d<<<blocks, threads>>>(
        coords.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        kdata.packed_accessor32<c10::complex<float>, 1, torch::RestrictPtrTraits>(),
        grid.packed_accessor32<c10::complex<float>, 3, torch::RestrictPtrTraits>(),
        N_k, 
        Nx, Ny, Nz,
        half_width_x, half_width_y, half_width_z,
        beta, kernel_norm, chop
    );    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return grid;

}


at::Tensor nufft_interpolation_cuda(
    at::Tensor grid,
    at::Tensor coords,
    double width,
    double beta,
    bool chop) {
    
    // Check input
    TORCH_CHECK(grid.device().is_cuda(), "Only GPU tensors supported");
    TORCH_CHECK(grid.dtype() == torch::kComplexFloat, "Only complex float64 tensors supported");
    TORCH_CHECK(grid.dim() == 3, "Expected 3D tensor");

    // Check coordinates
    TORCH_CHECK(coords.device().is_cuda(), "NUFFT::Gridding Only GPU tensors supported");
    TORCH_CHECK(coords.dtype() == torch::kFloat, "NUFFT::Gridding Only float32 tensors supported");
    TORCH_CHECK(coords.dim() == 2, "NUFFT::Gridding Expected 2D tensor for coordinates");
    TORCH_CHECK(coords.size(1) == 3, "NUFFT::Gridding Expected coordinate tensor of size (N, 3)");

    // Allocate grid
    const int64_t Nz = grid.size(0);
    const int64_t Ny = grid.size(1);
    const int64_t Nx = grid.size(2);
    
    // Allocate grid to contain the kspace data
    const int N_k = coords.size(0);
    at::Tensor kdata = torch::zeros( {N_k}, grid.options());
    
    // Half width
    const float half_width_x = ((float)width) / 2.0;
    const float half_width_y = ((float)width) / 2.0;
    const float half_width_z = ((float)width) / 2.0;
    
    // Number of points
    const int threads = 256;
    const int blocks = (N_k + threads - 1) / threads;
    
    // Integrate the kernel to normalize it
    float kernel_norm = 0.0;
    for(float k = 0; k <= 1.0; k+=0.01){
        kernel_norm += 0.02*kaiser_bessel_kernel_1d(k, beta);
    }
    kernel_norm = 1.0 / ( kernel_norm * kernel_norm * kernel_norm * half_width_z * half_width_y * half_width_x );

    // Launch the kernel    
    interpolation_kernel_3d<<<blocks, threads>>>(
        coords.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        kdata.packed_accessor32<c10::complex<float>, 1, torch::RestrictPtrTraits>(),
        grid.packed_accessor32<c10::complex<float>, 3, torch::RestrictPtrTraits>(),
        N_k, 
        Nx, Ny, Nz,
        half_width_x, half_width_y, half_width_z,
        beta, kernel_norm, chop
    );  
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return kdata;

}

TORCH_LIBRARY_IMPL(yanufft, CUDA, m) {
    m.impl("nufft_gridding", &nufft_gridding_cuda);
    m.impl("nufft_interpolation", &nufft_interpolation_cuda);
}

}