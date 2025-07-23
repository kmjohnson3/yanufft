#include <Python.h>
#include <ATen/Operators.h>
#include <ATen/ATen.h>
#include <torch/all.h>
#include <torch/library.h>
#include <omp.h>
#include <vector>


extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace yanufft {

class KspaceKernel {
public:
    virtual float kernel(float, float) {
        return 0.0;
    }
};


class SplineKspaceKernel : public KspaceKernel {
public:
    float kernel(float x, float order) override {

        if( fabs(x) > 1){
            return 0.0;
        }

        if(order==0){
            return(1.0);
        }else if(order == 1){
            return(1.0 - fabs(x));
        }else if(order == 2){
            if(fabs(x) > (1.0/3.0)){
                return(9.0 / 8.0 * powf(1.0 - fabs(x), 2.0));
            }else{
                return(3.0 / 4.0 * (1.0 - 3.0 * x*x));
            }
        }//order
        return 0.0;
    }//kernel
    
}; //class


class KBKspaceKernel : public KspaceKernel {
public:
    float kernel(float x, float beta) override {
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
};



/// Forward NUFFT with Image to K-space operator.
///
/// **Example:**
/// ```python
/// # Here is a Python code block
/// def foo(lst: list[int]) -> list[int]:
///   return [ x ** 2 for x in lst ]
/// ```
///
/// @param input  Image tensor, expected to be a 3D tensor of complex64 values.
/// @param coord  Coordinate tensor, expected to be a 2D tensor of float32 values of size (N, 3).
/// @param width  Width of the kernel in each dimension.
/// @param beta   Beta parameter for the kernel, used to control the shape of the kernel
/// @param chop   If true, the kernel is chopped to perform fft shift
///
/// @return kdata Tensor of size (N,) and type complex64 
///
at::Tensor nufft_interpolation(at::Tensor input, at::Tensor coords, double width, double beta, bool chop) {
    
    // Check input
    TORCH_CHECK(input.device().is_cpu(), "Only CPU tensors supported");
    TORCH_CHECK(input.dtype() == torch::kComplexFloat, "Only complex float64 tensors supported");
    TORCH_CHECK(input.dim() == 3, "Expected 3D tensor");

    // Check coordinates
    TORCH_CHECK(coords.device().is_cpu(), "Only CPU tensors supported");
    TORCH_CHECK(coords.dtype() == torch::kFloat, "Only float32 tensors supported");
    TORCH_CHECK(coords.dim() == 2, "Expected 2D tensor for coordinates");
    TORCH_CHECK(coords.size(1) == 3, "Expected coordinate tensor of size (N, 3)");

    // Grab access to the data for each tensoe
    auto inputA = input.accessor<c10::complex<float>, 3>();
    auto coordsA = coords.accessor<float, 2>();

    // Number of points
    auto Npts = coords.size(0);
    
    // Allocate grid to contain the gridded data
    at::TensorOptions options = torch::TensorOptions().dtype(torch::kComplexFloat);
    at::Tensor kdata = torch::zeros( {Npts}, options);
    auto kdataA = kdata.accessor<c10::complex<float>, 1>();

    // grab the correct kernel
    KspaceKernel* kernel = new KBKspaceKernel();

    // Set kernel parameters
    float half_width_x = width / 2.0;
    float half_width_y = width / 2.0;
    float half_width_z = width / 2.0;
    float param_x = beta;
    float param_y = beta;
    float param_z = beta;
    
    // Grab image size for bounds checking
    auto nz = input.size(0);
    auto ny = input.size(1);
    auto nx = input.size(2);

    // Integrate the kernel to normalize it
    float kernel_norm = 0.0;
    for(float k = 0; k <= 1.0; k+=0.01){
        kernel_norm += 0.02*kernel->kernel(k, param_x);
    }
    kernel_norm = 1.0 / ( kernel_norm * kernel_norm * kernel_norm * half_width_z * half_width_y * half_width_x );
        

    #pragma omp parallel for
    for(auto idx=0; idx<Npts; ++idx){
        
        // Grab coordinate for that point
        auto kz = coordsA[idx][0];
        auto ky = coordsA[idx][1];
        auto kx = coordsA[idx][2];
        
        // Start point
        auto x0 = static_cast<int>(std::ceil(kx - half_width_x));
        auto y0 = static_cast<int>(std::ceil(ky - half_width_y));
        auto z0 = static_cast<int>(std::ceil(kz - half_width_z));

        // End point
        auto x1 = static_cast<int>(std::floor(kx + half_width_x));
        auto y1 = static_cast<int>(std::floor(ky + half_width_y));
        auto z1 = static_cast<int>(std::floor(kz + half_width_z));
        
        // Skip any point not fully in the grid domain
        if( (x0 < 0) || (x1 >= nx) || (y0 < 0) || (y1 >= ny) || (z0 < 0) || (z1 >= nz)){
            continue;
        }

        for(auto z = z0; z <= z1; ++z){
            float wz = kernel_norm * kernel->kernel(( (float)z - kz) / (half_width_z), param_z);
            if( chop ){
                wz *= ((float)(2 * (z % 2) - 1));
            }

            for(auto y = y0; y<= y1; ++y){
                float wy = wz * kernel->kernel(((float)y - ky) / (half_width_y), param_y);
                if( chop ){
                    wy *= ((float)(2 * (y % 2) - 1));
                }

                for(auto x = x0; x<= x1; ++x){
                    float w = wy * kernel->kernel(((float)x - kx) / (half_width_x), param_x);
                    if( chop ){
                        w *= ((float)(2 * (x % 2) - 1));
                    }

                    kdataA[idx] += w * inputA[z][y][x];
                    
                } // x-kernel
            } // y-kernel
        } // z-kernel

    } // Points

    return kdata;
}

/// Adjoint NUFFT with K-space to image domain operator.
///
/// **Example:**
/// ```python
/// # Here is a Python code block
/// def foo(lst: list[int]) -> list[int]:
///   return [ x ** 2 for x in lst ]
/// ```
///
/// @param kdata  Tensor of size (N,) and type complex64 
/// @param coord  Coordinate tensor, expected to be a 2D tensor of float32 values of size (N, 3).
/// @param N      Tensor of size (3,) and type int describing the size of the grid
/// @param width  Width of the kernel in each dimension.
/// @param beta   Beta parameter for the kernel, used to control the shape of the kernel
/// @param chop   If true, the kernel is chopped to perform fft shift
///
/// @return input  Image tensor, expected to be a 3D tensor of complex64 values.
///
at::Tensor nufft_gridding(at::Tensor kdata, at::Tensor coords, at::Tensor N, double width, double beta, bool chop) {
    
    // Check input
    TORCH_CHECK(kdata.device().is_cpu(), "Only CPU tensors supported");
    TORCH_CHECK(kdata.dtype() == torch::kComplexFloat, "Only complex float64 tensors supported");
    TORCH_CHECK(kdata.dim() == 1, "Expected 1D tensor for kdata");

    // Check coordinates
    TORCH_CHECK(coords.device().is_cpu(), "Only CPU tensors supported");
    TORCH_CHECK(coords.dtype() == torch::kFloat, "Only float32 tensors supported");
    TORCH_CHECK(coords.dim() == 2, "Expected 2D tensor for coordinates");
    TORCH_CHECK(coords.size(1) == 3, "Expected coordinate tensor of size (N, 3)");

    // Check image size
    TORCH_CHECK(N.device().is_cpu(), "Only CPU tensors supported");
    TORCH_CHECK(N.dim() == 1, "Expected 1D tensor for image size");

    // Grab access to the data for each tensoe
    auto kdataA = kdata.accessor<c10::complex<float>, 1>();
    auto coordsA = coords.accessor<float, 2>();

    // Number of points
    auto Npts = coords.size(0);
    
    // grab image size
    auto nz = N.accessor<long, 1>()[0];
    auto ny = N.accessor<long, 1>()[1];
    auto nx = N.accessor<long, 1>()[2];
    
    // Allocate grid to contain the gridded data
    at::TensorOptions options = torch::TensorOptions().dtype(torch::kComplexFloat);
    at::Tensor grid = torch::zeros( {nz, ny, nx}, options);
    auto gridA = grid.accessor<c10::complex<float>, 3>();

    // grab the correct kernel
    KspaceKernel* kernel = new KBKspaceKernel();

    // Set kernel parameters
    float half_width_x = width / 2.0;
    float half_width_y = width / 2.0;
    float half_width_z = width / 2.0;
    float param_x = beta;
    float param_y = beta;
    float param_z = beta;
 
    // Integrate the kernel to normalize it
    float kernel_norm = 0.0;
    for(float k = 0; k <= 1.0; k+=0.01){
        kernel_norm += 0.02*kernel->kernel(k, param_x);
    }
    kernel_norm = 1.0 / ( kernel_norm * kernel_norm * kernel_norm * half_width_z * half_width_y * half_width_x );
     
    #pragma omp parallel for
    for(auto idx=0; idx<Npts; ++idx){
        
        // Grab coordinate for that point
        auto kz = coordsA[idx][0];
        auto ky = coordsA[idx][1];
        auto kx = coordsA[idx][2];
        
        // Start point
        auto x0 = static_cast<int>(std::ceil(kx - half_width_x));
        auto y0 = static_cast<int>(std::ceil(ky - half_width_y));
        auto z0 = static_cast<int>(std::ceil(kz - half_width_z));

        // End point
        auto x1 = static_cast<int>(std::floor(kx + half_width_x));
        auto y1 = static_cast<int>(std::floor(ky + half_width_y));
        auto z1 = static_cast<int>(std::floor(kz + half_width_z));
        
        // Skip any point not fully in the grid domain
        if( (x0 < 0) || (x1 >= nx) || (y0 < 0) || (y1 >= ny) || (z0 < 0) || (z1 >= nz)){
            continue;
        }

        for(auto z = z0; z <= z1; ++z){
            float wz = kernel_norm*kernel->kernel(((float)z - kz) / (half_width_z), param_z);
            if (chop){
                wz *= ((float)(2 * (z % 2) - 1));
            }

            for(auto y = y0; y<= y1; ++y){
                float wy = wz * kernel->kernel(((float)y - ky) / (half_width_y), param_y);
                if (chop){
                    wy *= ((float)(2 * (y % 2) - 1));
                }

                for(auto x = x0; x<= x1; ++x){
                    float w = wy * kernel->kernel(((float)x - kx) / (half_width_x), param_x);
                    if (chop){
                        w *= ((float)(2 * (x % 2) - 1));
                    }
                    
                    // Grab weighted data point
                    c10::complex<float> weighted_value = w * kdataA[idx];
                    float real_weighted_value = weighted_value.real();
                    float imag_weighted_value = weighted_value.imag();

                    // Grab pointers to grid location
                    float *R = reinterpret_cast<float *>(&gridA[z][y][x].real_);
                    float *I = reinterpret_cast<float *>(&gridA[z][y][x].imag_);
                    
                    // Atomic adds to prevent race conditions
                    #pragma omp atomic
                    *R += real_weighted_value;
                    
                    #pragma omp atomic
                    *I += imag_weighted_value;

                } // x-kernel
            } // y-kernel
        } // z-kernel

    } // Points

    return grid;
}

TORCH_LIBRARY(yanufft, m) {
   m.def("nufft_interpolation(Tensor input, Tensor coords, float width, float beta, bool chop) -> Tensor");
   m.def("nufft_gridding(Tensor input, Tensor coords, Tensor N, float width, float beta, bool chop) -> Tensor");
}

TORCH_LIBRARY_IMPL(yanufft, CPU, m) {
  m.impl("nufft_interpolation", &nufft_interpolation);
  m.impl("nufft_gridding", &nufft_gridding);
}

}