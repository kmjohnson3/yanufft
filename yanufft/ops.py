# -*- coding: utf-8 -*-
"""FFT and non-uniform FFT (NUFFT) functions.

"""

import torch
from torch import Tensor
import math

__all__ = [
    "nufft",
    "nufft_adjoint",
    "NUFFT",
    "NUFFTadjoint",
    "Nufft_op3D",
    "Gridding",
    "Interpolation",
    "estimate_shape",
    "nufft_gridding", 
    "nufft_interpolation"
]


def nufft_gridding(input: Tensor, coord: Tensor, os_shape: Tensor, width: float, beta: float, chop: bool) -> Tensor:

    if( input.dtype == torch.complex64 or input.dtype == torch.complex128):
        # Call the C++ or CUDA backend for gridding
        return torch.ops.yanufft.nufft_gridding.default(input, coord, os_shape, width, beta, chop)
    elif input.dtype == torch.float32 or input.dtype == torch.float64:
        # Cast to complex if not already
        input = input.to(torch.complex64)
        
        # Run the gridding operation (this will call the C++ or CUDA backend)
        output = torch.ops.yanufft.nufft_gridding.default(input, coord, os_shape, width, beta, chop)
        
        # Convert back to float if needed
        if output.dtype == torch.complex64:
            output = output.real
        return output
    else:
        raise TypeError("Unsupported input type. Expected complex or float tensors.")
    

def nufft_interpolation(input: Tensor, coords: Tensor, width: float, beta: float, chop: bool) -> Tensor:

    if input.dtype == torch.complex64 or input.dtype == torch.complex128:
        # Call the C++ or CUDA backend for interpolation
        return torch.ops.yanufft.nufft_interpolation.default(input, coords, width, beta, chop)
    elif input.dtype == torch.float32 or input.dtype == torch.float64:
        # Cast to complex if not already
        input = input.to(torch.complex64)
        
        # Run the interpolation operation (this will call the C++ or CUDA backend)
        output = torch.ops.yanufft.nufft_interpolation.default(input, coords, width, beta, chop)
        
        # Convert back to float if needed
        if output.dtype == torch.complex64:
            output = output.real
        return output
    else:
        raise TypeError("Unsupported input type. Expected complex or float tensors.")


def centered_fft(input: Tensor, s=None, dim= None, norm: str = "ortho") -> Tensor:
    """FFT function that supports centering.

    Args:
        input (array): input array.
        s (None or array of ints): output shape.
        dim (None or array of ints): Axes over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        array: FFT result of dimension oshape.
    """

    tmp = torch.fft.ifftshift(input, dim=dim)
    tmp = torch.fft.fftn(tmp, s=s, dim=dim, norm=norm)
    output = torch.fft.fftshift(tmp, dim=dim)

    return output


def standard_fft(input, s=None, dim=None, norm="ortho"):
    """Standard FFT function without centering.

    Args:
        input (array): input array.
        s (None or array of ints): output shape.
        dim (None or array of ints): Axes over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        array: FFT result of dimension oshape.
    """

    return torch.fft.fftn(input, s=s, dim=dim, norm=norm)


def centered_ifft(input, s=None, dim=None, norm="ortho"):
    """IFFT function that supports centering.

    Args:
        input (array): input array.
        s (None or array of ints): output shape.
        dim (None or array of ints): Axes over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        array: FFT result of dimension oshape.
    """

    tmp = torch.fft.fftshift(input, dim=dim)
    tmp = torch.fft.ifftn(tmp, s=s, dim=dim, norm=norm)
    output = torch.fft.ifftshift(tmp, dim=dim)

    return output


def standard_ifft(input, s=None, dim=None, norm="ortho"):
    """Standard IFFT function without centering.

    Args:
        input (array): input array.
        s (None or array of ints): output shape.
        dim (None or array of ints): Axes over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        array: FFT result of dimension oshape.
    """

    return torch.fft.ifftn(input, s=s, dim=dim, norm=norm)


def nufft(input, coord, oversamp=1.25, width=4):
    """Non-uniform Fast Fourier Transform.

    Args:
        input (tensor): input signal domain array of shape
            (..., n_{ndim - 1}, ..., n_1, n_0),
            where ndim is specified by coord.shape[-1]. The nufft
            is applied on the last ndim axes, and looped over
            the remaining axes.
        coord (tensor): Fourier domain coordinate array of shape (..., ndim).
            ndim determines the number of dimensions to apply the nufft.
            coord[..., i] should be scaled to have its range between
            -n_i // 2, and n_i // 2.
        oversamp (float): oversampling factor.
        width (float): interpolation kernel full-width in terms of
            oversampled grid.

    Returns:
        tensor: Fourier domain data of shape
            input.shape[:-ndim] + coord.shape[:-1].

    References:
        Fessler, J. A., & Sutton, B. P. (2003).
        Nonuniform fast Fourier transforms using min-max interpolation
        IEEE Transactions on Signal Processing, 51(2), 560-574.
        Beatty, P. J., Nishimura, D. G., & Pauly, J. M. (2005).
        Rapid gridding reconstruction with a minimal oversampling ratio.
        IEEE transactions on medical imaging, 24(6), 799-808.

    """
    ndim = coord.shape[-1]
    beta = math.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5
    os_shape = _get_oversamp_shape(input.shape, ndim, oversamp)

    output = input.clone()

    # Apodize
    _apodize(output, ndim, oversamp, width, beta, True)

    # Zero-pad
    output /= torch.prod(torch.tensor(input.shape[-ndim:])) ** 0.5
    output = _zeropad(output, os_shape)

    # FFT
    dim = [d for d in range(-ndim,0)]
    output = standard_fft(output, dim=dim, norm=None)

    # Interpolate
    coord = _scale_coord(coord, input.shape, oversamp,)
    
    # C++ or CUDA backend
    output = nufft_interpolation(output, coord, float(width), float(beta), True)

    return output


def _zeropad(input, target_shape):
    """Zero-pad input to shape.

    Args:
        input (array): input array.
        shape (tuple of ints): output shape.

    Returns:
        array: zero-padded array.
    """

    input_shape = input.shape # Get the full shape
    in_z, in_y, in_x = input_shape[-3:] 
    target_z, target_y, target_x = target_shape

    # Check if padding is needed (target size must be >= input size)
    if not (target_z >= in_z and target_y >= in_y and target_x >= in_x):
        raise ValueError("Target size must be greater than or equal to input size for padding.")

    # Calculate total padding needed
    total_pad_z = target_z - in_z
    total_pad_y = target_y - in_y
    total_pad_x = target_x - in_x
    
    # Calculate padding amounts for each side (centering)
    pad_front = total_pad_z // 2
    pad_back = total_pad_z - pad_front

    pad_top = total_pad_y // 2
    pad_bottom = total_pad_y - pad_top

    pad_left = total_pad_x // 2
    pad_right = total_pad_x - pad_left

    # Create the padding tuple in the order F.pad expects
    # (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
    padding_tuple = (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)

    # --- Apply Padding ---
    padded_tensor = torch.nn.functional.pad(input, padding_tuple, mode='constant', value=0)

    return padded_tensor


def _crop(input, target_shape):
    """Zero-pad input to shape.

    Args:
        input (array): input array.
        shape (tuple of ints): output shape.

    Returns:
        array: zero-padded array.
    """

    input_shape = input.shape # Get the full shape
    in_z, in_y, in_x = input_shape[-3:] 
    target_z, target_y, target_x = target_shape

    # Check if padding is needed (target size must be >= input size)
    if not (target_z <= in_z and target_y <= in_y and target_x <= in_x):
        raise ValueError("Target size must be less than or equal to input size for cropping.")

    # Calculate total padding needed
    total_pad_z = -target_z + in_z
    total_pad_y = -target_y + in_y
    total_pad_x = -target_x + in_x
    
    # Calculate padding amounts for each side (centering)
    pad_front = total_pad_z // 2
    pad_back = total_pad_z - pad_front

    pad_top = total_pad_y // 2
    pad_bottom = total_pad_y - pad_top

    pad_left = total_pad_x // 2
    pad_right = total_pad_x - pad_left

    # Crop 
    cropped_tensor = input[...,pad_front:-pad_back, 
                            pad_top:-pad_bottom, 
                            pad_left:-pad_right]

    return cropped_tensor


def estimate_shape(coord):
    """Estimate array shape from coordinates.

    Shape is estimated by the different between maximum and minimum of
    coordinates in each axis.

    Args:
        coord (array): Coordinates.
    """
    ndim = coord.shape[-1]
    shape = [
        int(coord[..., i].max() - coord[..., i].min()) for i in range(ndim)
    ]

    return shape



def nufft_adjoint(input, coord, oshape=None, oversamp=1.25, width=4):
    """Adjoint non-uniform Fast Fourier Transform.

    Args:
        input (array): input Fourier domain array of shape
            (...) + coord.shape[:-1]. That is, the last dimensions
            of input must match the first dimensions of coord.
            The nufft_adjoint is applied on the last coord.ndim - 1 axes,
            and looped over the remaining axes.
        coord (array): Fourier domain coordinate array of shape (..., ndim).
            ndim determines the number of dimension to apply nufft adjoint.
            coord[..., i] should be scaled to have its range between
            -n_i // 2, and n_i // 2.
        oshape (tuple of ints): output shape of the form
            (..., n_{ndim - 1}, ..., n_1, n_0).
        oversamp (float): oversampling factor.
        width (float): interpolation kernel full-width in terms of
            oversampled grid.

    Returns:
        array: signal domain array with shape specified by oshape.

    """
    ndim = coord.shape[-1]
    beta = math.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5
    if oshape is None:
        oshape = list(input.shape[: -coord.ndim + 1]) + estimate_shape(coord)
    else:
        oshape = list(oshape)

    os_shape = _get_oversamp_shape(oshape, ndim, oversamp)

    # Gridding
    coord = _scale_coord(coord, oshape, oversamp)
    
    # CUDA or C++ backend
    output = nufft_gridding(
        input, coord, torch.tensor(os_shape), float(width), float(beta), True
    )

    # IFFT
    dim = [d for d in range(-ndim, 0)]
    output = standard_ifft(output, dim=dim, norm=None)

    # Crop
    output = _crop(output, oshape)
    output *= torch.prod(torch.tensor(os_shape[-ndim:])) / torch.prod(torch.tensor(oshape[-ndim:])) ** 0.5

    # Apodize
    _apodize(output, ndim, oversamp, width, beta, True)

    return output


def _scale_coord(coord, shape, oversamp):
    ndim = coord.shape[-1]
    output = coord.clone()
    for i in range(-ndim, 0):
        scale = math.ceil(oversamp * shape[i]) / shape[i]
        shift = math.ceil(oversamp * shape[i]) // 2
        output[..., i] *= scale
        output[..., i] += shift

    return output


def _get_oversamp_shape(shape, ndim, oversamp):
    return list(shape)[:-ndim] + [math.ceil(oversamp * i) for i in shape[-ndim:]]


def _apodize(input, ndim, oversamp, width, beta, chop=False):
    
    output = input
    for a in range(-ndim, 0):
        i = output.shape[a]
        os_i = math.ceil(oversamp * i)
        idx = torch.arange(i, device=output.device).to(output.dtype)

        # Calculate apodization
        apod = (
            beta**2 - (math.pi * width * (idx - i // 2) / os_i) ** 2
        ) ** 0.5
        apod /= torch.sinh(apod)

        if chop:
            apod[::2] *= -1

        ## Kernel is normalized to 1 in gridding/interpolation
        #apod /= apod.abs().max()

        output *= apod.reshape([i] + [1] * (-a - 1))

    return output


class Gridding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, coord, os_shape, width, beta):
        """
        :param input: Tensor (total pts,) 
        :param coord: Tensor (total pts, ndim)
        :param os_shape: Tensor (ndim) output shape
        :param width: float. kernel width
        :param beta: float. beta parameter for the kernel
        :param chop: bool. whether to apply chopping
        :return: output, tensor of (xres, yres, zres) 
        """
        ctx.save_for_backward(coord)
        ctx.width = width
        ctx.beta = beta

        # Call the C++ or CUDA backend for gridding
        output = nufft_gridding(input, coord, os_shape, width, beta, False)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):

        # Unpack saved tensors
        coord, = ctx.saved_tensors
        width = ctx.width
        beta = ctx.beta

        # Gradient of coordinates
        grad_input = nufft_interpolation(grad_output, coord, width, beta, False)

        return grad_input, None, None, None, None


class Interpolation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, coord, width, beta):
        """
        :param input: Tensor (total pts,) 
        :param coord: Tensor (total pts, ndim)
        :param os_shape: Tensor (ndim) output shape
        :param width: float. kernel width
        :param beta: float. beta parameter for the kernel
        :param chop: bool. whether to apply chopping
        :return: output, tensor of (xres, yres, zres) 
        """
        ctx.save_for_backward(coord)
        ctx.width = width
        ctx.beta = beta
        ctx.os_shape = input.shape[-coord.shape[-1]:]  # Save the output shape

        # Forward pass
        output = nufft_interpolation(input, coord, width, beta, False)

        return output

    @staticmethod
    def backward(ctx, grad_output):

        # Unpack saved tensors
        coord, = ctx.saved_tensors
        width = ctx.width
        beta = ctx.beta
        os_shape = ctx.os_shape

        # Call the C++ or CUDA backend for gridding
        grad_input = nufft_gridding(grad_output, coord, os_shape, width, beta, False)
        
        return grad_input, None, None, None, None
    

class NUFFT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coord, im, smaps):
        """

        :param coord: Tensor (total pts, ndim). need to be on gpu
        :param im: Tensor (xres, yres, zres). need to be on gpu
        :param smaps: Tensor (c, xres, yres, zres). can be on cpu
        :return: kdata, tensor of (c, total pts) on gpu
        """
        ctx.save_for_backward(coord, im, smaps)
        
        # Create the operator
        A = Nufft_op3D(smaps, coord)

        # Apply forward NUFFT
        output = A.forward(im)

        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, = grad_outputs
        #print(f'grad_output shape {grad_output.shape}')
        
        # Unpack saved tensors
        coord, im, smaps = ctx.saved_tensors

        # Create the operator
        A = Nufft_op3D(smaps, coord)

        # Gradient coordinates
        grad_input = A.backward_forward(im, grad_output)
        
        # Gradient of image
        grad_input_im = A.adjoint(grad_output)
        
        return grad_input, grad_input_im, None


class NUFFTadjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coord, y, smaps):
        ctx.save_for_backward(coord, y, smaps)
        
        # NUFFT Operators
        A = Nufft_op3D(smaps, coord)

        # Adjoint NUFFT
        output = A.adjoint(y)
        return output
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, = grad_outputs
        #print(f'grad_output shape {grad_output.shape}')
        
        # Get saved tensors
        coord, y, smaps = ctx.saved_tensors

        # NUFFT Operators
        A = Nufft_op3D(smaps, coord)

        # Gradient of coordinates        
        grad_input = A.backward_adjoint(y, grad_output)      # (klength, 2)
        
        # Gradient of k-space
        grad_input_y = A.forward(grad_output)
        
        return grad_input, grad_input_y, None


class Nufft_op3D():
    def __init__(self, smaps: Tensor, coord: Tensor):
        self.torch_dtype = torch.float32
        self.torch_cpxdtype = torch.complex64

        self.smaps = smaps
        self.coord = coord

        # Shape of image 
        self.nx = smaps.shape[1]
        self.ny = smaps.shape[2]
        self.nz = smaps.shape[3]

        # Get Image shape
        self.image_shape = smaps.shape[1:]
        self.ncoils = smaps.shape[0]

        # Get grid array coordinates if coord requires gradient
        if coord.requires_grad:
            # Coordinates
            self.xx = torch.arange(self.nx, dtype=self.torch_dtype, device=smaps.device) - self.nx / 2.
            self.xy = torch.arange(self.ny, dtype=self.torch_dtype, device=smaps.device) - self.ny / 2.
            self.xz = torch.arange(self.nz, dtype=self.torch_dtype, device=smaps.device) - self.nz / 2.
            self.XX, self.XY, self.XZ = torch.meshgrid(self.xx, self.xy, self.xz, indexing='ij')

    def forward(self, im: Tensor) -> Tensor:
        
        # Get kspace for each coil
        y = []
        for coil in range(self.ncoils):
            y.append(nufft(self.smaps[coil]*im, self.coord))
        y = torch.stack(y, dim=0)
        
        return y

    def adjoint(self, y: Tensor)-> Tensor:

        # Create adjoint image for each coil
        im = nufft_adjoint(y[0], self.coord, oshape=self.image_shape)*torch.conj(self.smaps[0])
        for coil in range(1,self.ncoils):
            im += nufft_adjoint(y[coil],self.coord, oshape=self.image_shape)*torch.conj(self.smaps[coil])
     
        return im

    def backward_forward(self, im, g):
        
        if self.coord.requires_grad is False:
            return None
        
        else:
            # Gradient with respect to the coordinates
            grad = torch.zeros(self.coord.shape, dtype=self.torch_dtype, device=im.device)
            vec_fx = torch.mul(self.XX, im)
            vec_fy = torch.mul(self.XY, im)
            vec_fz = torch.mul(self.XZ, im)

            # print(f'backward_forward: vec_fx shape {vec_fx.shape}')     # torch.Size([256, 256, 256])

            # CT 03/2023 implementation of Guanhua's paper. See original nufftbindings for Alban's method.
            # The results are the same.
            xrd = self.forward(vec_fx)
            grad[:, 0] = ((torch.conj(xrd.mul_(0 - 1j)).mul_(g)).real * 2).sum(axis=0)
            xrd = self.forward(vec_fy)
            grad[:, 1] = ((torch.conj(xrd.mul_(0 - 1j)).mul_(g)).real * 2).sum(axis=0)
            xrd = self.forward(vec_fz)
            grad[:, 2] = ((torch.conj(xrd.mul_(0 - 1j)).mul_(g)).real * 2).sum(axis=0)

        return grad

    def backward_adjoint(self, y, g):
        
        if self.coord.requires_grad is False:
            return None
        else:
            # Gradient with respect to the coordinates
            grad = torch.zeros(self.coord.shape, dtype=self.torch_dtype, device=y.device)
            
            # CT 03/2023 implementation of Guanhua's paper. See original nufftbindings for Alban's method
            vecx_grad_output = torch.mul(self.XX, g)
            tmp = self.forward(vecx_grad_output)
            grad[:, 0] = ((tmp.mul_(torch.conj(y) * (0 - 1j))).real * 2).sum(axis=0)
            
            vecy_grad_output = torch.mul(self.XY, g)
            tmp = self.forward(vecy_grad_output)
            grad[:, 1] = ((tmp.mul_(torch.conj(y) * (0 - 1j))).real * 2).sum(axis=0)

            vecz_grad_output = torch.mul(self.XZ, g)
            tmp = self.forward(vecz_grad_output)
            grad[:, 2] = ((tmp.mul_(torch.conj(y) * (0 - 1j))).real * 2).sum(axis=0)

        return grad



