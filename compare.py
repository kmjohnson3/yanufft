import torch
import yanufft
import math

if __name__ == "__main__":

    # Test the nufft and nufft_adjoint functions with fake data
    print("Testing NUFFT and NUFFT Adjoint...")

    # Fake data
    N = 256
    Ncoils = 1
    image = torch.zeros((N,N,N)).to(torch.complex64)
    image[64:-64,64:-64,64:-64] = 1.0
    smaps = torch.ones((Ncoils,N,N,N),dtype=torch.complex64)

    # 3D Radial coordinates
    kr = torch.linspace(-N/2, N/2, 2*N+1, dtype=torch.float32)

    def sample_isotropic_directions_torch(n_points, device='cpu'):
        phi = torch.rand(n_points, device=device) * 2 * torch.pi  # Uniform in [0, 2π]
        cos_theta = torch.rand(n_points, device=device) * 2 - 1   # Uniform in [-1, 1]
        theta = torch.acos(cos_theta)                             # θ ∈ [0, π]

        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)

        return torch.stack((x, y, z), dim=1)

    n_points = 5000
    direction = sample_isotropic_directions_torch(n_points)

    print(f'direction shape: {direction.shape}')
    print(f'kr shape: {kr.shape}')
    coord = direction[:, None, :] * kr[None,:, None]

    coord = coord.reshape(-1,3)  # Reshape to (Npts, 3)
    Npts = coord.shape[0]       

    dcf = torch.ones((Npts,), dtype=torch.float32)
    #dcf = torch.sum(coord**2, dim=-1)

    kdata = torch.ones((Npts,)).to(torch.complex64)
    Ntensor = torch.tensor([N,N,N])

    if torch.cuda.is_available():
        devices = ["cuda", "cpu"]
    else:
        devices = ["cpu"]


    '''
        Calulate the timing of the NUFFT and NUFFT adjoint operations
    '''

    results = {'cpu': [], 'cuda': []}
    results_tkbn = {'cpu': [], 'cuda': []}
    results_sp = {'cpu': [], 'cuda': []}

    for device in devices:
        torch_device = torch.device(device)
        coord = coord.to(torch_device)
        image = image.to(torch_device)
        smaps = smaps.to(torch_device)
        dcf = dcf.to(torch_device)

        # YANUFFT Results
        kdata = yanufft.ops.NUFFT.apply(coord, image, smaps)
        out_image = yanufft.ops.NUFFTadjoint.apply(coord, kdata * dcf, smaps)

        import sigpy as sp
        import sigpy.mri as mri
        import cupy as cp
        image_sp = sp.from_pytorch(image)
        smaps_sp = sp.from_pytorch(smaps)
        coord_sp = sp.from_pytorch(coord)
        dcf_sp = sp.from_pytorch(dcf)

        sense_op = mri.linop.Sense(smaps_sp, coord=coord_sp, weights=dcf_sp, coil_batch_size=1)
        sense_adjoint = sense_op.H

        kdata_sigpy = sense_op.apply(image_sp)
        out_image_sigpy = sense_adjoint.apply(kdata_sigpy)

        # Get the error
        kdata_sigpy = torch.tensor(sp.to_device(kdata_sigpy, sp.Device(sp.cpu_device))).to(device)
        out_image_sigpy = torch.tensor(sp.to_device(out_image_sigpy, sp.Device(sp.cpu_device))).to(device)

        print(f"Device: {device}")
        print(f'YANUFFT {torch.mean(torch.abs(kdata))} {torch.max(torch.abs(kdata))}')
        print(f'YANUFFT Adjoint {torch.mean(torch.abs(out_image))} {torch.max(torch.abs(out_image))}')

        print(f'SigPy NUFFT {torch.mean(torch.abs(kdata_sigpy))} {torch.max(torch.abs(kdata_sigpy))}')
        print(f'SigPy NUFFT Adjoint {torch.mean(torch.abs(out_image_sigpy))} {torch.max(torch.abs(out_image_sigpy))}')

        print('Ratios:')
        print(f'YANUFFT/SigPy NUFFT: {torch.mean(torch.abs(kdata)) / torch.mean(torch.abs(kdata_sigpy))}')
        print(f'YANUFFT/SigPy NUFFT Adjoint: {torch.mean(torch.abs(out_image)) / torch.mean(torch.abs(out_image_sigpy))}')

        print('Errors:')
        print(f'YANUFFT/SigPy NUFFT Error: {torch.mean(torch.abs(kdata - kdata_sigpy))/ torch.mean(torch.abs(kdata_sigpy))}')
        print(f'YANUFFT/SigPy NUFFT Adjoint Error: {torch.mean(torch.abs(out_image - out_image_sigpy))/ torch.mean(torch.abs(out_image_sigpy))}')
        print('---')


# %%
