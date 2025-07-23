import torch
import yanufft
import math

if __name__ == "__main__":

    # Test the nufft and nufft_adjoint functions with fake data
    print("Testing NUFFT and NUFFT Adjoint...")

    # Fake data
    N = 256
    Ncoils = 8
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
    dcf = torch.sum(coord**2, dim=-1)

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

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        grid = yanufft.ops.NUFFT.apply(coord, image, smaps)
        torch.cuda.synchronize()  # Wait for all kernels to finish
        end_event.record()
        time_nufft = start_event.elapsed_time(end_event)  # Time in milliseconds

        start_event.record()
        out_image = yanufft.ops.NUFFTadjoint.apply(coord, grid * dcf, smaps)
        torch.cuda.synchronize()  # Wait for all kernels to finish
        end_event.record()
        time_nufft_adjoint = start_event.elapsed_time(end_event)  # Time in milliseconds

        print(f"Device: {device}")
        print(f"    NUFFT shape: {grid.shape}")  # Should be (N
        print(f"    NUFFT time ({device}): {time_nufft} ms")
        print(f"    NUFFT Adjoint time ({device}): {time_nufft_adjoint} ms")
        
        results[device].append({
            'nufft_shape': grid.shape,
            'nufft_time': time_nufft,
            'nufft_adjoint_time': time_nufft_adjoint
        })

    # Uncomment the following lines to visualize the output image
    # image_numpy = out_image.cpu().abs().numpy()

    # import matplotlib.pyplot as plt 
    # plt.figure()
    # plt.subplot(221)
    # plt.imshow(image_numpy[N // 2])
    # plt.subplot(222)
    # plt.imshow(image_numpy[:,N // 2])
    # plt.subplot(223)
    # plt.imshow(image_numpy[:,:,N // 2])
    # plt.subplot(224)
    # plt.imshow(image_numpy[10])
    
    for device in devices:
        torch_device = torch.device(device)
        coord = coord.to(torch_device)
        image = image.to(torch_device)
        smaps = smaps.to(torch_device)
        dcf = dcf.to(torch_device)
        
        coord_kb = coord.clone().permute(1,0)  # Change to (3, Npts)
        coord_kb *= math.pi / (N // 2)  # Scale to [-pi, pi]

        smaps_kb = smaps.clone().unsqueeze(0)  # Change to (1, Ncoils, N, N, N)
        image_kb = image.clone().unsqueeze(0)  # Change to (1, N, N, N)
        
        import torchkbnufft as tkbn
        
        im_size = (N, N, N)
        oversamp = 1.25
        grid_size = (int(N*oversamp), int(N*oversamp), int(N*oversamp))
        nufft_ob = tkbn.KbNufft(im_size=im_size, grid_size=grid_size).to(image)
        adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size).to(image)


        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        kdata = nufft_ob(image_kb, coord_kb, smaps=smaps_kb)
        torch.cuda.synchronize()  # Wait for all kernels to finish
        end_event.record()
        time_nufft = start_event.elapsed_time(end_event)  # Time in milliseconds

        start_event.record()
        kdata = kdata * dcf.unsqueeze(0).unsqueeze(0)  # Apply density compensation
        image_kb = adjnufft_ob(kdata, coord_kb, smaps=smaps_kb)
        torch.cuda.synchronize()  # Wait for all kernels to finish
        end_event.record()
        time_nufft_adjoint = start_event.elapsed_time(end_event)  # Time in
        
        print(f"Device: {device}")
        print(f"    Torch-KbNUFFT shape: {kdata.shape}")  #
        print(f"    Torch-KbNUFFT time ({device}): {time_nufft} ms")
        print(f"    Torch-KbNUFFT Adjoint time ({device}): {time_nufft_adjoint} ms")

        results_tkbn[device].append({
            'nufft_shape': grid.shape,
            'nufft_time': time_nufft,
            'nufft_adjoint_time': time_nufft_adjoint
        })

    for device in devices:

        # Put on the device
        torch_device = torch.device(device)
        coord = coord.to(torch_device)
        image = image.to(torch_device)
        smaps = smaps.to(torch_device)
        dcf = dcf.to(torch_device)

        import sigpy as sp
        import sigpy.mri as mri
        import cupy as cp
        image_sp = sp.from_pytorch(image)
        smaps_sp = sp.from_pytorch(smaps)
        coord_sp = sp.from_pytorch(coord)
        dcf_sp = sp.from_pytorch(dcf)

        sense_op = mri.linop.Sense(smaps_sp, coord=coord_sp, weights=dcf_sp, coil_batch_size=1)
        sense_adjoint = sense_op.H

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        grid = sense_op.apply(image_sp)
        torch.cuda.synchronize()  # Wait for all kernels to finish
        end_event.record()
        time_nufft = start_event.elapsed_time(end_event)  # Time in milliseconds

        start_event.record()
        out_image = sense_adjoint.apply(grid)
        torch.cuda.synchronize()  # Wait for all kernels to finish
        end_event.record()  
        time_nufft_adjoint = start_event.elapsed_time(end_event)  # Time in milliseconds


        print(f"Device: {device}")
        print(f"    Sigpy shape: {grid.shape}")  # Should be (N, N, N)
        print(f"    Sigpy NUFFT time ({device}): {time_nufft} ms")
        print(f"    Sigpy NUFFT Adjoint time ({device}): {time_nufft_adjoint} ms")  

        results_sp[device].append({
            'nufft_shape': grid.shape,
            'nufft_time': time_nufft,
            'nufft_adjoint_time': time_nufft_adjoint
        })

    # %% 

    # Plot the results
    import matplotlib.pyplot as plt
    import numpy as np
        
    algs = ['NUFFT(GPU)', 'NUFFT(GPU) Adjoint', 'NUFFT(CPU)', 'NUFFT(CPU) Adjoint']
    x = np.arange(len(algs))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    c = 1e-3  # Constant to convert ms to seconds for plotting
    nufft_times = {
        'SigPy': (results_sp['cuda'][0]['nufft_time'] * c, results_sp['cuda'][0]['nufft_adjoint_time'] * c, results_sp['cpu'][0]['nufft_time'] * c, results_sp['cpu'][0]['nufft_adjoint_time'] * c),
        'Torch-KbNUFFT': (results_tkbn['cuda'][0]['nufft_time'] * c, results_tkbn['cuda'][0]['nufft_adjoint_time'] * c, results_tkbn['cpu'][0]['nufft_time'] * c, results_tkbn['cpu'][0]['nufft_adjoint_time'] * c),
        'YANUFFT': (results['cuda'][0]['nufft_time'] * c, results['cuda'][0]['nufft_adjoint_time'] * c, results['cpu'][0]['nufft_time'] * c, results['cpu'][0]['nufft_adjoint_time'] * c), 
    }


    # Set font size and line width
    plt.rcParams.update({'font.size': 14, 'lines.linewidth': 2})
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['xtick.major.size'] = 14
    plt.rcParams['ytick.major.size'] = 14    
    plt.rcParams['xtick.major.width'] = 5
    plt.rcParams['ytick.major.width'] = 5
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16 
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['legend.title_fontsize'] = 16
    plt.rcParams['figure.titlesize'] = 18
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#a6cee3', '#1f78b4', '#b2df8a'])
    plt.rcParams['grid.color'] = 'gray'

    fig, ax = plt.subplots(figsize=(12, 6))
    for attribute, measurement in nufft_times.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3, fmt='{:,.1f}')
        multiplier += 1

    ax.set_ylim(0.01, 1000)  # Set y-axis limits for better visibility
    ax.set_ylabel('Time (s)')
    ax.set_title('NUFFT Performance Comparison')
    ax.set_xticks(x + width, algs)
    ax.legend(loc='upper left', ncols=3, fontsize=14)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_yscale('log')
    plt.show()


    # out_image = sp.to_pytorch(out_image, requires_grad=False)
    # out_image = out_image[...,0] + 1j * out_image[...,1]
    # print(f"Sigpy took {time.time() - start_time}")
    # except ImportError:
    #     print("Sigpy is not installed, skipping Sigpy test.")
    # image_numpy = out_image.cpu().abs().numpy()

    # import matplotlib.pyplot as plt 
    # plt.figure()
    # plt.subplot(221)
    # plt.imshow(image_numpy[N // 2])
    # plt.subplot(222)
    # plt.imshow(image_numpy[:,N // 2])
    # plt.subplot(223)
    # plt.imshow(image_numpy[:,:,N // 2])
    # plt.subplot(224)
    # plt.imshow(image_numpy[10])
    # plt.show()


# %%
