## this is a code for simulation of a batch velocities, using the deepwave package
import torch
import deepwave
from deepwave import scalar
from utils import *
from dataset import load_data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## config
ny = 70
nx = 70
dx = 10

# source and reciever
ns, nr = 5, 70
source_locations = torch.zeros(ns, 1, 2, dtype=torch.long, device=device)
receiver_locations = torch.zeros(ns, nr, 2, dtype=torch.long, device=device)

source_locations[...,1] = torch.tensor([[7],[21],[35],[49],[63]])
source_locations[:, 0, 0] = 0

receiver_locations[..., 1] = ((torch.arange(nr)).repeat(ns, 1))
receiver_locations[:, :, 0] = 0

# the wavelet
freq = 15  
nt = 1000  
dt = 0.001  
peak_time = 73/1000  # the peak time of the ricker wavelet's amplitude
source_amplitudes = (
    deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    .repeat(ns, 1, 1).to(device)
) 

# the core simulation
def simulation(v_batch:torch.Tensor):
    seis_data_batch = torch.tensor([], device=device)
    v_batch = v_batch.to(device)
    for v in v_batch:
        if v.ndim == 3: v = v[0]
        out = scalar(v, dx, dt, source_amplitudes=source_amplitudes,
                            source_locations=source_locations,
                            receiver_locations=receiver_locations,
                            accuracy=4,
                            pml_width=20,
                            pml_freq=freq)
        seis_data = out[-1].permute(0,2,1)
        seis_data_batch = torch.cat((seis_data_batch, seis_data.unsqueeze(0)), dim=0)
    return seis_data_batch

if __name__ == '__main__':
    # test the correction of the simulation
    train_loader = load_data('/data1/lr/FWIDATA/train/', 16, shuffle=True)
    (seis_data_batch, v_batch) = next(iter(train_loader))
    # seis_data_hat_batch = simulation(v_batch)
    # print(seis_data_hat_batch.shape)
    # print(fro(seis_data_hat_batch), fro(seis_data_batch), fro(seis_data_batch[:,:,:-1,:]-seis_data_hat_batch[:,:,1:,:]))

    # test the backward of the simulation
    v_batch.requires_grad_(True)
    seis_data_hat_batch = simulation(v_batch)
    fro(seis_data_hat_batch).backward()
    print(v_batch.grad.shape)














