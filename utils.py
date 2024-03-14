## some necessary function to need
import torch

def fro(a:torch.Tensor):
    # calculate the F norm of the tensor
    return torch.sqrt(torch.sum(a**2))