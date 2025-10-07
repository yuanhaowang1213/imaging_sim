import torch
import numpy as np
def rodrigues_rotation_matrix(k, theta): # theta: [rad]
    """
    This function implements the Rodrigues rotation matrix.
    """
    # cross-product matrix
    kx, ky, kz = k[0], k[1], k[2]
    K = torch.Tensor([
        [  0, -kz,  ky],
        [ kz,   0, -kx],
        [-ky,  kx,   0]
    ]).to(k.device)
    if not torch.is_tensor(theta):
        theta = torch.Tensor(np.asarray(theta)).to(k.device)
    return torch.eye(3, device=k.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * K @ K
