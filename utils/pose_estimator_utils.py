import sys
import torch

sys.path.append('../')
from parameters import parameters

if torch.cuda.is_available() is True:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def camera_matrix(cam_idx, use_cuda=True):
    fx = parameters.fx[cam_idx]
    fy = parameters.fy[cam_idx]
    cx = parameters.cx[cam_idx]
    cy = parameters.cy[cam_idx]

    if torch.cuda.is_available() is True and use_cuda is True:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return torch.tensor([[fx,   0.0,   cx],
                        [0.0,    fy,   cy],
                        [0.0,   0.0,  1.0]], device=device)

def from_homogeneous(v):
    return (v/v[-1])[:-1]

def from_homogeneous2(v):
    return (v/v[-1])


def get_distortion_coefficients(cam_idx):
    kd = [parameters.kd0[cam_idx], parameters.kd1[cam_idx], parameters.kd2[cam_idx]]
    return torch.tensor(kd, device = device)


def apply_distortion(kd, v):
    v2 = v.clone()
    r = torch.norm(v[:-1][:], dim=0)
    r = r*r
    v2[0][:] = v[0][:]*(1 + kd[0]*r + kd[1]*r*r + kd[2]*r*r*r)
    v2[1][:] = v[1][:]*(1 + kd[0]*r + kd[1]*r*r + kd[2]*r*r*r)
    return v2

