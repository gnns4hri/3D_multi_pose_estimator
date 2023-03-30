import sys
import torch
import numpy as np
import itertools
import cv2

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

def triangulate(points_2D, camera_matrices, distortion_coefficients, projection_matrices):
    result3D = dict()
    for idx_i in parameters.joint_list:
        idx = str(idx_i)
        mean_point3D = np.zeros((3, 1))
        if idx in points_2D.keys() and len(points_2D[idx]) > 1:
            cam_combinations = itertools.permutations(range(len(points_2D[idx].keys())), 2)
            n_comb = 0
            for comb in cam_combinations:
                cam1 = list(points_2D[idx].keys())[comb[0]]
                # i_cam1 = parameters.camera_names.index(cam1)
                cam2 = list(points_2D[idx].keys())[comb[1]]
                # i_cam2 = parameters.camera_names.index(cam2)
                point1 = np.array(points_2D[idx][cam1])
                new_point1 = cv2.undistortPoints(np.array([point1]), camera_matrices[cam1], distortion_coefficients[cam1])
                point2 = np.array(points_2D[idx][cam2])
                new_point2 = cv2.undistortPoints(np.array([point2]), camera_matrices[cam2], distortion_coefficients[cam2])
                point3d = cv2.triangulatePoints(projection_matrices[cam1], projection_matrices[cam2], new_point1, new_point2)
                point3d = point3d[0:3]/point3d[3]

                mean_point3D += point3d
                n_comb += 1
            result3D[idx] = mean_point3D/n_comb
    return result3D
