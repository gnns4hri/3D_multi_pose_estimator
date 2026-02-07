from statistics import median
import sys
import torch
import numpy as np
import itertools
import cv2


if torch.cuda.is_available() is True:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def camera_matrix(cam_idx, parameters, use_cuda=True):
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


def get_distortion_coefficients(cam_idx, parameters, fisheye=False):
    if fisheye:
        kd = [parameters.kd0[cam_idx], parameters.kd1[cam_idx], parameters.kd2[cam_idx], parameters.kd3[cam_idx]]
    else:
        kd = [parameters.kd0[cam_idx], parameters.kd1[cam_idx], parameters.kd2[cam_idx]]
    return torch.tensor(kd, device = device)

def apply_fisheye_distortion(kd, v):
    v2 = v.clone()
    r = torch.norm(v[:-1][:], dim=0)
    theta = torch.atan(r)
    theta2 = theta*theta
    theta4 = theta2*theta2
    theta6 = theta4*theta2
    theta8 = theta4*theta4
    theta_d = theta*(1 + kd[0]*theta2 + kd[1]*theta4 + kd[2]*theta6 + kd[3]*theta8)
    correction = torch.div(theta_d, r)
    v2[0][:] = v[0][:]*correction
    v2[1][:] = v[1][:]*correction
    return v2


def apply_distortion(kd, v):
    v2 = v.clone()
    r = torch.norm(v[:-1][:], dim=0)
    r = r*r
    v2[0][:] = v[0][:]*(1 + kd[0]*r + kd[1]*r*r + kd[2]*r*r*r)
    v2[1][:] = v[1][:]*(1 + kd[0]*r + kd[1]*r*r + kd[2]*r*r*r)
    return v2

def triangulate(points_2D, camera_matrices, distortion_coefficients, projection_matrices, fisheye, median_chek_axis, parameters):
    result3D = dict()
    for idx_i in parameters.joint_list:
        idx = str(idx_i)
        point3d_list = []
        if idx in points_2D.keys() and len(points_2D[idx]) > 1:
            cam_combinations = itertools.combinations(range(len(points_2D[idx].keys())), 2)
            for comb in cam_combinations:
                cam1 = list(points_2D[idx].keys())[comb[0]]
                cam2 = list(points_2D[idx].keys())[comb[1]]
                point1 = np.array(points_2D[idx][cam1])
                if fisheye[cam1]:
                    new_point1 = cv2.fisheye.undistortPoints(np.array([point1]), camera_matrices[cam1], distortion_coefficients[cam1])
                else:
                    new_point1 = cv2.undistortPoints(np.array([point1]), camera_matrices[cam1], distortion_coefficients[cam1])
                point2 = np.array(points_2D[idx][cam2])                                    
                if fisheye[cam2]:
                    new_point2 = cv2.fisheye.undistortPoints(np.array([point2]), camera_matrices[cam2], distortion_coefficients[cam2])
                else:
                    new_point2 = cv2.undistortPoints(np.array([point2]), camera_matrices[cam2], distortion_coefficients[cam2])
                point3d = cv2.triangulatePoints(projection_matrices[cam1], projection_matrices[cam2], new_point1, new_point2)
                point3d = point3d[0:3]/point3d[3]
                point3d_list.append(point3d)
            point3d_list = np.array(point3d_list)
            dist_to_0 = point3d_list[:,median_chek_axis]
            median = np.sort(dist_to_0, axis=0)[dist_to_0.shape[0]//2]
            dist_to_median = np.linalg.norm(dist_to_0-median, axis=1)
            new_point3d_list = [point3d_list[i,:] for i in range(dist_to_median.shape[0]) if dist_to_median[i] < 0.05]
            result3D[idx] = np.mean(np.array(new_point3d_list), axis=0)
    return result3D
