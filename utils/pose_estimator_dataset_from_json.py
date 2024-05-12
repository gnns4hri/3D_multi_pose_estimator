import os
import sys
sys.path.append('../')
from parameters import parameters 
number_of_joints = len(parameters.joint_list)
numbers_per_joint = parameters.numbers_per_joint
numbers_per_joint_for_loss = parameters.numbers_per_joint_for_loss


import torch
from torch.utils.data import Dataset
import random
import copy
import json
import pickle
import numpy as np
import cv2
import itertools
from data_augmentation import permutations_generator_random

MAX_COMBINATIONS_NUMBER = 5

sys.path.append('../')
from parameters import parameters 

from pose_estimator_utils import camera_matrix

tm = pickle.load(open(parameters.transformations_path, 'rb'))
camera_i_transforms = []
camera_d_transforms = []
camera_matrices = {}
distortion_coefficients = {}
projection_matrices = {}

for cam_iList, cam_idx in enumerate(parameters.cameras):
    cam = parameters.camera_names[cam_iList]
    # Add the direct transform (root to camera) to the list
    trfm = tm.get_transform("root", cam)
    camera_d_transforms.append(torch.from_numpy(trfm).type(torch.float32))
    # Add the inverse transform (camera to root) to the list
    camera_i_transforms.append(torch.from_numpy(tm.get_transform(cam, "root")).type(torch.float32))
    # Add the camera matrix to the list
    camera_matrices[cam] = camera_matrix(cam_idx, use_cuda=False).cpu().detach().numpy()

    distortion_coefficients[cam] = np.array([parameters.kd0[cam_idx], parameters.kd1[cam_idx], parameters.p1[cam_idx], parameters.p2[cam_idx], parameters.kd2[cam_idx]])

    projection_matrices[cam] = trfm[0:3, :]

def get_skeleton_indices(data):
    skeleton_indices = {}
    for cam in data.keys():
        joints_json = data[cam][0]
        skeletons = json.loads(joints_json)
        n_joints = 0
        index = 0
        for i, skeleton in enumerate(skeletons):
            if len(skeleton)>n_joints:
                n_joints = len(skeleton)
                index = i
        skeleton_indices[cam] = index
    return skeleton_indices

def get_3D_from_triangulation(data, skeleton_indices):
    points_2D = dict()
    for cam in data.keys():
        if cam in parameters.used_cameras:
            joints_json = data[cam][0]
            joints = json.loads(joints_json)
            if not joints:
                continue
            skeleton_index = skeleton_indices[cam]
            for j, pos in joints[skeleton_index].items():
                if j == "ID":
                    continue
                if pos[0] > 0.:
                    if not j in points_2D.keys():
                        points_2D[j] = dict()
                    points_2D[j][cam] = np.array([pos[1], pos[2]])


    result3D = dict()
    for idx_i in parameters.joint_list:
        idx = str(idx_i)
        mean_point3D = np.zeros((3, 1))
        if idx in points_2D.keys() and len(points_2D[idx]) > 1:
            cam_combinations = itertools.combinations(range(len(points_2D[idx].keys())), 2)
            n_comb = 0
            for comb in cam_combinations:
                cam1 = list(points_2D[idx].keys())[comb[0]]
                cam2 = list(points_2D[idx].keys())[comb[1]]
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


from data_augmentation import permutations_generator

image_width = parameters.image_width
image_height = parameters.image_height

class PoseEstimatorDataset(Dataset):
    def __init__(self, input_data, cameras, joint_list, transform=None, data_augmentation=False, reload=False, save=False, device=None):
        """
            input_data
               -> list[str]: List containing paths to the JSON files.
               -> str:       A single string containing the JSON text for a single sample.
            cameras (list of integers): Camera identifiers to be used.
               -> IGNORED IN THE CURRENT IMPLEMENTATION
            joint_list (list of integers): Joint identifiers to be extracted from the dataset.
               -> IGNORED IN THE CURRENT IMPLEMENTATION
        """
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.numbers_per_joint = numbers_per_joint
        self.numbers_per_joint_for_loss = numbers_per_joint_for_loss


        camera_section_length_total = len(parameters.joint_list)*numbers_per_joint_for_loss  # L joints/skeleton, X numbers/joint.
        camera_section_length_input = len(parameters.joint_list)*numbers_per_joint  # L joints/skeleton, X numbers/joint.
        skeleton_length_total = camera_section_length_total * len(parameters.cameras)
        skeleton_length_input = camera_section_length_input * len(parameters.used_cameras)

        self.data = []
        self.orig_data = []

        if reload is True:
            reload_fname = f'{input_data[-1]}.pytorch'
            if os.path.exists(reload_fname):
                loaded = torch.load(reload_fname)
                self.data = loaded['data']
                self.orig_data = loaded['orig_data']
                return

        ignored_names = []

        given = 0
        total = 0
        if type(input_data) is list:
            json_files = input_data

            for f in json_files:  # FOR EACH INPUT FILE
                print(f)
                json_data = json.loads(open(f, "rb").read())
                n_loaded = 0
                n_data = len(json_data)
                for data in json_data:  # FOR EACH SAMPLE IN A JSON FILE
                    view_from_robot = False
                    given += 1
                    flags = [0]*len(parameters.used_cameras)
                    skeleton_indices = get_skeleton_indices(data)
                    results_3D = get_3D_from_triangulation(data, skeleton_indices)
                    error_input = torch.zeros([skeleton_length_total])
                    network_input = torch.zeros([skeleton_length_input])
                    for c in data:  # FOR EACH CAMERA IN A SAMPLE
                        #include 2D information for all the cameras in error_input
                        try:
                            c_index = parameters.camera_names.index(c)
                        except ValueError:
                            # Ignore the sample if the camera is not in the list of cameras to be used while training
                            if c not in ignored_names:
                                print(f'Ignoring {c} because it\'s not in the list of cameras to be used while training')
                                ignored_names.append(c)
                            continue
                        c_offset = c_index * camera_section_length_total
                        skeleton = json.loads(data[c][0])
                        if not skeleton:
                            continue
                        skeleton = skeleton[skeleton_indices[c]]
                        for j, values in skeleton.items():
                            if j == "ID":
                                continue
                            j_offset = int(j) * numbers_per_joint_for_loss
                            error_input[c_offset + j_offset] = values[3]
                            error_input[c_offset + j_offset + 1] = values[1]
                            error_input[c_offset + j_offset + 2] = values[2]
                            error_input[c_offset + j_offset + 3] = values[4]

                        if c in parameters.used_cameras:
                            used_c_index = parameters.used_cameras.index(c)
                            used_c_offset = used_c_index * camera_section_length_input

                            cam_from_root = torch.matmul(camera_i_transforms[c_index], torch.tensor([0.0, 0.0, 0.0, 1.0]))  # world to camera transformation matrix, results_3d)
                            for j, values in skeleton.items():
                                if j == "ID":
                                    continue
                                if values[3] < 1.:
                                    continue
                                flags[used_c_index] = 1
                                view_from_robot = True
                                used_j_offset = int(j) * numbers_per_joint
                                network_input[used_c_offset + used_j_offset] = values[3]
                                network_input[used_c_offset + used_j_offset + 1] = (values[1] - image_width/2) / (image_width/2)
                                network_input[used_c_offset + used_j_offset + 2] = (values[2] - image_height/2) / (image_height/2)
                                network_input[used_c_offset + used_j_offset + 3] = values[4]

                                point = np.array([values[1], values[2]])                                
                                undistorted_point = cv2.undistortPoints(point, camera_matrices[c], distortion_coefficients[c])
                                undistorted_pix_ray = torch.from_numpy(undistorted_point[0][0]).type(torch.float32)
                                pix_ray_from_root = torch.matmul(camera_i_transforms[c_index], torch.cat((undistorted_pix_ray, torch.tensor([1.0, 0.0])))) #perform only rotation
                                network_input[used_c_offset + used_j_offset + 4: used_c_offset + used_j_offset + 7] = cam_from_root[0:3] / 10.
                                network_input[used_c_offset + used_j_offset + 7: used_c_offset + used_j_offset + 10] = pix_ray_from_root[0:3] / 10.

                    if view_from_robot:
                        for c_index in range(len(parameters.used_cameras)):  # Include 3D from triangulation
                            used_c_offset = c_index * camera_section_length_input
                            for j in results_3D:
                                used_j_offset = int(j) * numbers_per_joint
                                network_input[used_c_offset + used_j_offset + 10] = 1. # 3D is available
                                network_input[used_c_offset + used_j_offset + 11: used_c_offset + used_j_offset + 14] = torch.tensor(np.transpose(results_3D[j])[0]) / 10.

                        for combination in permutations_generator_random(flags, self.data_augmentation, MAX_COMBINATIONS_NUMBER):
                            network_input_DA = copy.deepcopy(network_input)
                            for c_index, part in enumerate(combination):
                                c_offset = c_index * camera_section_length_input
                                if part == 0:
                                    for j in parameters.joint_list:
                                        j_offset = int(j) * numbers_per_joint
                                        network_input_DA[c_offset + j_offset: c_offset + j_offset + 10] = 0.
                            total += 1
                            self.data.append(network_input_DA)
                            self.orig_data.append(error_input)

                        n_loaded += 1      
                  
                        if n_loaded % 1000 == 0:
                            print('Loaded', n_loaded, 'of', n_data)

            print(f'Given {given}\nTotal {total}')
        elif type(input_data) is dict:
            skeleton_indices = get_skeleton_indices(input_data)
            results_3D = get_3D_from_triangulation(input_data, skeleton_indices)
            output = torch.zeros([skeleton_length_input])
            for c in input_data:
                if c in parameters.used_cameras:
                    c_index = parameters.camera_names.index(c)
                    used_c_index = parameters.used_cameras.index(c)
                    used_c_offset = used_c_index * camera_section_length_input

                    cam_from_root = torch.matmul(camera_i_transforms[c_index], torch.tensor([0.0, 0.0, 0.0, 1.0])) / 10.  # world to camera transformation matrix, results_3d)                
                    skeleton = json.loads(input_data[c][0])
                    if not skeleton:
                        continue
                    skeleton = skeleton[skeleton_indices[c]]

                    point_list = []
                    for j, values in skeleton.items():
                        if j == "ID": continue
                        point_list.append([values[1], values[2]])
                    if point_list:
                        point_list = np.array(point_list)
                        norm_factors = np.array([[image_width/2, image_height/2]]*point_list.shape[0])
                        normalize_points = (point_list-norm_factors)/norm_factors                    
                        undistorted_point_list = cv2.undistortPoints(point_list, camera_matrices[c], distortion_coefficients[c]).squeeze(axis=1)
                        undistorted_pix_ray_list = torch.from_numpy(undistorted_point_list).type(torch.float32)
                        new_col = torch.tensor([[1.0, 0.0]]*point_list.shape[0])
                        pix_ray_from_root_list = torch.matmul(camera_i_transforms[c_index], torch.cat((undistorted_pix_ray_list, new_col), dim=1).transpose(dim0=1, dim1=0))/10. #perform only rotation
                        pix_ray_from_root_list = pix_ray_from_root_list.transpose(dim0=1, dim1=0)

                    i_point = 0
                    for j, values in skeleton.items():
                        if j == "ID": continue
                        j_offset = int(j) * numbers_per_joint
                        output[used_c_offset + j_offset] = values[3]
                        output[used_c_offset + j_offset + 1] = normalize_points[i_point][0] #(values[1] - image_width/2) / (image_width/2) 
                        output[used_c_offset + j_offset + 2] = normalize_points[i_point][1] #(values[2] - image_height/2) / (image_height/2)
                        output[used_c_offset + j_offset + 3] = values[4]

                        output[used_c_offset + j_offset + 4: used_c_offset + j_offset + 7] = cam_from_root[0:3]
                        output[used_c_offset + j_offset + 7: used_c_offset + j_offset + 10] = pix_ray_from_root_list[i_point][0:3] #pix_ray_from_root[0:3] / 10.
                        i_point += 1

            for c_index in range(len(parameters.used_cameras)):  # Include 3D from triangulation
                used_c_offset = c_index * camera_section_length_input
                for j in results_3D:
                    j_offset = int(j) * numbers_per_joint
                    output[used_c_offset + j_offset + 10] = 1. # 3D is available
                    output[used_c_offset + j_offset + 11: used_c_offset + j_offset + 14] = torch.tensor(np.transpose(results_3D[j])[0]) / 10.

            if torch.sum(torch.abs(output)) > 1:
                self.data.append(output)
            self.orig_data = self.data
        else:
            raise Exception(f'Invalid dataset input {type(input_data)} for json_files. Only list and dict are allowed.')

        if device is None:
            self.data = torch.stack(self.data)
            self.orig_data = torch.stack(self.orig_data)
        else:
            self.data = torch.stack(self.data).to(device=device)
            self.orig_data = torch.stack(self.orig_data).to(device=device)

        if save:
             torch.save({
                'data': self.data,
                'orig_data': self.orig_data
                }, f'{input_data[-1]}.pytorch')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        ret1 = self.data[idx]
        ret2 = self.orig_data[idx]

        if self.transform:
            ret1 = self.transform(ret1)

        return ret1, ret2
