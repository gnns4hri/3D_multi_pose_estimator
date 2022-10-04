import os
import sys
sys.path.append('../')
from parameters import parameters 
number_of_joints = len(parameters.joint_list)
numbers_per_joint = parameters.numbers_per_joint


import torch
from torch.utils.data import Dataset
import copy
import json
import pickle
import numpy as np
import cv2
import itertools

sys.path.append('../')
from parameters import parameters 

from pose_estimator_utils import camera_matrix

def get_distortion_coefficients(cam_idx):
    kd = [parameters.kd0[cam_idx], parameters.kd1[cam_idx], parameters.kd2[cam_idx]]
    return kd


tm = pickle.load(open(parameters.transformations_path, 'rb'))
camera_i_transforms = []
camera_d_transforms = []
camera_matrices = {}
distortion_coefficients = {}
projection_matrices = {}

for cam_idx, cam in enumerate(parameters.camera_names):
    # Add the direct transform (root to camera) to the list
    trfm = tm.get_transform("root", parameters.camera_names[cam_idx])
    camera_d_transforms.append(torch.from_numpy(trfm).type(torch.float32))
    # Add the inverse transform (camera to root) to the list
    camera_i_transforms.append(torch.from_numpy(tm.get_transform(parameters.camera_names[cam_idx], "root")).type(torch.float32))
    # Add the camera matrix to the list
    camera_matrices[cam] = camera_matrix(cam_idx, use_cuda=False).cpu().detach().numpy()

    distortion_coefficients[cam] = np.array([parameters.kd0[cam_idx], parameters.kd1[cam_idx], parameters.p1[cam_idx], parameters.p2[cam_idx], parameters.kd2[cam_idx]])
    # distortion_coefficients.append(get_distortion_coefficients(cam_idx))

    projection_matrices[cam] = trfm[0:3, :]

def get_3D_from_triangulation(data):
    points_2D = dict()
    for cam in data.keys():
        if cam in parameters.used_cameras:
            joints_json = data[cam][0]
            joints = json.loads(joints_json)
            if not joints:
                continue
            for j, pos in joints[0].items():
                if not j in points_2D.keys():
                    points_2D[j] = dict()
                points_2D[j][cam] = np.array([pos[1], pos[2]])


    result3D = dict()
    for idx_i in parameters.joint_list:
        idx = str(idx_i)
        mean_point3D = np.zeros((3, 1))
        if idx in points_2D.keys() and len(points_2D[idx]) > 1:
            cam_combinations = itertools.permutations(range(len(points_2D[idx].keys())), 2)
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
    def __init__(self, input_data, cameras, joint_list, transform=None, data_augmentation=False, reload=False, save=False):
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

        camera_section_length = len(parameters.joint_list)*numbers_per_joint  # L joints/skeleton, X numbers/joint.
        skeleton_length = camera_section_length * len(parameters.cameras)

        self.data = []
        self.orig_data = []

        if reload is True:
            reload_fname = f'{input_data[-1]}.pytorch'
            if os.path.exists(reload_fname):
                loaded = torch.load(reload_fname)
                self.data = loaded['data']
                self.orig_data = loaded['orig_data']
                return


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
                    given += 1
                    n_loaded += 1
                    results_3D = get_3D_from_triangulation(data)
                    output1 = torch.zeros([skeleton_length])
                    normalized_output = torch.zeros([skeleton_length])
                    flags = [0]*len(parameters.camera_names)
                    for c in data:  # FOR EACH CAMERA IN A SAMPLE
                        c_index = parameters.camera_names.index(c)
                        c_offset = c_index * camera_section_length
                        cam_from_root = torch.matmul(camera_i_transforms[c_index], torch.tensor([0.0, 0.0, 0.0, 1.0]))  # world to camera transformation matrix, results_3d)
                        for skeleton in json.loads(data[c][0]):
                            flags[c_index] = 1
                            for j, values in skeleton.items():
                                j_offset = int(j) * numbers_per_joint
                                output1[c_offset + j_offset] = values[3]
                                output1[c_offset + j_offset + 1] = values[1]
                                output1[c_offset + j_offset + 2] = values[2]
                                output1[c_offset + j_offset + 3] = values[4]

                                ############    14 ELEMENTS PER JOINT      ##############
                                point = np.array([values[1], values[2]])                                
                                undistorted_point = cv2.undistortPoints(point, camera_matrices[c], distortion_coefficients[c])
                                undistorted_pix_ray = torch.from_numpy(undistorted_point[0][0]).type(torch.float32)
                                pix_ray_from_root = torch.matmul(camera_i_transforms[c_index], torch.cat((undistorted_pix_ray, torch.tensor([1.0, 0.0])))) #perform only rotation
                                output1[c_offset + j_offset + 4: c_offset + j_offset + 7] = cam_from_root[0:3] / 10.
                                output1[c_offset + j_offset + 7: c_offset + j_offset + 10] = pix_ray_from_root[0:3] / 10.
                                normalized_output[c_offset + j_offset : c_offset + j_offset + 14] = output1[c_offset + j_offset : c_offset + j_offset + 14]
                                ##########################################################

                                # #############    8 ELEMENTS PER JOINT (2D and 3D only)     ##############
                                # if j in results_3D.keys():
                                #     output1[c_offset + j_offset + 4] = 1. # 3D is available
                                #     output1[c_offset + j_offset + 5: c_offset + j_offset + 8] = torch.tensor(np.transpose(results_3D[j])[0]) / 10.
                                # normalized_output[c_offset + j_offset : c_offset + j_offset + 8] = output1[c_offset + j_offset : c_offset + j_offset + 8]
                                # #########################################################################


                                normalized_output[c_offset + j_offset + 1] = (values[1] - image_width/2) / (image_width/2)
                                normalized_output[c_offset + j_offset + 2] = (values[2] - image_height/2) / (image_height/2)

                    for c_index in parameters.cameras:  # Include 3D from triangulation
                        c_offset = c_index * camera_section_length    
                        for j in results_3D:
                            j_offset = int(j) * numbers_per_joint
                            output1[c_offset + j_offset + 10] = 1. # 3D is available
                            output1[c_offset + j_offset + 11: c_offset + j_offset + 14] = torch.tensor(np.transpose(results_3D[j])[0]) / 10.
                            normalized_output[c_offset + j_offset + 10: c_offset + j_offset + 14] = output1[c_offset + j_offset + 10: c_offset + j_offset + 14]

                    if torch.sum(output1) > 1:
                        for combination in permutations_generator(flags, self.data_augmentation):
                            output2 = copy.deepcopy(normalized_output)
                            new_comb = False
                            if sum(combination) > 2:
                                new_comb = True
                                for c_index, part in enumerate(combination):
                                    c_offset = c_index * camera_section_length
                                    if part == 0:
                                        for j in parameters.joint_list:
                                            j_offset = int(j) * numbers_per_joint
                                            output2[c_offset + j_offset: c_offset + j_offset + 10] = 0.
                            if new_comb or np.array_equal(combination,np.array(flags)):
                                total += 1
                                self.data.append(output2)
                                self.orig_data.append(output1)
                    
                    if n_loaded % 1000 == 0:
                        print('Loaded', n_loaded, 'of', n_data)
                    if n_loaded > 10000:
                        break
            print(f'Given {given}\nTotal {total}')
        elif type(input_data) is dict:
            results_3D = get_3D_from_triangulation(input_data)
            output = torch.zeros([skeleton_length])
            for c in input_data:
                c_index = parameters.camera_names.index(c)
                c_offset = c_index * camera_section_length
                cam_from_root = torch.matmul(camera_i_transforms[c_index], torch.tensor([0.0, 0.0, 0.0, 1.0]))  # world to camera transformation matrix, results_3d)                
                for skeleton in json.loads(input_data[c][0]):
                    for j, values in skeleton.items():
                        j_offset = int(j) * numbers_per_joint
                        output[c_offset + j_offset] = values[3]
                        output[c_offset + j_offset + 1] = (values[1] - image_width/2) / (image_width/2) 
                        output[c_offset + j_offset + 2] = (values[2] - image_height/2) / (image_height/2)
                        output[c_offset + j_offset + 3] = values[4]

                        ############    14 ELEMENTS PER JOINT      ##############
                        point = np.array([values[1], values[2]])
                        undistorted_point = cv2.undistortPoints(point, camera_matrices[c], distortion_coefficients[c])
                        undistorted_pix_ray = torch.from_numpy(undistorted_point[0][0]).type(torch.float32)
                        pix_ray_from_root = torch.matmul(camera_i_transforms[c_index], torch.cat((undistorted_pix_ray, torch.tensor([1.0, 0.0])))) #perform only rotation
                        output[c_offset + j_offset + 4: c_offset + j_offset + 7] = cam_from_root[0:3] / 10.
                        output[c_offset + j_offset + 7: c_offset + j_offset + 10] = pix_ray_from_root[0:3] / 10.
                        ##########################################################

                        # #############    8 ELEMENTS PER JOINT (2D and 3D only)     ##############
                        # if j in results_3D.keys():
                        #     output[c_offset + j_offset + 4] = 1
                        #     output[c_offset + j_offset + 5: c_offset + j_offset + 8] = torch.tensor(np.transpose(results_3D[j])[0]) / 10.
                        # ###########################################################                            
            for c_index in parameters.cameras:  # Include 3D from triangulation
                c_offset = c_index * camera_section_length    
                for j in results_3D:
                    j_offset = int(j) * numbers_per_joint
                    output[c_offset + j_offset + 10] = 1. # 3D is available
                    output[c_offset + j_offset + 11: c_offset + j_offset + 14] = torch.tensor(np.transpose(results_3D[j])[0]) / 10.

            if torch.sum(output) > 1:
                self.data.append(output)
            self.orig_data = self.data
        else:
            raise Exception(f'Invalid dataset input {type(input_data)} for json_files. Only list and dict are allowed.')

        if save:
             torch.save({
                'data': self.data,
                'orig_data': self.orig_data
                }, f'{input_data[-1]}.pytorch')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        try:
            ret1 = [self.data[x] for x in idx]
            ret2 = [self.orig_data[x] for x in idx]
        except TypeError:
            ret1 = self.data[idx]
            ret2 = self.orig_data[idx]

        if self.transform:
            ret1 = self.transform(ret1)

        return ret1, ret2
