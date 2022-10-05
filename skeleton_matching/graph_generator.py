import os
import sys
import json
import copy
from collections import namedtuple
import math
import random
from networkx.classes.graph import Graph
import scipy as sp
import time

import torch as th
import dgl
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info
import numpy as np
import pickle
import torch

sys.path.append('../utils')
from pose_estimator_utils import camera_matrix, get_distortion_coefficients, from_homogeneous, apply_distortion
from data_augmentation import add_data_to_json


graphData = namedtuple('graphData', ['src_nodes', 'dst_nodes', 'n_nodes', 'features', 'edge_types', 'edge_norms'])

sys.path.append('../')
from parameters import parameters

joint_list = parameters.joint_list
cameras = parameters.cameras



#def get_distortion_coefficients(cam_idx):
    #kd = [parameters.kd0[cam_idx], parameters.kd1[cam_idx], parameters.kd2[cam_idx]]
    #return kd

def undistort(kd, v):
    r = v[0]*v[0] + v[1]*v[1]
    v[0] = v[0]/(1 + kd[0]*r + kd[1]*r*r + kd[2]*r*r*r)
    v[1] = v[1]/(1 + kd[0]*r + kd[1]*r*r + kd[2]*r*r*r)
    return v

if th.cuda.is_available() is True:
    device = th.device('cuda')
else:
    device = th.device('cpu')

tm = pickle.load(open(parameters.transformations_path, 'rb'))
camera_i_transforms = []
camera_d_transforms = []
camera_matrices = []
inverse_camera_matrices = []
distortion_coefficients = []
for cam_idx, cam in enumerate(parameters.cameras):
    # Add the direct transform (root to camera) to the list
    camera_d_transforms.append(
        torch.from_numpy(tm.get_transform("root", parameters.camera_names[cam_idx])).type(torch.float32))
    # Add the inverse transform (camera to root) to the list
    camera_i_transforms.append(
        torch.from_numpy(tm.get_transform(parameters.camera_names[cam_idx], "root")).type(torch.float32))
    # Add the camera matrix to the list
    camera_matrices.append(camera_matrix(cam_idx))
    # Add the inverse camera matrix to the list
    inverse_camera_matrices.append(torch.inverse(camera_matrix(cam_idx)))
    distortion_coefficients.append(get_distortion_coefficients(cam_idx))

CAMW = parameters.image_width
CAMH = parameters.image_height
USING_3D = False
USING_WORLD_COORDINATES = False


class HumanGraphFromView:
    if parameters.format == 'COCO':
        joints = {'0': "nose", '1': "left_eye", '2': "right_eye", '3': "left_ear", '4': "right_ear",
                        '5': "left_shoulder", '6': "right_shoulder",'7': "left_elbow", '8': "right_elbow",
                        '9': "left_wrist", '10': "right_wrist", '11': "left_hip", '12': "right_hip",
                        '13': "left_knee", '14': "right_knee", '15': "left_ankle", '16': "right_ankle", '17': "neck"}
    elif parameters.format == 'BODY_25':
        joints = {'0': "nose", '1': "neck", '2': "right_shoulder", '3': "right_elbow", '4': "right_hand",
                        '5': "left_shoulder", '6': "left_elbow",'7': "left_hand", '8': "hip",
                        '9': "right_hip", '10': "right_knee", '11': "right_ankle", '12': "left_hip",
                        '13': "left_knee", '14': "left_ankle", '15': "right_eye", '16': "left_eye", '17': "right_ear",
                        '18': "left_ear", '19': "left_foot_ball",  '20': "left_toes",  '21': "left_heel",
                        '22': "right_foot_ball",  '23': "right_toes",  '24': "right_heel" }


    def __init__(self, data, camera, alt):
        super(HumanGraphFromView, self).__init__()
        self.labels = None
        self.num_rels = -1

        self.n_nodes = 0
        self.num_joints = 0
        self.features = None
        self.src_nodes = None
        self.dst_nodes = None
        self.edge_types = None
        self.edge_norms = []
        self.position_by_id = None
        self.typeMap = None
        self.camera = camera
        self.camera_idx = parameters.camera_names.index(self.camera)

        # HumanGraphFromView.init_joints()
        # print('features', len(HumanGraphFromView.get_all_features()))

        if alt == '1':
            self.initializeWithAlternative1(data)
        elif alt == '2':
            self.initializeWithAlternative2(data)
        elif alt == '3':
            self.initializeWithAlternative3(data)            
        else:
            print(f'Unknown network alternative {alt}')
            sys.exit(-1)


    @staticmethod
    def get_node_types_one_hot():
        return ['head', 'edge_node'] + list(HumanGraphFromView.joints.values())
        # if parameters.format == 'COCO':
        #     return ['head', 'nose', 'neck', 'left_ear', 'left_eye', 'left_shoulder', 'left_elbow', 'left_wrist',
        #             'left_hip', 'left_knee', 'left_ankle', 'right_ear', 'right_eye', 'right_shoulder', 'right_elbow',
        #             'right_wrist', 'right_hip', 'right_knee', 'right_ankle', 'edge_node']
        # elif parameters.format == 'BODY_25':
        #     return ['head', 'nose', 'neck', 'left_ear', 'left_eye', 'left_shoulder', 'left_elbow', 'left_wrist',
        #             'left_hip', 'left_knee', 'left_ankle', 'right_ear', 'right_eye', 'right_shoulder', 'right_elbow',
        #             'right_wrist', 'right_hip', 'right_knee', 'right_ankle', 'edge_node']


    @staticmethod
    def get_cam_types():
        return parameters.camera_names

    @staticmethod
    def get_body_parts():
        if parameters.format == 'COCO':
            return {'e', 'ey', 'n', 's', 'el', 'w', 'hip', 'k', 'a', 'ne'}
        elif parameters.format == 'BODY_25':    
            return {'e', 'ey', 'n', 's', 'el', 'hi', 'hip', 'ha', 'he', 'k', 'a', 'ne', 'fb', 'to'}
        else:
            print(f'Unknown skeleton format in graph_generator.py {parameters.format}')
            sys.exit(-1)


        # e  = ear          r = right
        # s  = shoulder     l = left
        # el = elbow        b = body (global_node)
        # ey = eye 
        # w  = wrist
        # hip  = hip (left and right) 
        # hi = hip
        # ha = hand
        # he = heel
        # fb = foot_ball
        # k  = knee 
        # a  = ankle 
        # n  = nose 
        # ne = neck 
        # to = toes
    
    @staticmethod
    def get_body_part_abbreviation():
        return {"nose": 'n', "neck": 'ne', "right_shoulder": 'rs', "right_elbow": 'rel', "right_hand": 'rha',
                "left_shoulder": 'ls', "left_elbow": 'lel',"left_hand": 'lha', "hip": 'hi',
                "right_hip": 'rhip', "right_knee": 'rk', "right_ankle": 'ra', "left_hip": 'lhip',
                "left_knee": 'lk', "left_ankle": 'la', "right_eye": 'rey', "left_eye": 'ley', "right_ear": 're',
                "left_ear": 're', "left_foot_ball": 'lfb',  "left_toes": 'lto',  "left_heel": 'lhe',
                "right_foot_ball": 'rfb',  "right_toes": 'rto',  "right_heel": 'rhe', "right_wrist": 'rw', "left_wrist": 'lw' }


    @staticmethod
    def get_body_rels():
        if parameters.format == 'COCO':
            return {'s_el', 'el_w', 's_hip', 'hip_k', 'k_a', 'n_e', 'n_ne', 'ne_s', 'n_ey'}
        elif parameters.format == 'BODY_25':        
            return {'e_ey', 'n_ey', 'n_ne', 'ne_s', 's_el', 'el_ha', 'ne_hi', 'hi_hip', 'hip_k', 'k_a', 'a_he', 'a_fb', 'fb_to'}
        else:
            print(f'Unknown skeleton format in graph_generator.py {parameters.format}')
            sys.exit(-1)

    @staticmethod
    def get_all_features(alt = '1'):
        if alt == '1':
            return HumanGraphFromView.get_node_types_one_hot() + HumanGraphFromView.get_cam_types() + HumanGraphFromView.get_joint_metric_features() + \
                   HumanGraphFromView.get_other_features()
        elif alt == '2':
            features = ['head', 'edge_node']
            body_parts = HumanGraphFromView.joints.values()
            for cam in HumanGraphFromView.get_cam_types():
                for p in body_parts:
                    features.append(cam + '_' + p + '_i')
                    features.append(cam + '_' + p + '_j')
                    features.append(cam + '_' + p + '_valid')
                    features.append(cam + '_' + p + '_prob')

            return features
        elif alt == '3':
            features = ['head', 'edge_node']
            body_parts = HumanGraphFromView.joints.values()
            for cam in HumanGraphFromView.get_cam_types():
                for p in body_parts:
                    features.append(cam + '_' + p + '_i')
                    features.append(cam + '_' + p + '_j')
                    features.append(cam + '_' + p + '_valid')
                    features.append(cam + '_' + p + '_prob')
                    features.append(cam + '_' + p + '_line_pX')
                    features.append(cam + '_' + p + '_line_pY')
                    features.append(cam + '_' + p + '_line_pZ')
                    features.append(cam + '_' + p + '_line_vX')
                    features.append(cam + '_' + p + '_line_vY')
                    features.append(cam + '_' + p + '_line_vZ')
            return features


    @staticmethod
    def get_joint_metric_features():
        if USING_3D:
            if USING_WORLD_COORDINATES:
                return ['x_position', 'y_position', 'z_position', 'i_coordinate', 'j_coordinate', 'valid3D', 'valid2D',
                        'world_x', 'world_y', 'world_z']
            else:
                return ['x_position', 'y_position', 'z_position', 'i_coordinate', 'j_coordinate', 'valid3D', 'valid2D']
        else:
            return ['i_coordinate', 'j_coordinate', 'valid2D', 'probability']

    @staticmethod
    def get_other_features():
        return ['n_joints']

    @staticmethod
    def get_rels(alt = '1'):
        rels = set()
        if alt == '1':
            body_parts = HumanGraphFromView.get_body_parts()
            body_rels = HumanGraphFromView.get_body_rels()
            # Add body relations
            for relations in body_rels:
                split = relations.split('_')
                if split[0] == 'n':
                    if split[1] == 'ne':
                        rels.add(relations)
                    else:
                        rels.add(split[0] + '_' + 'r' + split[1])
                        rels.add(split[0] + '_' + 'l' + split[1])
                elif split[0] == 'ne':
                    if split[1] == 'hi':
                        rels.add(relations)
                    else:
                        rels.add(split[0] + '_' + 'r' + split[1])
                        rels.add(split[0] + '_' + 'l' + split[1])
                elif split[0] == 'hi':
                    rels.add(split[0] + '_' + 'r' + split[1])
                    rels.add(split[0] + '_' + 'l' + split[1])
                else:
                    rels.add('r' + split[0] + '_' + 'r' + split[1])
                    rels.add('l' + split[0] + '_' + 'l' + split[1])
            # Add pair relations, relations with body (global node) and self relations
            for part in body_parts:
                if part == 'n':
                    rels.add('h_n')
                    rels.add('n_n')  # self-loop
                elif part == 'ne':
                    rels.add('h_ne')
                    rels.add('ne_ne')  # self-loop
                elif part == 'hi':
                    rels.add('h_hi')
                    rels.add('hi_hi')  # self-loop
                else:
                    rels.add('r' + part + '_' + 'l' + part)
                    rels.add('r' + part + '_' + 'r' + part)  # self-loops
                    rels.add('l' + part + '_' + 'l' + part)  # self-loops
                    rels.add('h' + '_' + 'r' + part)
                    rels.add('h' + '_' + 'l' + part)
            # Adding inverses
            for e in list(rels):
                split = e.split('_')
                rels.add(split[1] + '_' + split[0])
            # Add global self relations
            rels.add('h_h')  # self-loop
            # Add relation to edge-nodes
            rels.add('link')
            rels.add('link_link')
        elif alt == '2':
            rels.add('h_h')  # self-loop
            # Add relation to edge-nodes
            rels.add('link')
            rels.add('link_link')
        elif alt == '3':
            rels.add('h_h')  # self-loop
            # Add relation to edge-nodes
            rels.add('link')
            rels.add('link_link')

        return sorted(list(rels))

    def initializeWithAlternative1(self, data):

        # We create a map to store the types of the nodes. We'll use it to compute edges' types
        id_by_type = dict()
        rels = HumanGraphFromView.get_rels()
        self.num_rels = len(rels)

        # Feature dimensions
        all_features = HumanGraphFromView.get_all_features()
        feature_dimensions = len(all_features)
        cameras = self.get_cam_types()

        # self.num_joints = len(data)
        
        #Compute the number of joints
        self.num_joints = 0
        for j, values in data.items():
            if values[3] > 0.5:
                self.num_joints += 1

        # Compute the number of nodes
        # One for superbody (global node) + cameras*joints
        self.n_nodes = 1 + self.num_joints


        self.features = th.zeros([self.n_nodes, feature_dimensions])

        # Nodes variables
        self.typeMap = dict()
        self.position_by_id = {}
        self.src_nodes = []  # List to store source nodes
        self.dst_nodes = []  # List to store destiny nodes

        self.edge_types = []  # List to store the relation of each edge
        self.edge_norms = []  # List to store the norm of each edge

        self.features[0, all_features.index('head')] = 1.
        self.features[0, all_features.index(self.camera)] = 1.
        self.features[0, all_features.index('n_joints')] = self.num_joints / len(self.joints)
        self.typeMap[0] = 'h'
        id_by_type['h'] = 0
        self.src_nodes.append(0)
        self.dst_nodes.append(0)
        self.edge_types.append(rels.index('h_h'))
        self.edge_norms.append([1.])

        max_used_id = 1  # 0 for the superbody (global node)

        for j, values in data.items():
            if values[3] < 0.5:
                continue

            if USING_WORLD_COORDINATES:
                if values[2][3] == 1.:  # valid3D
                    cam_idx = cameras.index(self.camera)
                    world_pos = torch.Tensor(values[2][0:3] + [1.])

                    TR = camera_i_transforms[cam_idx]
                    world_pos = torch.matmul(TR, world_pos)
                    world_pos = from_homogeneous(world_pos)

            joint = self.joints[j]

            abbr = HumanGraphFromView.get_body_part_abbreviation()[joint]
            self.typeMap[max_used_id] = abbr
            id_by_type[abbr] = max_used_id

            
            # if joint == 'nose':  # Special case because it hasn't got underscore
            #     self.typeMap[max_used_id] = 'n'
            #     id_by_type['n'] = max_used_id
            # elif joint == 'hip':
            #     self.typeMap[max_used_id] = 'hi'
            #     id_by_type['hi'] = max_used_id
            # elif joint == 'neck':
            #     self.typeMap[max_used_id] = 'ne'
            #     id_by_type['ne'] = max_used_id
            if joint == 'neck':
                if USING_3D:
                    self.features[0, all_features.index('x_position')] = values[2][0]  # / 10.
                    self.features[0, all_features.index('y_position')] = values[2][1]  # / 10.
                    self.features[0, all_features.index('z_position')] = values[2][2]  # / 10.
                    self.features[0, all_features.index('valid3D')] = values[2][3]
                    if USING_WORLD_COORDINATES and values[2][3] == 1.:
                        self.features[0, all_features.index('world_x')] = world_pos[0]  # / 10.
                        self.features[0, all_features.index('world_y')] = world_pos[1]  # / 10.
                        self.features[0, all_features.index('world_z')] = world_pos[2]  # / 10.

                # self.features[0, all_features.index('i_coordinate')] = (values[1][0] - CAMW / 2) / (CAMW / 2)
                # self.features[0, all_features.index('j_coordinate')] = (CAMH / 2 - values[1][1]) / (CAMH / 2)

                self.features[0, all_features.index('i_coordinate')] = (values[1] - CAMW / 2) / (CAMW / 2)
                self.features[0, all_features.index('j_coordinate')] = (CAMH / 2 - values[2]) / (CAMH / 2)
                self.features[0, all_features.index('valid2D')] = 1.
                self.features[0, all_features.index('probability')] = values[4]

            # elif joint.split('_')[1] == 'elbow' or joint.split('_')[1] == 'eye':  # Special case because the abbreviation has 2 letters (el, ey)
            #     self.typeMap[max_used_id] = joint[0] + joint.split('_')[1][0] + joint.split('_')[1][1]
            #     id_by_type[self.typeMap[max_used_id]] = max_used_id
            # else:
            #     self.typeMap[max_used_id] = joint[0] + joint.split('_')[1][0]
            #     id_by_type[self.typeMap[max_used_id]] = max_used_id
            self.features[max_used_id, all_features.index(joint)] = 1.
            self.features[max_used_id, all_features.index(self.camera)] = 1.
            if USING_3D:
                self.features[max_used_id, all_features.index('x_position')] = values[2][0]  # / 10.
                self.features[max_used_id, all_features.index('y_position')] = values[2][1]  # / 10.
                self.features[max_used_id, all_features.index('z_position')] = values[2][2]  # / 10.
                self.features[max_used_id, all_features.index('valid3D')] = values[2][3]
                if USING_WORLD_COORDINATES and values[2][3] == 1.:
                    self.features[max_used_id, all_features.index('world_x')] = world_pos[0]  # / 10.
                    self.features[max_used_id, all_features.index('world_y')] = world_pos[1]  # / 10.
                    self.features[max_used_id, all_features.index('world_z')] = world_pos[2]  # / 10.

            # self.features[max_used_id, all_features.index('i_coordinate')] = (values[1][0] - CAMW / 2) / (CAMW / 2)
            # self.features[max_used_id, all_features.index('j_coordinate')] = (CAMH / 2 - values[1][1]) / (CAMH / 2)

            self.features[max_used_id, all_features.index('i_coordinate')] = (values[1] - CAMW / 2) / (CAMW / 2)
            self.features[max_used_id, all_features.index('j_coordinate')] = (CAMH / 2 - values[2]) / (CAMH / 2)
            self.features[max_used_id, all_features.index('valid2D')] = 1.
            self.features[max_used_id, all_features.index('probability')] = values[4]
            max_used_id += 1

        # Edges #
        for relation in rels:
            if relation in ['h_h', 'link']:
                continue
            split = relation.split('_')
            node_type1 = split[0]
            node_type2 = split[1]
            if (node_type1 in id_by_type) and (node_type2 in id_by_type):
                self.src_nodes.append(id_by_type[node_type1])
                self.dst_nodes.append(id_by_type[node_type2])
                self.edge_types.append(rels.index(relation))
                self.edge_norms.append([1.])

    def initializeWithAlternative2(self, data):

        # We create a map to store the types of the nodes. We'll use it to compute edges' types

        id_by_type = dict()

        rels = HumanGraphFromView.get_rels('2')
        self.num_rels = len(rels)

        # Feature dimensions
        all_features = HumanGraphFromView.get_all_features('2')
        feature_dimensions = len(all_features)

        # This alternative uses only 1 node
        self.n_nodes = 1


        self.features = th.zeros([self.n_nodes, feature_dimensions])

        # Nodes variables
        self.typeMap = dict()
        self.position_by_id = {}
        self.src_nodes = []  # List to store source nodes
        self.dst_nodes = []  # List to store destiny nodes

        self.edge_types = []  # List to store the relation of each edge
        self.edge_norms = []  # List to store the norm of each edge

        self.features[0, all_features.index('head')] = 1.
        self.typeMap[0] = 'h'
        id_by_type['h'] = 0
        self.src_nodes.append(0)
        self.dst_nodes.append(0)
        self.edge_types.append(rels.index('h_h'))
        self.edge_norms.append([1.])

        self.num_joints = 0
        for j, values in data.items():
            joint = self.joints[j]
            self.features[0, all_features.index(self.camera + '_' + joint + '_i')] = (values[1] - CAMW / 2) / (CAMW / 2)
            self.features[0, all_features.index(self.camera + '_' + joint + '_j')] = (CAMH / 2 - values[2]) / (CAMH / 2)
            self.features[0, all_features.index(self.camera + '_' + joint + '_valid')] = values[3]
            self.features[0, all_features.index(self.camera + '_' + joint + '_prob')] = values[4]
            self.num_joints += 1

    def initializeWithAlternative3(self, data):

        # We create a map to store the types of the nodes. We'll use it to compute edges' types

        id_by_type = dict()

        rels = HumanGraphFromView.get_rels('3')
        self.num_rels = len(rels)

        # Feature dimensions
        all_features = HumanGraphFromView.get_all_features('3')
        feature_dimensions = len(all_features)


        # This alternative uses only 1 node
        self.n_nodes = 1


        self.features = th.zeros([self.n_nodes, feature_dimensions])

        # Nodes variables
        self.typeMap = dict()
        self.position_by_id = {}
        self.src_nodes = []  # List to store source nodes
        self.dst_nodes = []  # List to store destiny nodes

        self.edge_types = []  # List to store the relation of each edge
        self.edge_norms = []  # List to store the norm of each edge

        self.features[0, all_features.index('head')] = 1.
        self.typeMap[0] = 'h'
        id_by_type['h'] = 0
        self.src_nodes.append(0)
        self.dst_nodes.append(0)
        self.edge_types.append(rels.index('h_h'))
        self.edge_norms.append([1.])

        cam_from_root = torch.matmul(camera_i_transforms[self.camera_idx], torch.tensor([0.0, 0.0, 0.0, 1.0]))  # world to camera transformation matrix, results_3d)
        self.num_joints = 0
        for j, values in data.items():
            joint = self.joints[j]
            self.features[0, all_features.index(self.camera + '_' + joint + '_i')] = (values[1] - CAMW / 2) / (CAMW / 2)
            self.features[0, all_features.index(self.camera + '_' + joint + '_j')] = (CAMH / 2 - values[2]) / (CAMH / 2)
            self.features[0, all_features.index(self.camera + '_' + joint + '_valid')] = values[3]
            self.features[0, all_features.index(self.camera + '_' + joint + '_prob')] = values[4]
            self.features[0, all_features.index(self.camera + '_' + joint + '_line_pX')] = cam_from_root[0]
            self.features[0, all_features.index(self.camera + '_' + joint + '_line_pY')] = cam_from_root[1]
            self.features[0, all_features.index(self.camera + '_' + joint + '_line_pZ')] = cam_from_root[2]
            pix_ray = torch.matmul(inverse_camera_matrices[self.camera_idx].to('cpu'), torch.tensor([values[1], values[2], 1.0]))
            undistorted_pix_ray = pix_ray #undistort(distortion_coefficients[self.camera_idx], pix_ray)            
            pix_ray_from_root = torch.matmul(camera_i_transforms[self.camera_idx], torch.cat((undistorted_pix_ray, torch.tensor([0.0])))) #perform only rotation
            self.features[0, all_features.index(self.camera + '_' + joint + '_line_vX')] = pix_ray_from_root[0]
            self.features[0, all_features.index(self.camera + '_' + joint + '_line_vY')] = pix_ray_from_root[1]
            self.features[0, all_features.index(self.camera + '_' + joint + '_line_vZ')] = pix_ray_from_root[2]

            self.num_joints += 1

        




#################################################################
# Class to load the dataset
#################################################################


class MergedMultipleHumansDataset(DGLDataset):
    path_save = 'cache/'

    def __init__(self, paths, probabilities=[1.], limit='100000000', alt=None, mode='train', force_reload=False, verbose=True,
                 debug=False, raw_dir='.'):
        if alt is None:
            print('Alt is None')
            sys.exit(-1)
        self.inputs = []
        self.inputs_indices = []
        if type(paths) == list:
            for path in paths:
                print('PATH', path)
                input_data_from_a_file = json.loads(open(path, "rb").read())
                # if mode != 'test':
                #     input_data_from_a_file = add_data_to_json(input_data_from_a_file, 2)
                input_indices = list(range(len(input_data_from_a_file)))
                if mode != 'test':
                    random.shuffle(input_indices)
                    # input_data_from_a_file = add_data_to_json(input_data_from_a_file, min_number_of_views = 2)
                    # random.shuffle(input_data_from_a_file)
                # print('len', len(input_data_from_a_file))
                self.inputs.append(input_data_from_a_file)  # self.inputs is a list where the elements are list of samples
                self.inputs_indices.append(input_indices)
        elif type(paths) == dict:
            self.inputs.append(paths)
            input = list(range(len(paths)))
            self.inputs_indices.append(input)
        else:
            raise Exception('Unhandled type for MergedMultipleHumansDataset')

        # print('number of input files', len(self.inputs))
        
        self.probabilities = probabilities

        self.mode = mode
        self.alt = alt
        self.graphs = []
        self.labels = []
        self.data = dict()
        self.data['edge_nodes_indices'] = []
        self.data['nodes_camera'] = []
        self.debug = debug
        self.force_reload = force_reload

        if self.mode == 'test':
            self.force_reload = True

        self.device = device
        self.limit = limit

        super(MergedMultipleHumansDataset, self).__init__("MergedMultipleHumansDataset", raw_dir=".",
                                                          force_reload=self.force_reload, verbose=verbose)

    def get_dataset_name(self):
        graphs_path = self.name + '_' + self.mode + '_alt_' + self.alt + '_s_' + str(self.limit) + '.bin'
        info_path = self.name + '_info_' + self.mode + '_alt_' + self.alt + '_s_' + str(self.limit) + '.pkl'
        return graphs_path, info_path

    def load_people_view_graph(self, sample_view, store_heads_jsons=False):
        view_graph = []
        view_heads = {}
        view_heads_num_joints = {}

        self.jsons_for_head = dict()
        total_time = 0

        nodes_camera = []
        head_id = 0
        # print('ccccccccccccccccccccccccccc')
        for camera in sample_view:
            view_heads[camera] = []
            view_heads_num_joints[camera] = []
            for skeleton in json.loads(sample_view[camera][0]):  # sample_view[camera][1] is the timestamp
                # print(camera)
                # print(skeleton)
                # time0 = time.time()
                hgraph = HumanGraphFromView(skeleton, camera, self.alt)
                # time_hgraph = time.time()
                # print("time hgraph", time_hgraph - time0)
                # total_time += (time_hgraph - time0)
                skeleton_graph = graphData(hgraph.src_nodes, hgraph.dst_nodes, hgraph.n_nodes, hgraph.features,
                                           hgraph.edge_types, hgraph.edge_norms)
                view_graph.append(skeleton_graph)
                view_heads[camera].append(head_id)
                view_heads_num_joints[camera].append(hgraph.num_joints)
                if store_heads_jsons:
                    self.jsons_for_head[head_id] = skeleton
                head_id += skeleton_graph.n_nodes
                nodes_camera += [camera]*skeleton_graph.n_nodes
        view_graph = self.merge_graphs(view_graph)
        n_nodes = head_id
        # print("total", total_time)
        # print('------------------------')
        return view_graph, view_heads, view_heads_num_joints, head_id, nodes_camera

    def merge_graphs(self, graph_list):
        merged_src_nodes = []
        merged_dst_nodes = []
        merged_n_nodes = 0
        merged_features = th.Tensor()
        merged_edge_types = []
        merged_edge_norms = []

        for g in graph_list:
            merged_src_nodes += [n + merged_n_nodes for n in g.src_nodes]
            merged_dst_nodes += [n + merged_n_nodes for n in g.dst_nodes]
            merged_features = th.cat([merged_features, g.features], dim=0)
            merged_edge_types += g.edge_types
            merged_edge_norms += g.edge_norms
            merged_n_nodes += g.n_nodes

        mergedG = graphData(merged_src_nodes, merged_dst_nodes, merged_n_nodes, merged_features,
                            merged_edge_types, merged_edge_norms)
        return mergedG

    def add_edge_node_to_graph(self, G, features_list, relations, id_node, head1, head2):
        n_features = len(features_list)
        new_features = th.zeros([1, n_features])
        new_features[0, features_list.index('edge_node')] = 1
        new_features = th.cat([G.features, new_features], dim=0)
        G.src_nodes.append(head1)
        G.dst_nodes.append(id_node)
        G.edge_types.append(relations.index('link'))
        G.edge_norms.append([1.])
        G.src_nodes.append(id_node)
        G.dst_nodes.append(head1)
        G.edge_types.append(relations.index('link'))
        G.edge_norms.append([1.])
        G.src_nodes.append(head2)
        G.dst_nodes.append(id_node)
        G.edge_types.append(relations.index('link'))
        G.edge_norms.append([1.])
        G.src_nodes.append(id_node)
        G.dst_nodes.append(head2)
        G.edge_types.append(relations.index('link'))
        G.edge_norms.append([1.])
        G.src_nodes.append(id_node)
        G.dst_nodes.append(id_node)
        G.edge_types.append(relations.index('link_link'))
        G.edge_norms.append([1.])

        n_nodes = G.n_nodes + 1
        G = G._replace(n_nodes=n_nodes, features=new_features)

        return G

    # #################################################################
    # Implementation of abstract methods
    #################################################################

    def download(self):
        # No need to download any data
        pass

    def process(self):
        if self.mode != 'test':
            self.process_training()
        else:
            self.process_test()

    def process_training(self):
        freqs = [0 for _ in range(16)]
        def sample_and_remove(limit):
            for _ in range(limit):
                lists_are_empty = True
                for l in self.inputs:
                    if len(l) > 0:
                        lists_are_empty = False
                        break
                if lists_are_empty:
                    break

                views_to_add = []
                for index, probability in enumerate(self.probabilities):
                    if random.random() < probability:
                        try:
                            json_index = self.inputs_indices[index].pop()
                            views_to_add.append(self.inputs[index][json_index])
                        except IndexError:
                            return
                if len(views_to_add) == 0:
                    continue
                freqs[len(views_to_add)] += 1
                yield views_to_add

        idx = 0
        for multi_person in sample_and_remove(self.limit):  # EACH TUPLE OF SAMPLE SHOULD HAVE ONE SINGLE PERSON PLUS SPURIOUS
            if idx % 1000 == 0:
                print(idx)
            idx += 1
            G = []  # This is an empty graph

            #
            # CREATE MAIN NODES
            #

            # List containing lists of the heads of every person
            people = []  # People will hold a list of lists where the inner lists are the heads of a person
            # List containing lists of the spurious heads
            spurious_heads = []
            # number of nodes in the final graph
            total_nodes = 0
            # Populate lists

            nodes_camera = []
            for sample_view in multi_person:  # FOR EACH PERSON IN THE (PLUS SPURIOUS)
                person_heads = []
                # In the next line, `sample_view` would be a _natural_ sample
                view_graph, view_heads, view_num_joints, n_nodes, cur_nodes_camera = self.load_people_view_graph(sample_view)

                nodes_camera += cur_nodes_camera
                # print('sample_view')
                # print(sample_view)

                # `view_heads` is a dictionary: e.g.  view_heads['cam1'], view_heads['cam2']
                for cam_idx in sample_view:  # This loop is to get the real from the spurious heads
                    heads_cam = view_heads[cam_idx]
                    joints_cam = view_num_joints[cam_idx]
                    if len(joints_cam) > 0:
                        good_one_idx = max(enumerate(joints_cam), key=lambda x: x[1])[0]
                        spurious_heads += [(x + total_nodes, cam_idx) for x in heads_cam if
                                           x != heads_cam[good_one_idx]]
                        person_heads.append((heads_cam[good_one_idx] + total_nodes, cam_idx))

                # print('view_heads')
                # print(view_heads)

                # print('view_num_joints')
                # print(view_num_joints)

                # print('person_heads')
                # print(person_heads)

                people.append(person_heads)
                G.append(view_graph)
                total_nodes += n_nodes

            G = self.merge_graphs(G)

            edge_nodes_indices = []
            id_node = total_nodes
            labels = []
            features_list = HumanGraphFromView.get_all_features(self.alt)
            relations = HumanGraphFromView.get_rels(self.alt)

            #
            # CREATE ADDITIONAL NODES REPRESENTING RELATIONS "edge-nodes"
            #
            # For every person
            for ip, person in enumerate(people):
                # Add TRUE links to their other heads
                for head1, cam1 in person:
                    for head2, cam2 in person:
                        if cam1 == cam2:
                            continue
                        G = self.add_edge_node_to_graph(G, features_list, relations, id_node, head1, head2)
                        labels = np.append(labels, [1.])
                        edge_nodes_indices.append(id_node)
                        nodes_camera += ['']
                        id_node += 1

                # Add FALSE links to the heads of others
                for io, other in enumerate(people):
                    if io == ip:
                        continue
                    for head1, cam1 in person:
                        for head2, cam2 in other:
                            if cam1 == cam2:
                                continue
                            G = self.add_edge_node_to_graph(G, features_list, relations, id_node, head1, head2)
                            labels = np.append(labels, [0.])
                            edge_nodes_indices.append(id_node)
                            nodes_camera += ['']
                            id_node += 1

                # Add FALSE links to spurious heads
                for head1, cam1 in person:
                    for head2, cam2 in spurious_heads:
                        if cam1 == cam2:
                            continue
                        G = self.add_edge_node_to_graph(G, features_list, relations, id_node, head1, head2)
                        labels = np.append(labels, [0.])
                        edge_nodes_indices.append(id_node)
                        nodes_camera += ['']
                        id_node += 1

            for head1, cam1 in spurious_heads:
                for head2, cam2 in spurious_heads:
                    if cam1 == cam2:
                        continue
                    G = self.add_edge_node_to_graph(G, features_list, relations, id_node, head1, head2)
                    labels = np.append(labels, [0.])
                    edge_nodes_indices.append(id_node)
                    nodes_camera += ['']
                    id_node += 1

            if edge_nodes_indices:
                final_graph = dgl.graph((G.src_nodes, G.dst_nodes), num_nodes=G.n_nodes, idtype=th.int32)
                final_graph.ndata['h'] = G.features
                final_graph.edata.update({'rel_type': th.LongTensor(G.edge_types),
                                          'norm': th.Tensor(G.edge_norms)})  # , 'he': hgraph.edge_feats})

                # Append final data
                self.graphs.append(dgl.add_self_loop(final_graph))
                self.labels.append(th.tensor(labels, dtype=th.float64).unsqueeze(1))
                self.data['edge_nodes_indices'].append(th.tensor(edge_nodes_indices, dtype=th.int64).unsqueeze(1))
                self.data['nodes_camera'].append(nodes_camera)
                

        print(freqs)
    def process_test(self):
        assert len(self.inputs) == 1, "For testing, please provide __ONE__ single JSON file"

        idx = 0
        if type(self.inputs[0]) == list:
            iterate_over = self.inputs[0]
        else:
            iterate_over = self.inputs

        for json_view in iterate_over: # FOR EACH OF THE TUPLE OF SAMPLES (EACH ONE SHOULD HAVE ONE SINGLE PERSON PLUS SPURIOUS)
            if idx % 1000 == 0 and idx > 0:
                print(idx)
            if idx == self.limit:
                break
            idx += 1
            G = []  # This is an empty graph


            # print(json_view)

            #
            # CREATE MAIN NODES
            #
            # number of nodes in the final graph
            # time0 = time.time()
            total_nodes = 0
            # In the next line `sample_view` would be a _natural_ sample
            view_graph, view_heads, _, n_nodes, nodes_camera = self.load_people_view_graph(json_view, store_heads_jsons=True)
            # `view_heads` is a dictionary: e.g.  view_heads['cam1'], view_heads['cam2']
            G = view_graph
            total_nodes += n_nodes

            # time_separated_graphs = time.time()
            # print("time generating people graphs", time_separated_graphs-time0)

            edge_nodes_indices = []
            id_node = total_nodes
            labels = []
            features_list = HumanGraphFromView.get_all_features(self.alt)
            relations = HumanGraphFromView.get_rels(self.alt)

            #
            # CREATE ADDITIONAL NODES REPRESENTING RELATIONS "edge-nodes"
            #
            # For every person
            # print('view heads', view_heads)
            for h_index, (cam1, heads1) in enumerate(view_heads.items()):
                for (cam2, heads2) in list(view_heads.items())[h_index + 1:]:
                    if cam1 == cam2:
                        continue
                    for head1 in heads1:
                        for head2 in heads2:
                            G = self.add_edge_node_to_graph(G, features_list, relations, id_node, head1, head2)
                            labels = np.append(labels, [0.])
                            edge_nodes_indices.append(id_node)
                            nodes_camera += ['']
                            id_node += 1

            # time_edge_nodes = time.time()
            # print("time generating edge_nodes", time_edge_nodes - time_separated_graphs)

            if edge_nodes_indices:
                final_graph = dgl.graph((G.src_nodes, G.dst_nodes), num_nodes=G.n_nodes, idtype=th.int32)
                final_graph.ndata['h'] = G.features
                final_graph.edata.update({'rel_type': th.LongTensor(G.edge_types),
                                          'norm': th.Tensor(G.edge_norms)})  # , 'he': hgraph.edge_feats})

                # Append final data
                self.graphs.append(dgl.add_self_loop(final_graph))
                self.labels.append(th.tensor(labels, dtype=th.float64).unsqueeze(1))
                self.data['edge_nodes_indices'].append(th.tensor(edge_nodes_indices, dtype=th.int64).unsqueeze(1))
                self.data['nodes_camera'].append(nodes_camera)
            # time_dgl_graphs = time.time()
            # print("time generating dgl graphs", time_dgl_graphs-time_edge_nodes)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx], self.data['edge_nodes_indices'][idx]

    def __len__(self):
        return len(self.graphs)

    def save(self):
        if self.debug or self.mode == "test":
            return
        # Generate paths
        graphs_path, info_path = tuple((self.path_save + x) for x in self.get_dataset_name())
        print(graphs_path, info_path)
        os.makedirs(os.path.dirname(self.path_save), exist_ok=True)

        # Save graphs
        save_graphs(graphs_path, self.graphs)

        # Save additional info
        save_info(info_path, {'edge_nodes_indices': self.data['edge_nodes_indices'], 'labels': self.labels})

    def load(self):
        # Generate paths
        graphs_path, info_path = tuple((self.path_save + x) for x in self.get_dataset_name())
        # Load graphs
        self.graphs, _ = load_graphs(graphs_path)
        # Load info
        info_dict = load_info(info_path)
        self.data['edge_nodes_indices'] = info_dict['edge_nodes_indices']
        self.labels = info_dict['labels']
        print(f'LOADED {len(self.graphs)} graphs')

    def has_cache(self):
        # Generate paths
        graphs_path, info_path = tuple((self.path_save + x) for x in self.get_dataset_name())
        if self.debug:
            return False
        return os.path.exists(graphs_path) and os.path.exists(info_path)
