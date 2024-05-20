import time
from tkinter import Y

epochs = 10000
lr = 1e-4
batch_size = 2096
patience = 20 
optimise_matrices = False

WHOLE_DATASET_IN_GPU = False

import sys
import torch
import pickle
import cv2
import os

import numpy as np

from torch import nn
from torch._C import dtype
from torch.autograd import Variable
from torch.utils import data

sys.path.append('../utils')
from pose_estimator_utils import camera_matrix, get_distortion_coefficients, from_homogeneous, from_homogeneous2, apply_distortion
from pose_estimator_dataset_from_json import PoseEstimatorDataset
from mlp import PoseEstimatorMLP

from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager


import math
import signal
import argparse

parser = argparse.ArgumentParser(description='3D skeleton prediction training for 3D multi-human pose estimation')

parser.add_argument('--trainset', type=str, nargs='+', required=True, help='List of json files composing the training set')
parser.add_argument('--devset', type=str, nargs='+', required=True, help='List of json files composing the development set')

args = parser.parse_args()


print(torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using CUDA')
else:
    device = torch.device('cpu')
    print('Using CPU')


TRAIN_FILES = args.trainset
DEV_FILES = args.devset

print(f'Using {TRAIN_FILES} for training')
print(f'Using {DEV_FILES} for dev')

sys.path.append('../')
from parameters import parameters 
joint_list = parameters.joint_list
numbers_per_joint = parameters.numbers_per_joint
number_of_cameras = len(parameters.used_cameras)
print(f'number of cameras {number_of_cameras}')

def compute_error(parameters, joints, raw_inputs, orig_inputs, outputs, batch_size, camera_d_transforms, camera_matrices, distortion_coefficients):
    ones = torch.ones(1, batch_size, device=device)  # useful to convert to homogeneous coordinates
    error2D = torch.zeros(batch_size, device=device)  # we'll add up the 2D error for the batch in this variable

    for joint_idx in range(len(joints)):
        results_3d = torch.cat((torch.transpose(outputs[:, joint_idx * 3:joint_idx * 3 + 3]*10., 0, 1).to(device), ones),
                               0).to(device)
        # For every camera
        for cam_idx, camera in enumerate(parameters.cameras):
            TR = camera_d_transforms[cam_idx]  # world to camera transformation matrix
            from_camera_3D = torch.matmul(TR, results_3d)[:-1][:]
            from_camera = from_homogeneous2(from_camera_3D)
            from_camera_with_distorion = apply_distortion(distortion_coefficients[cam_idx], from_camera)
            C = camera_matrices[cam_idx]  # camera matrix
            in_camera = torch.matmul(C, from_camera_with_distorion)
            backprojections = torch.transpose(from_homogeneous(in_camera), 0, 1)
            # Extract the columns for the joints' coordinates in image for camera `cam_idx` (2D)
            begin = len(joints) * parameters.numbers_per_joint_for_loss * cam_idx + joint_idx * parameters.numbers_per_joint_for_loss + 1
            end = begin + 2
            coords = orig_inputs[:, begin:end]
            # Compute error
            error_from_one_cam = torch.sum(torch.abs(coords - backprojections), 1) #+ err2
            # Extract the columns for the boolean and filter the results
            begin = len(joints) * parameters.numbers_per_joint_for_loss * cam_idx + joint_idx * parameters.numbers_per_joint_for_loss
            end = begin + 1
            FILTER = orig_inputs[:, begin:end] < 0.5

            # Ignore error if we do not have the joint!
            error_from_one_cam[FILTER.squeeze()] = 0
            # Add up the error
            error2D += error_from_one_cam


    return error2D


def pause():
    k = cv2.waitKey(10)
    while k != 32:
        k = cv2.waitKey(10)


def signal_handler(sig, frame):
    global stop_training
    global ctrl_c_counter
    ctrl_c_counter += 1
    if ctrl_c_counter >= 3:
        stop_training = True
    print('You have to press Ctr+c 3 times to stop the training ({} times)'.format(ctrl_c_counter))


if __name__ == '__main__':
    global stop_training
    global ctrl_c_counter

    stop_training = False
    ctrl_c_counter = 0

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Set random number seed
    torch.manual_seed(58008)

    print(f'Final number of dimensions per joint {parameters.numbers_per_joint}')
    print(f'Final number of joints {len(joint_list)}')
    print(f'Final number of cameras {number_of_cameras}')

    # Get initial calibration matrices. This is stored in a pickled TransformManager.
    tm = pickle.load(open(parameters.transformations_path, 'rb'))
    camera_i_transforms = []
    camera_d_transforms = []
    camera_matrices = []
    distortion_coefficients = []

    for cam_idx, cam in enumerate(parameters.cameras):
        # Add the direct transform (root to camera) to the list
        trfm =   tm.get_transform("root", parameters.camera_names[cam_idx])
        trfm_i = tm.get_transform(parameters.camera_names[cam_idx], "root")
        camera_d_transforms.append(
                Variable(torch.from_numpy(trfm).type(torch.float32).to(device), requires_grad=optimise_matrices))
        # Add the inverse transform (camera to root) to the list
        camera_i_transforms.append(Variable(torch.from_numpy(trfm_i).type(torch.float32).to(device), requires_grad=optimise_matrices))
        # Add the camera matrix to the list
        camera_matrices.append(Variable(camera_matrix(cam), requires_grad=optimise_matrices))
        distortion_coefficients.append(Variable(get_distortion_coefficients(cam), requires_grad=optimise_matrices))

    # Instantiate the MLP
    in_dimensions = number_of_cameras*len(joint_list)*numbers_per_joint
    print(f'in_dim  {in_dimensions}')
    mlp = PoseEstimatorMLP(input_dimensions=in_dimensions, output_dimensions=len(joint_list)*3).to(device)

    # Load the dataset.
    print("Loading datasets")
    if WHOLE_DATASET_IN_GPU is False:
        data_device = 'cpu'
    else:
        data_device = device
    train_dataset = PoseEstimatorDataset(TRAIN_FILES, parameters.cameras, joint_list, data_augmentation=True, reload=True, save=True)
    valid_dataset = PoseEstimatorDataset(DEV_FILES, parameters.cameras, joint_list, data_augmentation=True, reload=True, save=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    print(f'dataset length: {len(train_dataset)}')

    # Define loss function and optimizer
    loss_function = nn.MSELoss()
    parameters_to_optimise = [x for x in mlp.parameters()]
    if optimise_matrices:
        parameters_to_optimise += camera_i_transforms+camera_d_transforms + camera_matrices
    optimizer = torch.optim.Adam(parameters_to_optimise, lr=lr)

    # Define parameters for the training loop and early stop
    cur_step = 0
    best_loss = -1
    min_train_loss = float("inf")
    min_dev_loss = float("inf")

    # Run the training loop
    print(f'Epochs {epochs}')
    for epoch in range(0, epochs):
        if stop_training:
            break

        # Print epoch and initialise match loss
        mlp.train()
        batch_loss = 0.0
        
        # Iterate over the DataLoader for training data
        for mini_batch, data_inputs in enumerate(train_dataloader, 0):
            raw_inputs = data_inputs[0].to(device)
            orig_inputs = data_inputs[1].to(device)
            # Zero the gradients
            optimizer.zero_grad()
            # Set auxiliary variables
            this_batch_size = raw_inputs.size()[0]

            #
            # Compute output (forward pass)
            #
            outputs = mlp(raw_inputs.to(device))

            #
            # Compute back projections and add up the error
            #
            error = compute_error(parameters, joint_list, raw_inputs, orig_inputs, outputs, this_batch_size,
                                    camera_d_transforms, camera_matrices, distortion_coefficients)

            # Compute loss
            target = torch.zeros(error.size(), device=device)  # We aim for zero error
            loss = loss_function(error, target)

            # Perform backward pass
            loss.backward()
            # Clip the gradients to avoid NaNs
            torch.nn.utils.clip_grad_norm(parameters=mlp.parameters(), max_norm=10, norm_type=2.0)
            # Perform optimization
            optimizer.step()
            # Set current loss
            batch_loss += loss.item()*this_batch_size

        loss_data = batch_loss / len(train_dataset)
        mae_per_coord = math.sqrt(loss_data) / len(parameters.cameras) / len(joint_list) / 2
        print(f'loss: {loss_data:.5f} error per coor: {mae_per_coord:.5f}')

        if loss_data < min_train_loss:
            min_train_loss = loss_data

        if epoch % 5 == 0:
            print("Epoch {:05d} | MAE/coord {:.6f} | Loss: {:.6f} | Patience: {} | ".format(epoch, mae_per_coord, loss_data, cur_step), end='')
            valid_batch_loss = 0.0
            for batch, valid_data in enumerate(valid_dataloader):
                with torch.no_grad():
                    raw_inputs = valid_data[0].to(device)
                    orig_inputs = valid_data[1].to(device)
                    mlp.eval()

                    this_batch_size = raw_inputs.size()[0]

                    outputs = mlp(raw_inputs.to(device))

                    error = compute_error(parameters, joint_list, raw_inputs, orig_inputs, outputs, this_batch_size,
                                            camera_d_transforms, camera_matrices, distortion_coefficients)

                    # Compute loss
                    target = torch.zeros(error.size(), device=device)  # We aim for zero error
                    loss = loss_function(error, target)
                    valid_batch_loss += loss.item() * this_batch_size

            val_loss_data = valid_batch_loss / len(valid_dataset)
            val_mae_per_coord = math.sqrt(val_loss_data) / len(parameters.cameras) / len(joint_list) / 2

            mean_val_loss = val_loss_data
            print(" val_MEAN: {:.6f} val_BEST: {:.6f} | val_MAE/coord {:.6f}".format(mean_val_loss, best_loss, val_mae_per_coord))

            # Early stop
            if best_loss > mean_val_loss or best_loss < 0:
                print('Saving...')
                best_loss = mean_val_loss
                if best_loss < min_dev_loss:
                    min_dev_loss = best_loss
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': mlp.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'average_training_loss': loss_data,
                        'average_validation_loss': val_loss_data,
                        'average_training_error_per_coord': mae_per_coord,
                        'average_validation_error_per_coord': val_mae_per_coord,
                        }, f'../pose_estimator.pytorch')
                cur_step = 0
            else:
                cur_step += 1
                if cur_step >= patience:
                    break
