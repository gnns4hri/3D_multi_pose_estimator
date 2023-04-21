import sys
import torch
import pickle
import json
import math
import numpy as np
import copy
import argparse
from tqdm import tqdm

sys.path.append('../skeleton_matching')
from gat2 import GAT2 as GAT
from graph_generator import MergedMultipleHumansDataset, HumanGraphFromView
sys.path.append('../utils')
from pose_estimator_utils import camera_matrix, get_distortion_coefficients, from_homogeneous, from_homogeneous2, triangulate


sys.path.append('../utils')
from pose_estimator_dataset_from_json import PoseEstimatorDataset
from mlp import PoseEstimatorMLP
from skeleton_matching_utils import get_person_proposal_from_network_output


sys.path.append('../')
from parameters import parameters 

parser = argparse.ArgumentParser(description='Compute the reprojection error, for each camera, of the estimated 3D, triangulated 3D and, optionally, ground truth 3D')

parser.add_argument('--testfiles', type=str, nargs='+', required=True, help='List of json files used as input')
parser.add_argument('--showgt', action='store_true', help='Show ground truth reprojection error')
parser.add_argument('--tmdir', type=str, nargs=1, help='Directory that contains the files with the transfomation matrices of the ground truth')
parser.add_argument('--modelsdir', type=str, nargs='?', required=False, default='../models/', help='Directory that contains the models\' files')
parser.add_argument('--datastep', type=int, nargs='?', required=False, default=12, help='Data step used to compute the reprojection error')


args = parser.parse_args()

if args.showgt and args.tmdir is None:
    parser.error("--showgt requires --tmdir")

TEST_FILES = args.testfiles

if args.tmdir is not None:
    tm_dir = args.tmdir[0]
    if tm_dir[-1] != '/':
        tm_dir += '/'

MODELSDIR = args.modelsdir
if MODELSDIR[-1] != '/':
    MODELSDIR += '/'

WITH_GROUND_TRUTH = args.showgt


num_features = len(HumanGraphFromView.get_all_features())

if torch.cuda.is_available() is True:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


torch.set_grad_enabled(False)

tm = pickle.load(open(parameters.transformations_path, 'rb'))
camera_i_transforms = {}
camera_d_transforms = {}
camera_matrices = {}
distortion_coefficients = {}
camera_matrices_np = {}
distortion_coefficients_np = {}

projection_matrices = {}
for cam_idx, cam in enumerate(parameters.camera_names):
    # Add the direct transform (root to camera) to the list
    trfm =   tm.get_transform("root", cam)
    trfm_i = tm.get_transform(cam, "root")
    camera_d_transforms[cam] = torch.from_numpy(trfm).type(torch.float32)

    camera_i_transforms[cam] = torch.from_numpy(trfm_i).type(torch.float32)

    camera_matrices[cam] = camera_matrix(cam_idx)
    camera_matrices_np[cam] = camera_matrix(cam_idx).cpu().detach().numpy()
    distortion_coefficients[cam] = get_distortion_coefficients(cam_idx).to('cpu')
    distortion_coefficients_np[cam] = np.array([parameters.kd0[cam_idx], parameters.kd1[cam_idx], parameters.p1[cam_idx], parameters.p2[cam_idx], parameters.kd2[cam_idx]])
    projection_matrices[cam] = trfm[0:3, :]


def get_projected_coordinates(p3D, camera):
    ones = torch.ones(1)

    # Project skeleton into view
    TR = camera_d_transforms[camera]  # world to camera transformation matrix

    from_camera_3D = torch.matmul(TR, torch.cat((torch.tensor(p3D), ones), 0))[:-1]
    from_camera = from_homogeneous2(from_camera_3D)

    from_camera_with_distorion = from_camera.clone()
    kd = distortion_coefficients[camera]
    r = torch.norm(from_camera[:-1], dim=0)
    r = r*r
    from_camera_with_distorion[0] = from_camera[0]*(1 + kd[0]*r + kd[1]*r*r + kd[2]*r*r*r)
    from_camera_with_distorion[1] = from_camera[1]*(1 + kd[0]*r + kd[1]*r*r + kd[2]*r*r*r)

    C = camera_matrices[camera].to('cpu')  # camera matrix
    pt = from_homogeneous(torch.matmul(C, from_camera_with_distorion))
    return pt.cpu().detach().numpy()


with open("../human_pose.json", 'r') as f:
    human_pose = json.load(f)
    skeleton = human_pose["skeleton"]
    keypoints = human_pose["keypoints"]

CLASSIFICATION_THRESHOLD = 0.5


########################################

# METRICS

n_data = 0
time_3D_person = 0.
rep_error_est = {}
n_est = {}
rep_error_gt = {}
n_gt = {}
rep_error_triang = {}
n_triang = {}
for k in parameters.camera_names:
    rep_error_est[k] = []
    n_est[k] = 0
    rep_error_gt[k] = []
    n_gt[k] = 0
    rep_error_triang[k] = []
    n_triang[k] = 0
    
DATASTEP = args.datastep

#######################################

numbers_per_joint = parameters.numbers_per_joint
mlp = PoseEstimatorMLP(input_dimensions=len(parameters.used_cameras)*len(parameters.joint_list)*numbers_per_joint, output_dimensions=54)
saved = torch.load(MODELSDIR+'pose_estimator.pytorch', map_location=device)
mlp.load_state_dict(saved['model_state_dict'])

params = pickle.load(open(MODELSDIR+'skeleton_matching.prms', 'rb'))
model = GAT(None, params['gnn_layers'], params['num_feats'], params['n_classes'], params['num_hidden'], params['heads'],
        params['nonlinearity'], params['final_activation'], params['in_drop'], params['attn_drop'], params['alpha'], params['residual'], bias=True)
model.load_state_dict(torch.load(MODELSDIR+'skeleton_matching.tch', map_location=device))
model = model.to(device)


total_data = 0

n_input = 0

n_sample = 0

for file in TEST_FILES:
    print(file)
    if WITH_GROUND_TRUTH:
        dataset_name = file.split('/')[-1]
        tm_file = tm_dir + 'tm_' + dataset_name.split('_')[0] + '_' + dataset_name.split('_')[1] + '.pickle'
        tm_dataset = pickle.load(open(tm_file, 'rb'))
        dataset_camera_d_transforms = []
        for cam_idx, cam in enumerate(parameters.camera_names):
            trfm_dataset = tm_dataset.get_transform("root", parameters.camera_names[cam_idx])
            dataset_camera_d_transforms.append(torch.from_numpy(trfm_dataset).type(torch.float32))

    input_data = json.load(open(file, 'rb'))

    total_data = len(input_data)

    for input_element in tqdm(input_data):

        n_input += 1

        if (n_input - 1) % DATASTEP == 0:

            if WITH_GROUND_TRUTH:
                # LOAD GROUND TRUTH
                first_cam = list(input_element.keys())[0]
                if len(input_element[first_cam]) != 4:
                    print("There is no ground truth in the specified file")
                    exit()
            

                for c in input_element:
                    if len(input_element[c][3]) >  len(input_element[first_cam][3]):
                        first_cam = c

                if len(input_element[first_cam][3]) == 0:
                    continue

                joints_3D_all = input_element[first_cam][3]

                GT_3D = []

                valid_GT = True
                for joints_3D in joints_3D_all:

                    if not '-1' in joints_3D.keys():
                        valid_GT = False
                
                    GT_3D_torch = torch.zeros(len(parameters.joint_list)*3)

                    for j in parameters.joint_list:
                        idx = str(j)
                        if idx in joints_3D:
                            GT_3D_torch[j*3: j*3 + 3] = torch.tensor(np.array(joints_3D[idx])/100.)

                    GT_3D_torch = GT_3D_torch.reshape((len(parameters.joint_list), 3)).transpose(0,1)

                    ones = torch.ones(1, GT_3D_torch.shape[1])

                    TR_dataset = dataset_camera_d_transforms[1]
                    TRi = camera_i_transforms[parameters.camera_names[1]]
                    from_camera = torch.matmul(TR_dataset, torch.cat((GT_3D_torch, ones), 0))
                    to_world = torch.transpose(torch.matmul(TRi, from_camera)[:-1][:], 0, 1)

                    GT_3D_person = {}
                    for j in parameters.joint_list:
                        idx = str(j)
                        if idx in joints_3D:
                            GT_3D_person[idx] = to_world[j].numpy()
                        else:
                            valid_GT = False

                    GT_3D.append(copy.deepcopy(GT_3D_person))

                if not valid_GT:
                    continue

            n_sample += 1
            # INIT OF GRAPH MATCHING

            processed_input = dict()
            for cam in input_element:
                data = json.loads(input_element[cam][0])
                cam_data = []
                for s in data:
                    cam_data.append(s)
                if cam_data:
                    processed_input[cam] = []
                    processed_input[cam].append(json.dumps(cam_data))
                    processed_input[cam].append(input_element[cam][1])

            scenario = MergedMultipleHumansDataset(processed_input, mode='test', limit=10000, debug=True, alt=parameters.graph_alternative, verbose=False)

            if len(scenario.graphs)==0:
                continue

            n_data += 1

            try:
                subgraph = scenario.graphs[0].to(device)
                indices = scenario.data['edge_nodes_indices'][0].to(device)
                nodes_camera = scenario.data['nodes_camera'][0]
                feats = subgraph.ndata['h'].to(device)

                model.g = subgraph
                for layer in model.layers:
                    layer.g = subgraph
                outputs = torch.squeeze(model(feats.float(), subgraph))

                indices = torch.squeeze(indices).to('cpu')
            except:
                continue

            # Process the output graph as it comes from the GNN
            final_output = get_person_proposal_from_network_output(outputs, subgraph, indices, nodes_camera, scenario.jsons_for_head, CLASSIFICATION_THRESHOLD)

            final_results = list()
            triang_3D = list()
            coords_2D = list()
            for person in final_output:

                raw_input = dict()
                valid_input = False
                for camera in parameters.camera_names:
                    if camera in person.keys() and person[camera] is not None:
                        pc = person[camera]
                        all_joints_data = [scenario.jsons_for_head[pc]]
                        raw_input[camera] = [json.dumps(all_joints_data)]
                        if camera in parameters.used_cameras:
                            valid_input = True

                points_2D = dict()
                for cam in raw_input.keys():
                    if cam in parameters.camera_names:
                        joints_json = raw_input[cam][0]
                        joints = json.loads(joints_json)
                        if not joints:
                            continue
                        for j, pos in joints[0].items():
                            if pos[0] > 0.:
                                if not j in points_2D.keys():
                                    points_2D[j] = dict()
                                points_2D[j][cam] = np.array([pos[1], pos[2]])

                triang_3D_person = triangulate(points_2D, camera_matrices_np, distortion_coefficients_np, projection_matrices, parameters.axes_3D['Y'][0])                    

                triang_3D.append(triang_3D_person)

                coords_2D.append(raw_input)

                if not valid_input:
                    continue
                inputs = PoseEstimatorDataset(raw_input, parameters.cameras, parameters.joint_list, save=False)
                inputs = inputs[0][0].reshape([1, inputs[0][0].size()[0]])

                outputs = mlp(inputs)

                results_3d = torch.squeeze(outputs)*10.

                x3D = results_3d[::3]
                y3D = results_3d[1::3]
                z3D = results_3d[2::3]

                person_result = list()

                for idx_joint in range(len(parameters.joint_list)):

                    person_result.append(np.array([x3D[idx_joint], y3D[idx_joint], z3D[idx_joint]]))
                final_results.append(person_result)

    ##########################

                # Get GT index of person
                if WITH_GROUND_TRUTH:
                    min_err = 10000000000.
                    GT_person = -1
                    for iGT in range(len(GT_3D)):
                        n_joints = 0
                        mean_error = 0
                        for j, gt3D in GT_3D[iGT].items():
                            idx = int(j)
                            if idx in parameters.used_joints:
                                p3D = person_result[idx]
                                err = np.linalg.norm(p3D - gt3D)
                                mean_error += err
                                n_joints += 1

                        if n_joints > 0:
                            mean_error = mean_error/n_joints
                            if min_err > mean_error:
                                GT_person = iGT
                                min_err = mean_error

                projected_est = dict()
                projected_gt = dict()
                projected_triang = dict()
                for k in raw_input:
                    image_coords = json.loads(raw_input[k][0])[0]
                    projected_est[k] = dict()
                    est_err = 0
                    n_common = 0
                    for i_joint, p3D in enumerate(person_result):
                        if not i_joint in parameters.used_joints:
                            continue
                        p2D = get_projected_coordinates(p3D, k)
                        projected_est[k][str(i_joint)] = p2D
                        if str(i_joint) in image_coords:
                            if image_coords[str(i_joint)][3] > 0.5:
                                rep_err = math.sqrt((p2D[0]-image_coords[str(i_joint)][1])**2 + (p2D[1]-image_coords[str(i_joint)][2])**2)
                                est_err += rep_err
                                rep_error_est[k].append(rep_err)
                            n_common += 1

                    n_est[k] += n_common
                    if n_common > 0:
                        est_err = est_err / n_common
                    else:
                        est_err = -1

                    if WITH_GROUND_TRUTH:
                        projected_gt[k] = dict()
                        gt_err = 0
                        n_common = 0
                        if GT_person >= 0:
                            for i_joint, p3D in GT_3D[GT_person].items():
                                p2D = get_projected_coordinates(p3D, k)
                                projected_gt[k][i_joint] = p2D
                            if i_joint in image_coords:
                                if image_coords[str(i_joint)][3] > 0.5:
                                    rep_err = math.sqrt((p2D[0]-image_coords[i_joint][1])**2 + (p2D[1]-image_coords[i_joint][2])**2)
                                    gt_err += rep_err
                                    rep_error_gt[k].append(rep_err)      
                                n_common += 1

                            n_gt[k] += n_common
                            if n_common > 0:
                                gt_err = gt_err / n_common
                            else:
                                gt_err = -1
                        else:
                            gt_err = -1


                    projected_triang[k] = dict()
                    triang_err = 0
                    n_common = 0
                    for i_joint, p3D in triang_3D_person.items():
                        p3D = p3D.astype(np.float32)
                        p2D = get_projected_coordinates(np.transpose(p3D)[0], k)
                        projected_triang[k][i_joint] = p2D
                        if i_joint in image_coords:
                            if image_coords[str(i_joint)][3] > 0.5:
                                rep_err = math.sqrt((p2D[0]-image_coords[i_joint][1])**2 + (p2D[1]-image_coords[i_joint][2])**2)
                                triang_err += rep_err
                                rep_error_triang[k].append(rep_err)
                            n_common += 1

                    n_triang[k] += n_common
                    if n_common > 0:
                        triang_err = triang_err / n_common
                    else:
                        triang_err = -1


print('**********************  REPROJECTION ERRORS (mean and median) **********************')
for k in parameters.camera_names:
    print('------------------','CAMERA', k, '------------------')
    if n_est[k]>0.:
        print('est', np.mean(np.array(rep_error_est[k])), np.median(np.array(rep_error_est[k])))
    if n_gt[k]>0.:
        print('GT', np.mean(np.array(rep_error_gt[k])), np.median(np.array(rep_error_gt[k])))
    if n_triang[k]>0.:
        print('triang', np.mean(np.array(rep_error_triang[k])), np.median(np.array(rep_error_triang[k])))                        
                                
