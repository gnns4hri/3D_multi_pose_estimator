import time
import sys
import torch
import pickle
import json
import numpy as np
from tqdm import tqdm
import itertools
import copy
import argparse

sys.path.append('../skeleton_matching')
from gat2 import GAT2 as GAT
from graph_generator import MergedMultipleHumansDataset, HumanGraphFromView


sys.path.append('../utils')
from pose_estimator_dataset_from_json import PoseEstimatorDataset
from mlp import PoseEstimatorMLP
from skeleton_matching_utils import get_person_proposal_from_network_output


sys.path.append('../')
from parameters import parameters 


parser = argparse.ArgumentParser(description='Print accuracy and time metrics of the skeleton-matching and pose estimation models (CMU Panoptic only)')

parser.add_argument('--testfiles', type=str, nargs='+', required=True, help='List of json files used as input')
parser.add_argument('--tmdir', type=str, nargs=1,required=True, help='Directory that contains the files with the transfomation matrices')
parser.add_argument('--modelsdir', type=str, nargs='?', required=False, default='../models/', help='Directory that contains the models\' files')
parser.add_argument('--datastep', type=int, nargs='?', required=False, default=12, help='Data step used to compute the metrics')


args = parser.parse_args()

TEST_FILES = args.testfiles

tm_dir = args.tmdir[0]
if tm_dir[-1] != '/':
    tm_dir += '/'

MODELSDIR = args.modelsdir
if MODELSDIR[-1] != '/':
    MODELSDIR += '/'

num_features = len(HumanGraphFromView.get_all_features())

if torch.cuda.is_available() is True:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

torch.set_grad_enabled(False)

tm = pickle.load(open(parameters.transformations_path, 'rb'))
camera_i_transforms = []
for cam_idx, cam in enumerate(parameters.camera_names):
    # Add the inverse transform (camera to root) to the list
    camera_i_transforms.append(torch.from_numpy(tm.get_transform(parameters.camera_names[cam_idx], "root")).type(torch.float32))


CLASSIFICATION_THRESHOLD = 0.5


########################################

# METRICS

mpjpe_threshold = np.arange(25, 155, 25)
global_acum_err = 0
n_data = 0
correct_poses = [0.]*len(mpjpe_threshold)
TP = []
FP = []
for i in range(len(mpjpe_threshold)):
    TP.append([])
    FP.append([])
n_poses = 0
n_gt = 0
n_matching_poses = 0
time_graph_matching = 0.
time_graph_matching_person = 0.
time_3D = 0.
time_3D_person = 0.
DATASTEP = args.datastep

#######################################

numbers_per_joint = parameters.numbers_per_joint
mlp = PoseEstimatorMLP(input_dimensions=len(parameters.cameras)*len(parameters.joint_list)*numbers_per_joint, output_dimensions=54)
saved = torch.load(MODELSDIR + 'pose_estimator.pytorch', map_location=device)
mlp.load_state_dict(saved['model_state_dict'])
mlp = mlp.to(device)

params = pickle.load(open(MODELSDIR + 'skeleton_matching.prms', 'rb'))
model = GAT(None, params['gnn_layers'], params['num_feats'], params['n_classes'], params['num_hidden'], params['heads'],
        params['nonlinearity'], params['final_activation'], params['in_drop'], params['attn_drop'], params['alpha'], params['residual'], bias=True)
model.load_state_dict(torch.load(MODELSDIR + 'skeleton_matching.tch', map_location=device))
model = model.to(device)



total_data = 0 
n_input = 0

for file in TEST_FILES:
    print(file)
    dataset_name = file.split('/')[-1]
    tm_file = tm_dir + 'tm_' + dataset_name.split('_')[0] + '_' + dataset_name.split('_')[1] + '.pickle'
    tm_dataset = pickle.load(open(tm_file, 'rb'))
    dataset_camera_d_transforms = []
    for cam_idx, cam in enumerate(parameters.camera_names):
        trfm_dataset = tm_dataset.get_transform("root", parameters.camera_names[cam_idx])
        dataset_camera_d_transforms.append(torch.from_numpy(trfm_dataset).type(torch.float32))


    input_data = json.load(open(file, 'rb'))
    total_data += len(input_data)
    for input_element in tqdm(input_data):

        n_input += 1

        if (n_input - 1) % DATASTEP == 0:

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
            valid_GT = []

            for joints_3D in joints_3D_all:
            
                GT_3D_torch = torch.zeros(len(parameters.joint_list)*3)

                for j in parameters.joint_list:
                    idx = str(j)
                    if idx in joints_3D:
                        GT_3D_torch[j*3: j*3 + 3] = torch.tensor(np.array(joints_3D[idx])/100.)

                GT_3D_torch = GT_3D_torch.reshape((len(parameters.joint_list), 3)).transpose(0,1)

                ones = torch.ones(1, GT_3D_torch.shape[1])

                TR_dataset = dataset_camera_d_transforms[1]
                TRi = camera_i_transforms[1]
                from_camera = torch.matmul(TR_dataset, torch.cat((GT_3D_torch, ones), 0))
                to_world = torch.transpose(torch.matmul(TRi, from_camera)[:-1][:], 0, 1)

                GT_3D_person = {}
                for j in parameters.joint_list:
                    idx = str(j)
                    if idx in joints_3D:
                        GT_3D_person[idx] = to_world[j].numpy()


                GT_3D.append(copy.deepcopy(GT_3D_person))
                if '-1' in joints_3D.keys():
                    valid_GT.append(True)
                else:
                    valid_GT.append(False)

            # INIT OF GRAPH MATCHING

            time_ini = time.time()

            # INIT OF GRAPH MATCHING
            if len(parameters.used_cameras) > 1:
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

            else:
                cam = parameters.used_cameras[0]
                joints_json = input_element[cam][0]
                skeletons = json.loads(joints_json)
                final_output = []
                for sk in skeletons:
                    person = dict()
                    person[cam] = list()
                    person[cam].append(json.dumps([sk]))
                    person[cam] += input_element[cam][1:]
                    final_output.append(person)


            time_GM_i = time.time() - time_ini

            if len(final_output) > 0:
                time_graph_matching += time_GM_i
                time_graph_matching_person += time_GM_i / len(final_output)


            time_a = time.time()

            final_results = list()
            person_visible_joints = list()
            batched_input = []
            for person in final_output:

                visible_joints = list()
                if len(parameters.used_cameras)>1:
                    raw_input = dict()
                    for cam_idx, camera in enumerate(parameters.used_cameras):
                        if person[camera] is not None:
                            pc = person[camera]
                            all_joints_data = [scenario.jsons_for_head[pc]]
                            raw_input[camera] = [json.dumps(all_joints_data)]
                            for j, values in all_joints_data[0].items():
                                if values[3] > 0.5:
                                    visible_joints.append(j)
                                        
                else:
                    raw_input = person
                    for cam in raw_input:
                        all_joints_data = json.loads(raw_input[cam][0])
                        for j, values in all_joints_data[0].items():
                            if values[3] > 0.5:
                                visible_joints.append(j)


                inputs = PoseEstimatorDataset(raw_input, parameters.cameras, parameters.joint_list, save=False)
                if inputs.__len__()==0:
                    continue

                person_visible_joints.append(visible_joints)

                inputs = inputs[0][0].reshape([1, inputs[0][0].size()[0]]).to(device)

                batched_input.append(inputs)       

            # GET the 3D skeleton of all the detected persons
            
            if batched_input:
                input_all = torch.cat(batched_input, dim=0)
                output_all = mlp(input_all.to(device))
                for person_id in range(output_all.shape[0]):
                    results_3d = torch.squeeze(output_all[person_id])*10.
                    results_3d = results_3d.to('cpu')

                    x3D = results_3d[::3] 
                    y3D = results_3d[1::3]
                    z3D = results_3d[2::3]

                    person_result = list()

                    for idx_joint in range(len(parameters.joint_list)):

                        person_result.append(np.array([x3D[idx_joint], y3D[idx_joint], z3D[idx_joint]]))
                    final_results.append(person_result)

            time_3D_i = time.time() - time_a

            if len(final_results) > 0:
                time_3D += time_3D_i
                time_3D_person += time_3D_i / len(final_results)

            
            err_table = np.zeros((len(GT_3D), len(final_results)))

            for iGT in range(len(GT_3D)):
                for iR in range(len(final_results)):

                    mean_error = 0
                    n_joints = 0
                    for j, gt3D in GT_3D[iGT].items():
                        idx = int(j)
                        if idx in parameters.used_joints:
                            p3D = final_results[iR][idx]
                            err = np.linalg.norm(p3D - gt3D)
                            mean_error += err
                            n_joints += 1

                    if n_joints > 0:
                        mean_error = mean_error/n_joints
                        err_table[iGT, iR] = mean_error

            if len(GT_3D) <= len(final_results):
                perm = itertools.permutations(range(len(final_results)), len(GT_3D))
            else:
                perm = itertools.permutations(range(len(GT_3D)), len(GT_3D))
            min_acum_err = 10000.
            min_perm = None

            for p in perm:
                
                acum_err = 0
                for iGT, iR in enumerate(p):
                    if iR < len(final_results):
                        acum_err += err_table[iGT, iR]
                if acum_err < min_acum_err:
                    min_acum_err = acum_err
                    min_perm = p

            n_poses += len(final_results)
            n_gt += len(GT_3D)

            for iR in range(len(final_results)):
                if iR in min_perm:
                    iGT = min_perm.index(iR)
                    if valid_GT[iGT]:
                        n_matching_poses += 1
                        global_acum_err += err_table[iGT,iR]
                    else:
                        # print('not valid')
                        n_gt -= 1
                for i_th, th in enumerate(mpjpe_threshold):
                    if iR in min_perm:
                        iGT = min_perm.index(iR)
                        if not valid_GT[iGT]:
                            continue
                        err = err_table[iGT,iR]
                        if err*1000. < th:
                            correct_poses[i_th] += 1
                            TP[i_th].append(1)
                            FP[i_th].append(0)
                        else:
                            TP[i_th].append(0)
                            FP[i_th].append(1)
                    else:
                        TP[i_th].append(0)
                        FP[i_th].append(1)

for i_th, th in enumerate(mpjpe_threshold):
    TP_np = np.array(TP[i_th])
    FP_np = np.array(FP[i_th])
    TP_np = np.cumsum(TP_np)
    FP_np = np.cumsum(FP_np)
    recall = TP_np / (n_gt + 1e-5)
    precise = TP_np / (TP_np + FP_np + 1e-5)
    total_num = len(TP[i_th])
    for n in range(total_num - 2, -1, -1):
        precise[n] = max(precise[n], precise[n + 1])
    precise = np.concatenate(([0], precise, [0]))
    recall = np.concatenate(([0], recall, [1]))
    index = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])
    print('AP, precise and recall for', th, ':', ap, precise[-2], recall[-2])

if n_matching_poses > 0:
    print("MEAN ERR (mm)", global_acum_err*1000./n_matching_poses)
if n_data > 0:
    print('Mean time for graph matching', time_graph_matching / n_data)
    print('Mean time for graph matching (per person)', time_graph_matching_person / n_data)
    print('Mean time for 3D', time_3D / n_data)
    print('Mean time for 3D (per person)', time_3D_person / n_data)
