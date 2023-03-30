import sys
import torch
import pickle
import json
import argparse
import numpy as np
import copy
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure


sys.path.append('../skeleton_matching')
from gat2 import GAT2 as GAT
from graph_generator import MergedMultipleHumansDataset, HumanGraphFromView
sys.path.append('../utils')
from skeleton_matching_utils import get_person_proposal_from_network_output


sys.path.append('../')
from parameters import parameters 

parser = argparse.ArgumentParser(description='Print metrics of the skeleton-matching model (ground truth is required)')

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

n_data = 0
DATASTEP = args.datastep
n_measures = 0
n_correct_matching = 0
n_true_matching = 0
n_estimated_matching = 0
r_score = 0
homogeneity = 0
completeness = 0
v_measure = 0
AMI_score = 0
n_people_gt = 0
n_people_estimation = 0

#######################################

numbers_per_joint = parameters.numbers_per_joint

params = pickle.load(open('../models_panoptic/skeleton_matching.prms', 'rb'))
model = GAT(None, params['gnn_layers'], params['num_feats'], params['n_classes'], params['num_hidden'], params['heads'],
        params['nonlinearity'], params['final_activation'], params['in_drop'], params['attn_drop'], params['alpha'], params['residual'], bias=True)
model.load_state_dict(torch.load('../models_panoptic/skeleton_matching.tch', map_location=device))
model = model.to(device)

non_used_cameras = [] #['trackerc','trackerd','trackere']

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

            n_measures += 1

            # LOAD GROUND TRUTH

            first_cam = list(input_element.keys())[0]
            if len(input_element[first_cam]) != 4:
                print("There is no ground truth in the specified file")
                exit()

            valid_gt = True
            # for each person, save the indices of the skeletons in every camera
            GT_3D = []
            local_metrics_gt = []
            valid = True
            for cam in input_element:
                if not cam in parameters.used_cameras:
                    continue
                joints_3D_all = input_element[cam][3]
                for id_skeleton, joints_3D in enumerate(joints_3D_all):
                    if not '-1' in joints_3D.keys():
                        valid = False
                    min_dist = 1000000000.
                    matched_person = -1
                    n_joints = 0
                    for id_person, person in enumerate(GT_3D):
                        cur_GT = person['3D']
                        dist = 0
                        n_joints_cur = 0
                        for idx, p3D in cur_GT.items():
                            if idx in joints_3D:
                                dist += np.linalg.norm(np.array(joints_3D[idx]) - np.array(p3D))
                                n_joints_cur += 1
                        if dist < min_dist:
                            min_dist = dist
                            matched_person = id_person
                            n_joints = n_joints_cur

                    if n_joints == 0 or min_dist/n_joints>1.:
                        matched_person = -1
                    if matched_person < 0:
                        new_person = dict()
                        new_person['3D'] = copy.deepcopy(joints_3D)
                        new_person['skeletons'] = dict()
                        matched_person = len(GT_3D)
                        GT_3D.append(new_person)
                    GT_3D[matched_person]['skeletons'][cam] = id_skeleton
                    local_metrics_gt.append(matched_person+n_people_gt)
            
            matching_GT = []
            for gt in GT_3D:
                matching_GT.append(gt['skeletons'])

            if not matching_GT or not valid:
                continue

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

            try:
                subgraph = scenario.graphs[0].to(device)
                indices = scenario.data['edge_nodes_indices'][0].to(device)
                nodes_camera = scenario.data['nodes_camera'][0]
                feats = subgraph.ndata['h'].to(device)
                all_nodes = subgraph.nodes()
                all_nodes = all_nodes.tolist()


                model.g = subgraph
                for layer in model.layers:
                    layer.g = subgraph
                outputs = torch.squeeze(model(feats.float(), subgraph))

                indices = torch.squeeze(indices).to('cpu')
                head_nodes = list(set(all_nodes) -  set(indices.tolist()))                
            except:
                continue

            n_data += 1
            n_people_gt += len(GT_3D)
            metrics_gt = local_metrics_gt

            # Process the output graph as it comes from the GNN
            final_output = get_person_proposal_from_network_output(outputs, subgraph, indices, nodes_camera, scenario.jsons_for_head, CLASSIFICATION_THRESHOLD)

            metrics_estimation = []
            for h in head_nodes:
                person_index = 0
                for person in final_output:
                    if h in list(person.values()):
                        break
                    person_index += 1
                metrics_estimation.append(person_index+n_people_estimation)

            r_score += adjusted_rand_score(metrics_gt, metrics_estimation)            
            l_homogeneity, l_completeness, l_v_measure = homogeneity_completeness_v_measure(metrics_gt, metrics_estimation)
            homogeneity += l_homogeneity
            completeness += l_completeness
            v_measure += l_v_measure

print('rand score', r_score/n_data)
print('homogeneity', homogeneity/n_data)
print('completeness', completeness/n_data)
print('v_measure', v_measure/n_data)
