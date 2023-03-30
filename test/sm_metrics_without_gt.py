import sys
import torch
import pickle
import json
import argparse
import dgl
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure


sys.path.append('../skeleton_matching')
from gat2 import GAT2 as GAT
from graph_generator import MergedMultipleHumansDataset, HumanGraphFromView
sys.path.append('../utils')
from skeleton_matching_utils import get_person_proposal_from_network_output


sys.path.append('../')
from parameters import parameters 

parser = argparse.ArgumentParser(description='Print metrics of the skeleton-matching model (ground truth is not required)')

parser.add_argument('--testfiles', type=str, nargs='+', required=True, help='List of json files used as input (each file contains data of a single individual)')
parser.add_argument('--modelsdir', type=str, nargs='?', required=False, default='../models/', help='Directory that contains the models\' files')
parser.add_argument('--datastep', type=int, nargs='?', required=False, default=12, help='Data step used to compute the metrics')


args = parser.parse_args()

TEST_FILES = args.testfiles

MODELSDIR = args.modelsdir
if MODELSDIR[-1] != '/':
    MODELSDIR += '/'


if torch.cuda.is_available() is True:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

torch.set_grad_enabled(False)


def collate(batch):
    graphs = [batch[0][0]]         # graphs are merged later using dgl.batch
    batched_labels = batch[0][1]   # labels are merged using `cat`
    batched_indices = batch[0][2]  # indices are merged using `cat`
    batched_nodes_camera = batch[0][3]  # indices are merged using `cat`
    total_nodes = batch[0][0].number_of_nodes()  # number of processed nodes after the first sampled graph

    # Process remaining samples
    for graph, labels, indices, nodes_camera in batch[1:]:
        graphs.append(graph)
        batched_labels = torch.cat([batched_labels, labels], dim=0)
        batched_indices = torch.cat([batched_indices, indices+total_nodes], dim=0)
        batched_nodes_camera = torch.cat([batched_nodes_camera, nodes_camera], dim=0)
        total_nodes += graph.number_of_nodes()
    batched_graphs = dgl.batch(graphs).to(torch.device(device))
    batched_labels.to(torch.device(device))
    batched_indices.to(torch.device(device))

    return batched_graphs, batched_labels, batched_indices, batched_nodes_camera

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

params = pickle.load(open(MODELSDIR + 'skeleton_matching.prms', 'rb'))
model = GAT(None, params['gnn_layers'], params['num_feats'], params['n_classes'], params['num_hidden'], params['heads'],
        params['nonlinearity'], params['final_activation'], params['in_drop'], params['attn_drop'], params['alpha'], params['residual'], bias=True)
model.load_state_dict(torch.load(MODELSDIR + 'skeleton_matching.tch', map_location=device))
model = model.to(device)

non_used_cameras = [] #['trackerc','trackerd','trackere']

total_data = 0 
n_input = 0

first_length = len(json.loads(open(TEST_FILES[0], "rb").read()))
probabilities_set = [0.8]
for filename in TEST_FILES[1:]:
    this_length = len(json.loads(open(filename, "rb").read()))
    probabilities_set.append(0.8*this_length/first_length)

print('LOADING GRAPHS...')
test_dataset = MergedMultipleHumansDataset(TEST_FILES, probabilities_set, limit=1000, mode='test_generated', alt='3', raw_dir='.')

test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate)

for data in tqdm(test_dataloader):
    subgraph, labels, indices, nodes_camera = data

    feats = subgraph.ndata['h'].to(device)
    all_nodes = subgraph.nodes()
    all_nodes = all_nodes.tolist()
    
    model.g = subgraph
    for layer in model.layers:
        layer.g = subgraph
    outputs = torch.squeeze(model(feats.float(), subgraph))

    labels = torch.squeeze(labels).to('cpu')
    indices = torch.squeeze(indices).to('cpu')
    head_nodes = list(set(all_nodes) -  set(indices.tolist()))

    n_data += 1

    output_features = outputs.tolist()
    final_output = get_person_proposal_from_network_output(outputs, subgraph, indices, nodes_camera, None, CLASSIFICATION_THRESHOLD)

    metrics_estimation = []
    for h in head_nodes:
        person_index = 0
        for person in final_output:
            if h in list(person.values()):
                break
            person_index += 1
        metrics_estimation.append(person_index)

    
    output_features = [0.]*len(all_nodes)
    for (i, v) in zip(indices, labels.tolist()):
        output_features[i] = v

    final_output = get_person_proposal_from_network_output(output_features, subgraph, indices, nodes_camera, None, CLASSIFICATION_THRESHOLD)

    metrics_gt = []
    for h in head_nodes:
        person_index = 0
        for person in final_output:
            if h in list(person.values()):
                break
            person_index += 1
        metrics_gt.append(person_index)

    r_score += adjusted_rand_score(metrics_gt, metrics_estimation)            
    l_homogeneity, l_completeness, l_v_measure = homogeneity_completeness_v_measure(metrics_gt, metrics_estimation)
    homogeneity += l_homogeneity
    completeness += l_completeness
    v_measure += l_v_measure

print('rand score', r_score/n_data)
print('homogeneity', homogeneity/n_data)
print('completeness', completeness/n_data)
print('v_measure', v_measure/n_data)
