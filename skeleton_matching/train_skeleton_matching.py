#
# TODO:
# - Transformation matrices should be 3 rows only (the last row should be fixed to [0,0,0,1]).
# - Camera matrices should only have four values really, not 3x3 (we don't want torch to optimise the whole matrix).
import sys
import copy
import time
import json
import torch
import pickle

import numpy as np

from torch import nn
from torch._C import dtype
from torch.autograd import Variable
from torch.utils import data
import dgl

from gat2 import GAT2 as GAT

from torch.utils.data import DataLoader

from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

from graph_generator import MergedMultipleHumansDataset, HumanGraphFromView

alt = '3'
num_features = len(HumanGraphFromView.get_all_features(alt))
print(f'num_features: {num_features}')

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# CONSTANTS
SOFTMAX = False
if SOFTMAX:
    out_activation = torch.nn.Softmax(dim=1)
    loss_function = nn.BCELoss()
    n_output = 2
else:
    n_output = 1
    out_activation = torch.nn.Sigmoid()
    loss_function = nn.MSELoss()
    # loss_function = nn.BCELoss()

epochs = 100
lr = 1e-4
batch_size = 15
weight_decay = 1.e-20

hidden = [40, 40, 40, 30]
heads = [10, 10, 8, 5]
gnn_layers = len(hidden)+1
alpha = 0.15
in_drop = 0.
attn_drop = 0.
residual = False
limit = 100000000000
hid_activation = torch.nn.LeakyReLU()



if torch.cuda.is_available() is True:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def collate(batch):
    # Process first sample
    graphs = [batch[0][0]]         # graphs are merged later using dgl.batch
    batched_labels = batch[0][1]   # labels are merged using `cat`
    batched_indices = batch[0][2]  # indices are merged using `cat`
    total_nodes = batch[0][0].number_of_nodes()  # number of processed nodes after the first sampled graph

    # Process remaining samples
    for graph, labels, indices in batch[1:]:
        graphs.append(graph)
        batched_labels = torch.cat([batched_labels, labels], dim=0)
        batched_indices = torch.cat([batched_indices, indices+total_nodes], dim=0)
        total_nodes += graph.number_of_nodes()
    batched_graphs = dgl.batch(graphs).to(torch.device(device))
    batched_labels.to(torch.device(device))
    batched_indices.to(torch.device(device))

    if SOFTMAX:
        labels2 = torch.cat((batched_labels, -batched_labels+1.), dim=1)
    else:
        labels2 = batched_labels
    return batched_graphs, labels2, batched_indices


timea = time.time()

if len(sys.argv) < 2:
    print(f'Run: {sys.argv[0]} train1.json train2.json ... trainN.json')
    sys.exit(-1)


# Set input files
json_paths = [f for f in sys.argv[1:]]

# Set probabilities based on the number of items
first_length = len(json.loads(open(json_paths[0], "rb").read()))
probabilities = [0.8]
for filenamee in json_paths[1:]:
    this_length = len(json.loads(open(filenamee, "rb").read()))
    probabilities.append(0.8*this_length/first_length)
print(probabilities)

assert len(json_paths) == len(probabilities), "the lists of json files and probabilities must be the same length"
train_dataset = MergedMultipleHumansDataset(json_paths, probabilities, limit=limit, mode='train', alt=alt, raw_dir='.')
print(f'Dataset load time: {time.time()-timea}.')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate)

model = GAT(None, gnn_layers, num_features, n_output, hidden, heads, hid_activation, out_activation,  in_drop, attn_drop, alpha, residual, bias=True)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

model = model.to(device)

for epoch in range(0, epochs):
    model.train()
    batch_loss = 0.0
    batches = 0
    batch_loss_binary = 0.0
    for batch, data in enumerate(train_dataloader):
        subgraph, labels, indices = data

        optimizer.zero_grad()

        feats = subgraph.ndata['h'].to(device)

        # model.gnn_object.g = subgraph
        model.g = subgraph
        for layer in model.layers:
            layer.g = subgraph
        outputs = torch.squeeze(model(feats.float(), subgraph))

        labels = torch.squeeze(labels).to(device, dtype=torch.float32)
        indices = torch.squeeze(indices).to(device)
        filtered_output = outputs[indices]

        # print('labels', labels)

        # print('labels', filtered_output)

        # Compute loss
        loss = loss_function(filtered_output, labels)

        # Perform backward pass
        loss.backward()
        # Perform optimization
        optimizer.step()
        # Set current loss
        batch_loss += loss.item()
        batches += 1

        # Calculate 
        batch_loss_binary += mean_absolute_error((filtered_output.data.cpu().numpy()>0.5)*1., labels.data.cpu().numpy())

    print(epoch)
    print(f'loss: {batch_loss/batches:.5f}')
    print(f'loss (crisp MAE): {batch_loss_binary/batches:.5f}')

    torch.save(model.state_dict(), 'skeleton_matching.tch')

    params = {'loss': batch_loss,
              'net': 'gat',
              'gnn_layers': gnn_layers,
              'num_feats': num_features,
              'num_hidden': hidden,
              'graph_type': '1',
              'n_classes': n_output,
              'heads': heads,
              'nonlinearity': hid_activation,
              'final_activation': out_activation,
              'in_drop': in_drop,
              'attn_drop': attn_drop,
              'alpha': alpha,
              'residual': residual
              }
    pickle.dump(params, open('skeleton_matching.prms', 'wb'))


