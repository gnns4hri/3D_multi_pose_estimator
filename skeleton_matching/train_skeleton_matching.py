import sys
import time
import json
import torch
import pickle

import numpy as np

from torch import nn
from torch.utils import data
import dgl
import argparse

from gat2 import GAT2 as GAT

from torch.utils.data import DataLoader

from graph_generator import MergedMultipleHumansDataset, HumanGraphFromView

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser(description='Skeleton-matching training for 3D multi-human pose estimation')

parser.add_argument('--trainset', type=str, nargs='+', required=True, help='List of json files composing the training set')
parser.add_argument('--devset', type=str, nargs='+', required=True, help='List of json files composing the development set')
parser.add_argument('--testset', type=str, nargs='+', required=True, help='List of json files composing the test set')

args = parser.parse_args()

# CONSTANTS
USE_BCE = False
n_output = 1
out_activation = nn.Sigmoid()
if USE_BCE:
    loss_function = nn.BCELoss()
else:
    loss_function = nn.MSELoss()

epochs = 100
lr = 1e-4
batch_size = 15
weight_decay = 1.e-20
patience = 5

hidden = [40, 40, 40, 30]
heads = [10, 10, 8, 5]
gnn_layers = len(hidden)+1
alpha = 0.15
in_drop = 0.
attn_drop = 0.
residual = False
limit = 120000
hid_activation = torch.nn.LeakyReLU()

alt = '3'
num_features = len(HumanGraphFromView.get_all_features(alt))
print(f'num_features: {num_features}')


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
    for graph, labels, indices, _ in batch[1:]:
        graphs.append(graph)
        batched_labels = torch.cat([batched_labels, labels], dim=0)
        batched_indices = torch.cat([batched_indices, indices+total_nodes], dim=0)
        total_nodes += graph.number_of_nodes()
    batched_graphs = dgl.batch(graphs).to(torch.device(device))
    batched_labels.to(torch.device(device))
    batched_indices.to(torch.device(device))

    return batched_graphs, batched_labels, batched_indices


def evaluate(feats, indices, model, subgraph, labels, loss_fcn):
    model.eval()

    with torch.no_grad():
        feats.to(device)
        model.g = subgraph
        for layer in model.layers:
            layer.g = subgraph

        output = torch.squeeze(model(feats.float(), subgraph))

        labels = torch.squeeze(labels).to(device, dtype=torch.float32)
        indices = torch.squeeze(indices).to(device)
        filtered_output = output[indices]

        a = filtered_output.float().flatten()
        b = labels.float().flatten()
        loss_data = loss_fcn(a.to(device), b.to(device))
        predict = a.data.cpu().numpy()
        got = b.data.cpu().numpy()
        score = mean_squared_error(got, predict)
        return score, loss_data.item()


timea = time.time()

train_paths = args.trainset
dev_paths = args.devset
test_paths = args.testset

print(f'Using {train_paths} for training')
print(f'Using {dev_paths} for dev')
print(f'Using {test_paths} for testing')


probabilities = []
for json_paths in [train_paths, dev_paths, test_paths]:
    # Set probabilities based on the number of items
    first_length = len(json.loads(open(json_paths[0], "rb").read()))
    probabilities_set = [0.8]
    for filenamee in json_paths[1:]:
        this_length = len(json.loads(open(filenamee, "rb").read()))
        probabilities_set.append(0.8*this_length/first_length)
    print(probabilities_set)
    assert len(json_paths) == len(probabilities_set), "the lists of json files and probabilities must be the same length"
    probabilities.append(probabilities_set)

train_dataset = MergedMultipleHumansDataset(train_paths, probabilities[0], limit=limit, mode='train', alt=alt, raw_dir='.')
valid_dataset = MergedMultipleHumansDataset(dev_paths, probabilities[1], limit=limit, mode='dev', alt=alt, raw_dir='.')
test_dataset = MergedMultipleHumansDataset(test_paths, probabilities[2], limit=limit, mode='dev', alt=alt, raw_dir='.')
print(f'Dataset load time: {time.time()-timea}.')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate)

cur_step = 0
best_loss = -1
min_train_loss = float("inf")
min_dev_loss = float("inf")

model = GAT(None, gnn_layers, num_features, n_output, hidden, heads, hid_activation, out_activation,  in_drop,
            attn_drop, alpha, residual, bias=True)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

model = model.to(device)

print("Training...")

for epoch in range(0, epochs):
    model.train()
    batch_loss = 0.0
    batches = 0
    batch_loss_binary = 0.0
    for batch, data in enumerate(train_dataloader):
        subgraph, labels, indices = data

        optimizer.zero_grad()

        feats = subgraph.ndata['h'].to(device)

        model.g = subgraph
        for layer in model.layers:
            layer.g = subgraph
        outputs = torch.squeeze(model(feats.float(), subgraph))

        filtered_output = outputs[indices]


        # Compute loss
        loss = loss_function(filtered_output.float().to(device), labels.float().to(device))

        # Perform backward pass
        loss.backward()
        # Perform optimization
        optimizer.step()
        # Set current loss
        batch_loss += loss.item()
        batches += 1

        # Calculate 
        batch_loss_binary += mean_absolute_error((filtered_output.data.cpu().numpy() > 0.5)*1.,
                                                 labels.data.cpu().numpy())

    print("---------------------------------------")
    print(f'loss: {batch_loss/batches:.5f}')
    print(f'loss (crisp MAE): {batch_loss_binary/batches:.5f}')

    loss_data = batch_loss/batches
    if loss_data < min_train_loss:
        min_train_loss = loss_data

    if epoch % 5 == 0:
        print("---------------------------------------")
        print("Epoch {:05d} | Loss: {:.6f} | Patience: {} | ".format(epoch, loss_data, cur_step), end='')
        score_list = []
        val_loss_list = []

        for batch, valid_data in enumerate(valid_dataloader):
            subgraph, labels, indices = valid_data

            feats = subgraph.ndata['h']

            score, val_loss = evaluate(feats, indices, model, subgraph, labels, loss_function)
            score_list.append(score)
            val_loss_list.append(val_loss)

        mean_score = np.array(score_list).mean()
        mean_val_loss = np.array(val_loss_list).mean()

        print("Score: {:.6f} MEAN: {:.6f} BEST: {:.6f}".format(mean_score, mean_val_loss, best_loss))

        # early stop
        if best_loss > mean_val_loss or best_loss < 0:
            print('Saving...')

            best_loss = mean_val_loss
            if best_loss < min_dev_loss:
                min_dev_loss = best_loss

            # Save the model
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
            cur_step = 0
        else:
            cur_step += 1
            if cur_step >= patience:
                break

time_a = time.time()
test_score_list = []
valid_score_list = []
model.load_state_dict(torch.load('skeleton_matching.tch', map_location=device))

for check in ['test', 'dev']:
    if check == 'test':
        check_dataloader = test_dataloader
        check_score_list = test_score_list
    elif check == 'dev':
        check_dataloader = valid_dataloader
        check_score_list = valid_score_list
    else:
        raise Exception('check must be either "test" or "dev"')

    for batch, check_data in enumerate(check_dataloader):
        subgraph, labels, indices = check_data
        feats = subgraph.ndata['h']
        check_score_list.append(evaluate(feats, indices, model, subgraph, labels, loss_function)[1])

time_b = time.time()
time_delta = float(time_b-time_a)
test_loss = np.array(test_score_list).mean()
print(f'Test time {time_delta} seconds')
print("MSE for the test set {}".format(test_loss))

