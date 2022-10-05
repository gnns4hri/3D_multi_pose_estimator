import sys
import torch
import pickle
import json

import numpy as np
import networkx as nx

sys.path.append('../skeleton_matching')
from gat2 import GAT2 as GAT
from graph_generator import MergedMultipleHumansDataset, HumanGraphFromView

# from torch.utils.data import DataLoader

num_features = len(HumanGraphFromView.get_all_features())

if torch.cuda.is_available() is True:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

torch.set_grad_enabled(False)

sys.path.append('../')
from parameters import parameters 

sys.path.append('../utils')
from pose_estimator_dataset_from_json import PoseEstimatorDataset
from mlp import PoseEstimatorMLP
from skeleton_matching_utils import get_person_proposal_from_network_output

with open("../human_pose.json", 'r') as f:
    human_pose = json.load(f)
    skeleton = human_pose["skeleton"]
    keypoints = human_pose["keypoints"]

PLOTPERIOD = 0  # In miliseconds
CLASSIFICATION_THRESHOLD = 0.5

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np

class Visualizer(object):
    def __init__(self, period, json_files):
        self.plotLines = dict()
        self.app = QtGui.QApplication(sys.argv)
        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = 4
        self.w.setBackgroundColor((255, 255, 255, 255))
        self.w.setWindowTitle('3D Multi-pose estimator')
        self.w.setGeometry(0, 110, 1080, 1080)
        self.w.show()

        # create the background grids
        gx = gl.GLGridItem()
        gx.setColor((150,150,150))
        gx.setSize(3, 10, 8)
        gx.rotate(90, 0, 1, 0)
        gx.translate(-5, 0, 1.5)
        self.w.addItem(gx)
        gy = gl.GLGridItem()
        gy.setColor((150,150,150))
        gy.setSize(10, 3, 0)
        gy.rotate(90, 1, 0, 0)
        gy.translate(0, -5, 1.5)
        self.w.addItem(gy)
        gz = gl.GLGridItem()
        gz.setColor((150,150,150))
        gz.setSize(10, 10, 0)
        self.w.addItem(gz)

        self.period = period
        self.plotPoints = None
        self.plotLines = None

        self.init_models()

        self.input_data = []
        for json_file in json_files:
            self.input_data += json.load(open(json_file, 'rb'))
        self.itert = 0

    def init_models(self):
        # Instantiate the MLP
        numbers_per_joint = parameters.numbers_per_joint
        self.mlp = PoseEstimatorMLP(input_dimensions=len(parameters.cameras)*len(parameters.joint_list)*numbers_per_joint, output_dimensions=54)
        saved = torch.load('../pose_estimator.pytorch', map_location=device)
        self.mlp.load_state_dict(saved['model_state_dict'])

        params = pickle.load(open('../skeleton_matching.prms', 'rb'))
        self.model = GAT(None, params['gnn_layers'], params['num_feats'], params['n_classes'], params['num_hidden'], params['heads'],
                params['nonlinearity'], params['final_activation'], params['in_drop'], params['attn_drop'], params['alpha'], params['residual'], bias=True)
        self.model.load_state_dict(torch.load('../skeleton_matching.tch', map_location=device))
        self.model = self.model.to(device)


    def process_data(self):
        self.itert += 1
        if self.itert >= len(self.input_data):
            exit()

        input_element = self.input_data[self.itert]
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
            print('empty scenario')
            return

        subgraph = scenario.graphs[0].to(device)
        indices = scenario.data['edge_nodes_indices'][0].to(device)
        nodes_camera = scenario.data['nodes_camera'][0]
        feats = subgraph.ndata['h'].to(device)

        self.model.g = subgraph
        for layer in self.model.layers:
            layer.g = subgraph
        outputs = torch.squeeze(self.model(feats.float(), subgraph))

        indices = torch.squeeze(indices).to('cpu')

        final_output = get_person_proposal_from_network_output(outputs, subgraph, indices, nodes_camera, CLASSIFICATION_THRESHOLD)

        lines = []
        points = []
        points_pid = []
        lines_pid = []
        for person_id, person in enumerate(final_output):

            raw_input = dict()
            for cam_idx in parameters.cameras:
                camera = parameters.camera_names[cam_idx]
                if person[camera] is not None:
                    pc = person[camera]
                    all_joints_data = [scenario.jsons_for_head[pc]]
                    raw_input[camera] = [json.dumps(all_joints_data)]

            inputs = PoseEstimatorDataset(raw_input, parameters.cameras, parameters.joint_list, save=False)
            inputs = inputs[0][0].reshape([1, inputs[0][0].size()[0]])

            outputs = self.mlp(inputs)

            results_3d = torch.squeeze(outputs)*10.

            x3D = results_3d[::3]
            z3D = -results_3d[1::3]
            y3D = results_3d[2::3]

            for idx in range(len(skeleton)):
                line_x3D = []
                line_y3D = []
                line_z3D = []
                if skeleton[idx][0]-1 in parameters.used_joints and skeleton[idx][1]-1 in parameters.used_joints:
                    line_x3D.append(x3D[skeleton[idx][0]-1])
                    line_y3D.append(y3D[skeleton[idx][0]-1])
                    line_z3D.append(z3D[skeleton[idx][0]-1])
                    line_x3D.append(x3D[skeleton[idx][1]-1])
                    line_y3D.append(y3D[skeleton[idx][1]-1])
                    line_z3D.append(z3D[skeleton[idx][1]-1])
                    lines.append((line_x3D, line_y3D, line_z3D))
                    lines_pid.append(person_id)

            #
            # Plot the coordinates in 3D
            #
            for p in range(len(x3D)):
                if p in parameters.used_joints:
                    points.append([x3D[p], y3D[p], z3D[p]])
                    points_pid.append(person_id)


        for i, line in enumerate(lines):
            lines[i] = np.array([[line[0][0].item(), line[1][0].item(), line[2][0].item()],
                            [line[0][1].item(), line[1][1].item(), line[2][1].item()]])

        self.update_step(np.array(points), lines, points_pid, lines_pid)


    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def update_step(self, points, lines, points_pid, lines_pid):
        color_list = ['r', 'g', 'b', 'm', 'c', 'y', 'k']
        colors = []
        for pid in points_pid:
            colors.append(pg.glColor(color_list[pid]))

        if self.plotPoints is not None:
            width = 5
            self.plotPoints.setData(pos=points, color=np.array(colors), size=width)
        else:
            self.plotPoints = gl.GLScatterPlotItem(pos=points, color=np.array(colors), size=5., pxMode=True)
            self.plotPoints.setGLOptions('opaque')
            
            self.w.addItem(self.plotPoints)

        if self.plotLines is not None:
            for i in range(len(self.plotLines)):
                self.w.removeItem(self.plotLines[i])
            self.plotLines.clear()
        else:
            self.plotLines = dict()
        for i, line in enumerate(lines):
            self.plotLines[i] = gl.GLLinePlotItem(pos=line, color=pg.glColor(
                color_list[lines_pid[i]]), width=3, antialias=True)
            self.w.addItem(self.plotLines[i])


    def animation(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.process_data)
        self.timer.start(self.period)
        self.start()


v = Visualizer(PLOTPERIOD, sys.argv[1:])
v.animation()

