import sys
import torch
import pickle
import json

import cv2
import numpy as np
import networkx as nx

sys.path.append('../skeleton_matching')
from gat2 import GAT2 as GAT
from graph_generator import MergedMultipleHumansDataset, HumanGraphFromView

num_features = len(HumanGraphFromView.get_all_features())

if torch.cuda.is_available() is True:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

import itertools

torch.set_grad_enabled(False)

sys.path.append('../')
from parameters import parameters 

sys.path.append('../utils')
from pose_estimator_utils import camera_matrix
from skeleton_matching_utils import get_person_proposal_from_network_output

tm = pickle.load(open(parameters.transformations_path, 'rb'))
projection_matrices = {}
distortion_coefficients = {}
cam_matrix = {}

image_size = (parameters.image_width, parameters.image_height)
for cam_idx, cam in enumerate(parameters.camera_names):
    # Add the direct transform (root to camera) to the list
    trfm =   tm.get_transform("root", parameters.camera_names[cam_idx])
    cam_matrix[cam] = camera_matrix(cam_idx).cpu().detach().numpy()
    distortion_coefficients[cam] = np.array([parameters.kd0[cam_idx], parameters.kd1[cam_idx], parameters.p1[cam_idx], parameters.p2[cam_idx], parameters.kd2[cam_idx]])
    projection = trfm[0:3, :]
    projection_matrices[cam] = projection


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
        self.w.setWindowTitle('Human tracker')
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
        # Instantiate the skeleton matching model

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

        indices = torch.squeeze(indices).to(device)

        final_output = get_person_proposal_from_network_output(outputs, subgraph, indices, nodes_camera, CLASSIFICATION_THRESHOLD)

        #
        #
        # Get 3D from triangulation
        #
        #


        lines = []
        points = []
        points_pid = []
        lines_pid = []
        for person_id, person in enumerate(final_output):

            # Organize the points considering the different views
            points_2D = dict()

            for cam_idx in parameters.cameras:
                camera = parameters.camera_names[cam_idx]
                if person[camera] is not None:
                    pc = person[camera]
                    all_joints_data = scenario.jsons_for_head[pc]
                    for j, pos in all_joints_data.items():
                        if not j in points_2D.keys():
                            points_2D[j] = dict()
                        points_2D[j][camera] = np.array([pos[1], pos[2]])
                    

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
                        new_point1 = cv2.undistortPoints(np.array([point1]), cam_matrix[cam1], distortion_coefficients[cam1])
                        point2 = np.array(points_2D[idx][cam2])
                        new_point2 = cv2.undistortPoints(np.array([point2]), cam_matrix[cam2], distortion_coefficients[cam2])
                        point3d = cv2.triangulatePoints(projection_matrices[cam1], projection_matrices[cam2], new_point1, new_point2)
                        point3d = point3d[0:3]/point3d[3]
                        mean_point3D += point3d
                        n_comb += 1
                    result3D[idx] = mean_point3D/n_comb
            

            number_of_joints = len(parameters.joint_list)
            x3D = np.zeros(number_of_joints)
            y3D = np.zeros(number_of_joints)
            z3D = np.zeros(number_of_joints)

            for j in range(number_of_joints):
                idx = str(j)
                if idx in result3D:
                    x3D[j] = result3D[idx][0][0]
                    z3D[j] = -result3D[idx][1][0]
                    y3D[j] = result3D[idx][2][0]


            for idx in range(len(skeleton)):
                line_x3D = []
                line_y3D = []
                line_z3D = []
                if str(skeleton[idx][0]-1) in result3D.keys() and str(skeleton[idx][1]-1) in result3D.keys():
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
            for j in result3D.keys():
                p = int(j)
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



