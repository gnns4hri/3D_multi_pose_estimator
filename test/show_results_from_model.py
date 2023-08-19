import sys
import torch
import pickle
import json
import copy
import numpy as np
import argparse

sys.path.append('../skeleton_matching')
from gat2 import GAT2 as GAT
from graph_generator import MergedMultipleHumansDataset, HumanGraphFromView

sys.path.append('../')
from parameters import parameters 

sys.path.append('../utils')
from pose_estimator_dataset_from_json import PoseEstimatorDataset
from mlp import PoseEstimatorMLP
from skeleton_matching_utils import get_person_proposal_from_network_output

parser = argparse.ArgumentParser(description='Display 3D multi-pose results using the 3D pose estimation model')

parser.add_argument('--testfile', type=str, nargs=1, required=True, help='Test file used as input')
parser.add_argument('--showgt', action='store_true', help='Show ground truth')
parser.add_argument('--tmfile', type=str, nargs=1, help='Directory that contains the files with the transfomation matrices')
parser.add_argument('--modelsdir', type=str, nargs='?', required=False, default='../models/', help='Directory that contains the models\' files')
parser.add_argument('--plotperiod', type=int, nargs='?', required=False, default=0, help='Plot period (miliseconds)')
parser.add_argument('--datastep', type=int, nargs='?', required=False, default=10, help='Data step used to plot the results')

args = parser.parse_args()

if args.showgt and args.tmfile is None:
    parser.error("--showgt requires --tmfile")

TEST_FILE = args.testfile

MODELSDIR = args.modelsdir
if MODELSDIR[-1] != '/':
    MODELSDIR += '/'

num_features = len(HumanGraphFromView.get_all_features())

if torch.cuda.is_available() is True:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

torch.set_grad_enabled(False)

with open("../human_pose.json", 'r') as f:
    human_pose = json.load(f)
    skeleton = human_pose["skeleton"]
    keypoints = human_pose["keypoints"]

PLOTPERIOD = args.plotperiod  # In miliseconds
DATASTEP = args.datastep
CLASSIFICATION_THRESHOLD = 0.5

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np

SHOW_GT = args.showgt

if SHOW_GT:
    tm = pickle.load(open(parameters.transformations_path, 'rb'))
    camera_i_transforms = []
    tm_dataset = pickle.load(open(args.tmfile[0], 'rb'))
    dataset_camera_d_transforms = []
    for cam_idx, cam in enumerate(parameters.camera_names):
        trfm_model = tm.get_transform(parameters.camera_names[cam_idx], "root")
        camera_i_transforms.append(torch.from_numpy(trfm_model).type(torch.float32))
        trfm_dataset = tm_dataset.get_transform("root", parameters.camera_names[cam_idx])
        dataset_camera_d_transforms.append(torch.from_numpy(trfm_dataset).type(torch.float32))


class Visualizer(object):
    def __init__(self, period, json_files):
        self.plotLines = dict()
        self.app = QtWidgets.QApplication(sys.argv)
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

        self.axes_3D = parameters.axes_3D

        self.input_data = []
        for json_file in json_files:
            self.input_data += json.load(open(json_file, 'rb'))
        self.itert = 0

    def init_models(self):
        # Instantiate the MLP
        numbers_per_joint = parameters.numbers_per_joint
        self.mlp = PoseEstimatorMLP(input_dimensions=len(parameters.used_cameras)*len(parameters.joint_list)*numbers_per_joint, output_dimensions=54)
        saved = torch.load(MODELSDIR + 'pose_estimator.pytorch', map_location=device)
        self.mlp.load_state_dict(saved['model_state_dict'])
        self.mlp = self.mlp.to(device)

        # Instantiate the skeleton matching model
        if len(parameters.used_cameras)>1:
            params = pickle.load(open(MODELSDIR + 'skeleton_matching.prms', 'rb'))
            self.model = GAT(None, params['gnn_layers'], params['num_feats'], params['n_classes'], params['num_hidden'], params['heads'],
                    params['nonlinearity'], params['final_activation'], params['in_drop'], params['attn_drop'], params['alpha'], params['residual'], bias=True)
            self.model.load_state_dict(torch.load(MODELSDIR + 'skeleton_matching.tch', map_location=device))
            self.model = self.model.to(device)
        else:
            self.model = None


    def process_data(self):
        self.itert += 1
        if self.itert >= len(self.input_data):
            exit()

        if self.itert%DATASTEP!=0:
            return

        input_element = self.input_data[self.itert]
        if len(parameters.used_cameras)>1:
            processed_input = dict()
            for cam in input_element:
                data = json.loads(input_element[cam][0]) #input_element[cam][0]#
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

            indices = torch.squeeze(indices).to('cpu')#.to(device)

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

        lines = []
        points = []
        points_pid = []
        lines_pid = []

        ################### Show ground truth ###################

        if SHOW_GT:
            first_cam = list(input_element.keys())[0]
            if len(input_element[first_cam]) != 4:
                print("There is no ground truth in the specified file")
                exit()

            for c in input_element:
                if len(input_element[c][3]) >  len(input_element[first_cam][3]):
                    first_cam = c

            
            if len(input_element[first_cam][3]) == 0:
                return

            joints_3D_all = input_element[first_cam][3]

            GT_3D = []

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

            for person_id, person in enumerate(GT_3D):
                number_of_joints = len(parameters.joint_list)
                x3D = np.zeros(number_of_joints)
                y3D = np.zeros(number_of_joints)
                z3D = np.zeros(number_of_joints)

                for j in range(number_of_joints):
                    idx = str(j)
                    if idx in person and j in parameters.used_joints:
                        x3D[j] = person[idx][self.axes_3D['X'][0]]*self.axes_3D['X'][1]
                        y3D[j] = person[idx][self.axes_3D['Y'][0]]*self.axes_3D['Y'][1]
                        z3D[j] = person[idx][self.axes_3D['Z'][0]]*self.axes_3D['Z'][1]


                for idx in range(len(skeleton)):
                    line_x3D = []
                    line_y3D = []
                    line_z3D = []
                    if skeleton[idx][0]-1 in parameters.used_joints and skeleton[idx][1]-1 in parameters.used_joints:
                        if str(skeleton[idx][0]-1) in person.keys() and str(skeleton[idx][1]-1) in person.keys():
                            line_x3D.append(x3D[skeleton[idx][0]-1])
                            line_y3D.append(y3D[skeleton[idx][0]-1])
                            line_z3D.append(z3D[skeleton[idx][0]-1])
                            line_x3D.append(x3D[skeleton[idx][1]-1])
                            line_y3D.append(y3D[skeleton[idx][1]-1])
                            line_z3D.append(z3D[skeleton[idx][1]-1])
                            lines.append((line_x3D, line_y3D, line_z3D))
                            lines_pid.append(6)                        

        ################### Get and show estimation ###################

        batched_input = []        
        for person_id, person in enumerate(final_output):

            if len(parameters.used_cameras)>1:
                raw_input = dict()
                for cam_idx, camera in enumerate(parameters.used_cameras):
                    if person[camera] is not None:
                        pc = person[camera]
                        all_joints_data = [scenario.jsons_for_head[pc]]
                        raw_input[camera] = [json.dumps(all_joints_data)]
            else:
                raw_input = person
                for cam in raw_input:
                    all_joints_data = json.loads(raw_input[cam][0])

            if not raw_input:
                continue

            inputs = PoseEstimatorDataset(raw_input, parameters.cameras, parameters.joint_list, save=False)
            inputs = inputs[0][0].reshape([1, inputs[0][0].size()[0]]).to(device)
            batched_input.append(inputs)       

        # GET the 3D skeleton of all the detected persons
        input_all = torch.cat(batched_input, dim=0)
        output_all = self.mlp(input_all.to(device))

        for person_id in range(output_all.shape[0]):
            results_3d = torch.squeeze(output_all[person_id])*10.
            results_3d = results_3d.to('cpu')

            x3D = results_3d[self.axes_3D['X'][0]::3]*self.axes_3D['X'][1]
            y3D = results_3d[self.axes_3D['Y'][0]::3]*self.axes_3D['Y'][1]
            z3D = results_3d[self.axes_3D['Z'][0]::3]*self.axes_3D['Z'][1]

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
            QtWidgets.QApplication.instance().exec_()

    def update_step(self, points, lines, points_pid, lines_pid):
        color_list = ['r', 'g', 'b', 'm', 'c', 'y', 'd']
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


v = Visualizer(PLOTPERIOD, TEST_FILE)
v.animation()

