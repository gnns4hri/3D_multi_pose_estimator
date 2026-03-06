import sys
import torch
import pickle
import json
import copy
import numpy as np
import argparse

#sys.path.append('../skeleton_matching')
#from graph_generator import MergedMultipleHumansDataset, HumanGraphFromView, get_working_temp_data


sys.path.append('../utils')
from pose_estimator_utils import camera_matrix, triangulate
from skeleton_matching_utils import get_person_proposal_from_network_output
from pose_estimator_dataset_from_json import build_support_data, PoseEstimatorDataset

from mlp import PoseEstimatorMLP

parser = argparse.ArgumentParser(description='Display 3D multi-pose results using triangulation')

parser.add_argument('--testfile', type=str, nargs=1, required=True, help='Test file used as input')
parser.add_argument('--modelsdir', type=str, nargs='?', required=False, default='../models/', help='Directory that contains the models\' files')
parser.add_argument('--plotperiod', type=int, nargs='?', required=False, default=0, help='Plot period (miliseconds)')
parser.add_argument('--datastep', type=int, nargs='?', required=False, default=10, help='Data step used to plot the results')
parser.add_argument('--config', type=str, required=True, help='YAML config file')
args = parser.parse_args()


sys.path.append('../')
# from parameters import parameters 
from parameters import generate_tracker_parameters_from_file
parameters = generate_tracker_parameters_from_file(args.config)
from pose_estimator_dataset_from_json import build_support_data
parameters = build_support_data(parameters)
#get_working_temp_data(parameters)


TEST_FILE = args.testfile

MODELSDIR = args.modelsdir


#num_features = len(HumanGraphFromView.get_all_features())


if torch.cuda.is_available() is True:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

torch.set_grad_enabled(False)

tm = pickle.load(open(parameters.transformations_path, 'rb'))
projection_matrices = {}
distortion_coefficients = {}
cam_matrix = {}
fisheye = {}


image_size = (parameters.image_width, parameters.image_height)
for cam_idx, cam in enumerate(parameters.camera_names):
    # Add the direct transform (root to camera) to the list
    trfm =   tm.get_transform(parameters.root, cam)
    cam_matrix[cam] = camera_matrix(cam_idx, parameters).cpu().detach().numpy()
    if parameters.fisheye[cam_idx]:
        distortion_coefficients[cam] = np.array([parameters.kd0[cam_idx], parameters.kd1[cam_idx], parameters.kd2[cam_idx], parameters.kd3[cam_idx]])
    else:
        distortion_coefficients[cam] = np.array([parameters.kd0[cam_idx], parameters.kd1[cam_idx], parameters.p1[cam_idx], parameters.p2[cam_idx], parameters.kd2[cam_idx]])
    fisheye[cam] = parameters.fisheye[cam_idx]        
    projection = trfm[0:3, :]
    projection_matrices[cam] = projection


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

class Visualizer(object):
    def __init__(self, period, json_files):
        self.plotLines = dict()
        self.app = QtWidgets.QApplication(sys.argv)
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

        self.axes_3D = parameters.axes_3D        

        self.input_data = []
        for json_file in json_files:
            self.input_data += json.load(open(json_file, 'rb'))
        self.itert = 0

        numbers_per_joint = parameters.numbers_per_joint
        self.mlp = PoseEstimatorMLP(input_dimensions=len(parameters.cameras)*len(parameters.joint_list)*numbers_per_joint, output_dimensions=3*len(parameters.joint_list))
        saved = torch.load(MODELSDIR + 'pose_estimator.pytorch', map_location=device)
        self.mlp.load_state_dict(saved['model_state_dict'])
        self.mlp = self.mlp.to(device)



    def process_data(self):
        self.itert += 1
        if self.itert >= len(self.input_data):
            exit()

        if self.itert%DATASTEP!=0:
            return
        
        input_element = self.input_data[self.itert]
        inputs = PoseEstimatorDataset(input_element, parameters.cameras, parameters.joint_list, parameters, save=False)
        inputs = inputs[0][0].reshape([1, inputs[0][0].size()[0]]).to(device)

        input_all = inputs #torch.tensor(inputs)
        output_all = self.mlp(input_all.to(device))

        result3D = torch.squeeze(output_all[0])*10.
        result3D = result3D.reshape((-1,3)).to('cpu')
        
        # print(result3D.shape)


        number_of_joints = len(parameters.joint_list)
        x3D = np.zeros(number_of_joints)
        y3D = np.zeros(number_of_joints)
        z3D = np.zeros(number_of_joints)

        for j in parameters.used_joints:
            x3D[j] = result3D[j][self.axes_3D['X'][0]]*self.axes_3D['X'][1]
            y3D[j] = result3D[j][self.axes_3D['Y'][0]]*self.axes_3D['Y'][1]
            z3D[j] = result3D[j][self.axes_3D['Z'][0]]*self.axes_3D['Z'][1]

        lines = []
        points = []
        points_pid = []
        lines_pid = []
        person_id = 0

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
        for j in parameters.used_joints:
            p = int(j)
            if p in parameters.used_joints:
                points.append([x3D[p], y3D[p], z3D[p]])
                points_pid.append(person_id)

        for i, line in enumerate(lines):
            lines[i] = np.array([[line[0][0].item(), line[1][0].item(), line[2][0].item()],
                            [line[0][1].item(), line[1][1].item(), line[2][1].item()]])

        self.update_step(np.array(points), lines, points_pid, lines_pid)


    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtWidgets.QApplication.instance().exec()

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

