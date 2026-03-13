import sys
import torch
import pickle
import json
import copy
import numpy as np
import argparse

sys.path.append('../skeleton_matching')
#from gat2 import GAT2 as GAT
#from graph_generator import MergedMultipleHumansDataset, HumanGraphFromView, get_working_temp_data


sys.path.append('../utils')
from pose_estimator_utils import camera_matrix, triangulate
from skeleton_matching_utils import get_person_proposal_from_network_output
from pose_estimator_dataset_from_json import build_support_data

parser = argparse.ArgumentParser(description='Display 3D multi-pose results using triangulation')

parser.add_argument('--testfile', type=str, nargs=1, required=True, help='Test file used as input')
parser.add_argument('--plotperiod', type=int, nargs='?', required=False, default=10, help='Plot period (miliseconds)')
parser.add_argument('--datastep', type=int, nargs='?', required=False, default=1, help='Data step used to plot the results')
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


with open("../checkerboard.json", 'r') as f:
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


    def process_data(self):
        self.itert += 1
        if self.itert >= len(self.input_data):
            exit()

        if self.itert%DATASTEP!=0:
            return
        input_element = self.input_data[self.itert]

        joints_data = dict()
        for cam in input_element:
            # print('INPUT', input_element[cam][0])
            all_cam_data = json.loads(input_element[cam][0])
            # print('CAM DATA', all_cam_data)
            if len(all_cam_data) == 0:
                continue
            cam_data = all_cam_data[0] # assuming only one human
            for j in cam_data:
                if cam_data[j][3] > 0.5:
                    if not j in joints_data.keys():
                        joints_data[j] = {}
                    if cam_data[j][1] > image_size[0] or cam_data[j][2] > image_size[1]:
                        print('fuera de imagen')
                    joints_data[j][cam] = [cam_data[j][1], cam_data[j][2]]

        result3D = triangulate(joints_data, cam_matrix, distortion_coefficients, projection_matrices, fisheye, parameters.axes_3D['Y'][0], parameters)

        number_of_joints = len(parameters.joint_list)
        x3D = np.zeros(number_of_joints)
        y3D = np.zeros(number_of_joints)
        z3D = np.zeros(number_of_joints)

        for j in parameters.used_joints:
            idx = str(j)
            if idx in result3D:
                x3D[j] = result3D[idx][self.axes_3D['X'][0]][0]*self.axes_3D['X'][1]
                y3D[j] = result3D[idx][self.axes_3D['Y'][0]][0]*self.axes_3D['Y'][1]
                z3D[j] = result3D[idx][self.axes_3D['Z'][0]][0]*self.axes_3D['Z'][1]

        lines = []
        points = []
        points_pid = []
        lines_pid = []
        person_id = 0

        for idx in range(len(skeleton)):
            line_x3D = []
            line_y3D = []
            line_z3D = []
            if str(skeleton[idx][0]-1) in result3D.keys() and str(skeleton[idx][1]-1) in result3D.keys():
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
        for j in result3D.keys():
            p = int(j)
            if p in parameters.used_joints:
                points.append([x3D[p], y3D[p], z3D[p]])
                points_pid.append(person_id)

        for i, line in enumerate(lines):
            lines[i] = np.array([[line[0][0].item(), line[1][0].item(), line[2][0].item()],
                            [line[0][1].item(), line[1][1].item(), line[2][1].item()]])

        if len(points)>0:
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

