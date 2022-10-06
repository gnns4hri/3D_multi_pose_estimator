import sys
import os
import cv2
import numpy as np
import os.path
import copy
import json
import torchvision.transforms as transforms
import panutils
import torch
from easydict import EasyDict as edict
import yaml
import sys
import time
import pose_resnet
import torchvision.transforms as transforms
from string import ascii_lowercase
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager
import pickle


def load_panoptic_model():
    config_file = "./cfg/prn64_cpn80x80x20_960x512_cam5.yaml"
    with open(config_file) as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))


    backbone = pose_resnet.get_pose_net(cfg, is_train=True)

    model = backbone

    model = model.to('cuda')
    model.eval()
    return model

def get_output_from_panoptic_model(img, model):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tr = transforms.Compose([transforms.ToTensor(), normalize, ])

    image_size = (960, 512)
    img_input = cv2.resize(img, (960, 512))

    t = tr(img_input)

  
    res = model(t[None, :].to('cuda'))
    res = res.to('cpu')

    return res

import trt_pose.coco
import trt_pose.plugins


class ParseObjects(object):
    
    def __init__(self, topology, cmap_threshold=0.1, link_threshold=0.1, cmap_window=5, line_integral_samples=7, max_num_parts=100, max_num_objects=100):
        self.topology = topology
        self.cmap_threshold = cmap_threshold
        self.link_threshold = link_threshold
        self.cmap_window = cmap_window
        self.line_integral_samples = line_integral_samples
        self.max_num_parts = max_num_parts
        self.max_num_objects = max_num_objects
    
    def __call__(self, cmap):
        
        peak_counts, peaks = trt_pose.plugins.find_peaks(cmap, self.cmap_threshold, self.cmap_window, self.max_num_parts)
        normalized_peaks = trt_pose.plugins.refine_peaks(peak_counts, peaks, cmap, self.cmap_window)
        return normalized_peaks

with open('../human_pose.json', 'r') as f:
    human_pose = json.load(f)
topology = trt_pose.coco.coco_category_to_topology(human_pose)
num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])
parse_objects = ParseObjects(topology, cmap_threshold=0.15, link_threshold=0.15)

draw = False

# Setup paths
seq_name = sys.argv[1]

if seq_name[-1] == '/':
    seq_name = seq_name[:-1]

hd_skel_json_path = seq_name+'/hdPose3d_stage1_coco19/'

# Load camera calibration parameters
with open(seq_name+'/calibration_{0}.json'.format(seq_name.split('/')[-1])) as cfile:
    calib = json.load(cfile)

# Cameras are identified by a tuple of (panel#,node#)
cameras = {(cam['panel'],cam['node']):cam for cam in calib['cameras']}

# Convert data into numpy arrays for convenience
for k,cam in cameras.items():    
    cam['K'] = np.matrix(cam['K'])
    cam['distCoef'] = np.array(cam['distCoef'])
    cam['R'] = np.matrix(cam['R'])
    cam['t'] = np.array(cam['t']).reshape((3,1))

selected_cameras = dict()
selected_cameras['trackera'] = cameras[(0,3)]
selected_cameras['trackerb'] = cameras[(0,6)]
selected_cameras['trackerc'] = cameras[(0,12)]
selected_cameras['trackerd'] = cameras[(0,13)]
selected_cameras['trackere'] = cameras[(0,23)]

# Create cameras' transformations file
tm = TransformManager()

for cam_name, cam in selected_cameras.items():
    tr = pt.transform_from(cam['R'], cam['t'][:,0]/100.)
    tm.add_transform("root", cam_name, tr)

pickle.dump(tm, open('tm_'+seq_name.split('/')[-1]+'.pickle', 'wb'))


# Transform coco 19 into coco 18
id_joint = dict()
id_joint[0] = '17'  # Neck
id_joint[1] = '0'   # Nose
# joint 2 is bodyCenter (not in coco 18)
id_joint[3] = '5'   # left shoulder
id_joint[4] = '7'   # left elbow
id_joint[5] = '9'   # left wrist
id_joint[6] = '11'  # left hip
id_joint[7] = '13'  # left knee
id_joint[8] = '15'  # left ankle
id_joint[9] = '6'   # right shoulder
id_joint[10] = '8'  # right elbow
id_joint[11] = '10' # right wrist
id_joint[12] = '12' # right hip
id_joint[13] = '14' # right knee
id_joint[14] = '16' # right ankle
id_joint[15] = '1'  # left eye
id_joint[16] = '3'  # left ear
id_joint[17] = '2'  # right eye
id_joint[18] = '4'  # right ear 


# Access to the images' folder
hd_imgs_path = seq_name+'/hdImgs/'

cam_directories = os.listdir(hd_imgs_path) #['00_03']
cam_directories.sort()

camera_names = dict()
letters = [ascii_lowercase[i] for i in range(len(cam_directories))]
for i, l in enumerate(letters):
    cam = int(cam_directories[i].split('_')[-1])
    camera_names[cam] = 'tracker'+l


# Get images' paths and organize them

images_info = {}
for c in cam_directories:
    cam_id = int(c.split('_')[-1])
    print(cam_id)
    imgs_path = os.path.join(hd_imgs_path, c)
    imgs = [f for f in os.listdir(imgs_path) if os.path.isfile(os.path.join(imgs_path, f))]
    imgs.sort()
    for img_name in imgs:
        img_id = img_name.split('.')[-2].split('_')[-1]
        if not img_id in images_info.keys():
            images_info[img_id] = {}
            images_info[img_id]['cameras'] = {}
            images_info[img_id]['json'] = os.path.join(hd_skel_json_path, 'body3DScene_'+img_id+'.json')
        images_info[img_id]['cameras'][cam_id] = os.path.join(imgs_path, img_name)

humans_json = list()
cont = 0
model = load_panoptic_model()
for image in images_info.values():
    if not os.path.exists(image['json']):
        continue

    cont += 1
    with open(image['json']) as dfile:
        bframe = json.load(dfile)

    # kps_per_human_and_cam = dict()

    kps_per_cam = dict()
    for cam in image['cameras']:
        bodies_kps = list()

        bodies_3D = list()

        # Detection from image
        img_name = image['cameras'][cam]
        color_image = cv2.imread(img_name)
        ret = copy.deepcopy(color_image)

        panoptic_output = get_output_from_panoptic_model(color_image, model)

        peaks = parse_objects(panoptic_output)

        # Projection from 3D
        joints_3D = dict()
        projected_people = {}
        for body in bframe['bodies']:
            id_person = body['id']
            joints_3D[id_person] = dict()
            skel = np.array(body['joints19']).reshape((-1,4)).transpose()

            # Project skeleton into view (this is like cv2.projectPoints)
            pt = panutils.projectPoints(skel[0:3,:],
                        cameras[(0,cam)]['K'], cameras[(0,cam)]['R'], cameras[(0,cam)]['t'], 
                        cameras[(0,cam)]['distCoef'])


            # Show only points detected with confidence
            valid = skel[3,:]>0.1

            pt = pt.transpose()

            kps = dict()
            for i, joint in enumerate(pt):
                if not valid[i]:
                    continue
                if i!=2:
                    kp = id_joint[i]
                else:
                    kp = '-1'
                joints_3D[id_person][kp] = [float(skel[0][i]), float(skel[1][i]), float(skel[2][i])]
                if joint[0] < 0 or joint[0] >= cameras[(0,cam)]['resolution'][0] or joint[1] < 0 or joint[1] >= cameras[(0,cam)]['resolution'][1]:
                    continue
                x = joint[0]
                y = joint[1]
                kps[int(kp)] = [int(kp), float(x), float(y), 1, 1]
                if draw:
                    for idx, joint in data.items():
                        x = joint[1]
                        y = joint[2]
                        cv2.circle(ret, (int(x), int(y)), 1, (0, 0, 255), 2)
                        cv2.putText(ret, "%d" % int(kp), (int(x) + 5, int(y)),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
            projected_people[id_person] = copy.deepcopy(kps)

        # Organize detected joints into separate skeletons according to the their proximity to projected people

        detected_joints = dict()
        for j, person in enumerate(peaks[0]):
            if torch.count_nonzero(person) == 0:
                break
            if j == 2:
                continue
            idx = int(id_joint[j])
            detected_joints[idx] = list()
            for kp in person:
                if torch.count_nonzero(kp) == 0:
                    continue
                y = kp[0] * cameras[(0, cam)]['resolution'][1]
                x = kp[1] * cameras[(0, cam)]['resolution'][0]
                detected_joints[idx].append([x, y])
                if draw:
                    cv2.circle(ret, (int(x), int(y)), 1, (0, 255, 0), 2)
                    if j!=2:
                        cv2.putText(ret, "%d" % int(idx), (int(x) + 5, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        detected_people = dict()
        for id_person, skeleton in projected_people.items():
            kps = dict()
            for j, joint in skeleton.items(): 
                if j in detected_joints.keys():
                    p2D = np.array(joint[1:3])
                    min_dist = 100000000
                    nearest = None
                    for i, coor in enumerate(detected_joints[j]):
                        d2D = np.array(coor)
                        dist = np.linalg.norm(p2D - d2D)
                        if dist < min_dist:
                            min_dist = dist
                            nearest = coor
                    if min_dist < 25:
                        kps[j] = [j, float(nearest[0]), float(nearest[1]), 1, 1]
            if kps:
                detected_people[id_person] = copy.deepcopy(kps)
                bodies_kps.append(kps)
                bodies_3D.append(joints_3D[id_person])

        kps_per_cam[camera_names[cam]] = [json.dumps(bodies_kps), time.time(), 'no_image', bodies_3D]

        if draw:
            for id_person, data in detected_people.items():
                for idx, joint in data.items():
                    x = joint[1]
                    y = joint[2]
                    cv2.circle(ret, (int(x), int(y)), 1, (0, 255, 0), 2)
                    cv2.putText(ret, "%d" % int(idx), (int(x) + 5, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

            cv2.imshow("test", cv2.resize(ret, dsize=None, fx=1, fy=1))
            if cv2.waitKey(0) % 256 == 27:
                print("Escape hit, closing...")
                cv2.destroyAllWindows()
                sys.exit(0)

    humans_json.append(dict(kps_per_cam))

output_file = open(seq_name.split('/')[-1] + '_from_image_multi.json', 'w')
output_file.write(json.dumps(humans_json))
output_file.close()
