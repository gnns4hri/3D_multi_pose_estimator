import yaml

from collections import namedtuple

FORMAT = 'COCO'

if FORMAT == 'BODY_25':
    JOINT_LIST = [x for x in range(25)]
elif FORMAT == 'COCO':
    JOINT_LIST = [x for x in range(18)]
else:
    raise Exception('Format not set correctly in parameters.py!')

fields = (
    'image_width',
    'image_height',    
    'cameras',
    'camera_names',
    'fisheye',
    'widths',
    'heights',
    'fx',
    'fy',
    'cx',
    'cy',
    'r_s',
    'r_w',
    'c_s',
    'c_w',
    'kd0',
    'kd1',
    'kd2',
    'kd3',
    'p1',
    'p2',
    'joint_list',
    'numbers_per_joint',
    'numbers_per_joint_for_loss',
    'transformations_path',
    'used_cameras',
    'used_cameras_skeleton_matching',
    'used_joints',
    'min_number_of_views',
    'format',
    'graph_alternative',
    'axes_3D',
    'root',
    'temp'
)

TrackerParameters = namedtuple('TrackerParameters', fields, defaults=(None,) * len(fields))

# CONFIGURATION = 'PANOPTIC' # values = {PANOPTIC, ARPLAB}
#CONFIGURATION = '../ring/ring.yaml'
#CONFIGURATION = '../gym2/gym22.yaml'

def generate_tracker_parameters_from_file(config_path):
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return TrackerParameters(
        image_width = data["image_width"],
        image_height = data["image_height"],
        cameras = data["cameras"],
        camera_names = data["camera_names"],
        fisheye = data["fisheye"],
        fx = data["fx"],
        fy = data["fy"],
        cx = data["cx"],
        cy = data["cy"],
        kd0 = data["kd0"],
        kd1 = data["kd1"],
        kd2 = data["kd2"],
        kd3 = data["kd3"],
        p1 = data["p1"],
        p2 = data["p2"],
        joint_list = data["joint_list"],
        numbers_per_joint = data["numbers_per_joint"],
        numbers_per_joint_for_loss = data["numbers_per_joint_for_loss"],
        transformations_path = data["transformations_path"],
        used_cameras = data["used_cameras"],
        used_cameras_skeleton_matching = data["used_cameras_skeleton_matching"],
        used_joints = data["used_joints"],
        min_number_of_views = data["min_number_of_views"],
        format = data["format"],
        graph_alternative = data["graph_alternative"],
        root = data["root"],
        axes_3D = { key: tuple(value) for key, value in data["axes_3D"].items() }
    )

