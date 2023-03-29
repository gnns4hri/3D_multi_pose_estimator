import sys

from collections import namedtuple

FORMAT = 'COCO'

if FORMAT == 'BODY_25':
    NECK_ID = 1
    LEFTSHOULDER_ID = 5
    RIGHTSHOULDER_ID = 2
    JOINT_LIST = [x for x in range(25)]
elif FORMAT == 'COCO':
    NECK_ID = 17
    LEFTSHOULDER_ID = 5
    RIGHTSHOULDER_ID = 6
    JOINT_LIST = [x for x in range(18)]
else:
    raise Exception('Format not set correctly in parameters.py!')

fields = (
    'tag_size',
    'cameras',
    'camera_names',
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
    'no3d',
    'draw',
    'format',
    'neck_id',
    'leftshoulder_id',
    'rightshoulder_id',
    'dataset_generator_records_only_one_skeleton',
    'graph_alternative',
    'camera_colours',
    'old_data_to_remove',
    'image_width',
    'image_height'
)

TrackerParameters = namedtuple('TrackerParameters', fields, defaults=(None,) * len(fields))

CONFIGURATION = 'ARPLAB' # values = {PANOPTIC, ARPLAB}

#
#  PARAMETERS
#
if CONFIGURATION == 'PANOPTIC':
    parameters = TrackerParameters(
        image_width=1920,
        image_height=1080,
        tag_size=0.452,  # before 0.313,
        cameras=[0, 1, 2, 3, 4],
        camera_names=['trackera', 'trackerb', 'trackerc', 'trackerd', 'trackere'],
        fx=[1395.59, 1395.94, 1395.31, 1591.32, 1572.31],
        fy=[1392.03, 1392.22, 1391.77, 1587.2, 1567.51],
        cx=[950.046, 950.459, 966.65, 940.617, 942.938],
        cy=[564.906, 547.877, 562.988, 560.913, 559.888],
        kd0=[-0.28619, -0.279874, -0.284888, -0.232872, -0.237061],
        kd1=[0.179547, 0.166215, 0.179936, 0.194125, 0.18403],
        kd2=[-0.0451919, -0.035049, -0.0468637, 0.0125375, 0.0149481],
        p1=[-0.00010526, -0.000189415, -0.000119731, 4.22e-05, -0.000448556],
        p2=[6.45495e-05, 0.00107791, 0.000701704, 0.000877748, 0.00062731],
        joint_list=JOINT_LIST,
        numbers_per_joint=14, 
        numbers_per_joint_for_loss=4,
        transformations_path='../tm_panoptic.pickle',
        used_cameras=['trackera', 'trackerb', 'trackerc', 'trackerd', 'trackere'],
        used_cameras_skeleton_matching=['trackera', 'trackerb', 'trackerc', 'trackerd', 'trackere'],
        used_joints = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        min_number_of_views = 2,
        no3d=True,
        draw=True,
        format=FORMAT,
        neck_id=NECK_ID,
        leftshoulder_id=LEFTSHOULDER_ID,
        rightshoulder_id=RIGHTSHOULDER_ID,
        graph_alternative='3',
        camera_colours={'trackera': (255, 0, 0), 'trackerb': (0, 255, 0), 'trackerc': (0, 0, 255),
                        'trackerd': (127, 127, 0), 'trackere': (0, 127, 127)},
        old_data_to_remove=0.3
    )
elif CONFIGURATION == 'ARPLAB':
    F = 848. / 1280.
    ZEN_F = 720. / 1080.
    parameters = TrackerParameters(
        image_width=1280,
        image_height=720,
        cameras=[0, 1, 2, 3, 4, 5],
        camera_names=['trackera', 'trackerb', 'trackerc', 'trackerd', 'orinbot_l', 'orinbot_r'], 
        tag_size=0.452,
        kd0=[0., 0., 0., 0., 0., 0., 0.],
        kd1=[0., 0., 0., 0., 0., 0., 0.],
        kd2=[0., 0., 0., 0., 0., 0., 0.],
        p1=[0., 0., 0., 0., 0., 0., 0.],
        p2=[0., 0., 0., 0., 0., 0., 0.],

        widths=[1280,        1280,              1280,            1280,      1280,          1280],
        heights=[720,         720,               720,             720,      720,           720],
        fx=[634.0370 * F, 633.6757 * F, 636.5411 * F, 635.4050 * F, 1097.2998046875*ZEN_F,   1097.2998046875*ZEN_F],
        fy=[633.5662 * F, 633.0649 * F, 636.1349 * F, 634.5941 * F, 1097.2998046875*ZEN_F,   1097.2998046875*ZEN_F],
        cx=[631.7626 * F, 635.7685 * F, 638.4467 * F, 638.3454 * F, 953.3253173828125*ZEN_F, 953.3253173828125*ZEN_F],
        cy=[355.3067 * F, 358.7285 * F, 370.3130 * F, 362.9503 * F, 553.707763671875*ZEN_F,  553.707763671875*ZEN_F],

        r_s=[0,                0,                 0,               0,         0,             0],
        r_w=[720,            720,               720,             720,       720,           720],
        c_s=[0,                0,                 0,              0,         0,              0],
        c_w=[1280,          1280,              1280,            1280,       1280,          1280],

        joint_list=JOINT_LIST,
        numbers_per_joint=14,  # 1 (joint detected?) 2 (x)  3 (y)  4 (over the threshold?)  5 (certainty)
        numbers_per_joint_for_loss=4,  # 1 (joint detected?) 2 (x)  3 (y)  4 (over the threshold?)  5 (certainty)
        transformations_path='../tm_new.pickle',
        used_cameras=['trackera', 'trackerb', 'trackerc', 'trackerd', 'orinbot_l', 'orinbot_r'],
        used_cameras_skeleton_matching = ['trackera', 'trackerb', 'trackerc', 'trackerd', 'orinbot_l', 'orinbot_r'],
        used_joints = [x for x in range(18)],
        min_number_of_views = 1,
        no3d=True,
        draw=True,
        format=FORMAT,
        neck_id=NECK_ID,
        leftshoulder_id=LEFTSHOULDER_ID,
        rightshoulder_id=RIGHTSHOULDER_ID,
        dataset_generator_records_only_one_skeleton=False,
        graph_alternative='3',
        camera_colours={'trackera': (255, 0, 0), 'trackerb': (20, 150, 20), 'trackerc': (0, 255, 0), 'trackerd': (0, 0, 255),
                        'orinbot_l': (0, 100, 160), 'orinbot_r': (0, 160, 100)},
        old_data_to_remove=0.08
    )
else:
    print('NO VALID CONFIGURATION')
    exit()

#
#  ASSERTS
#
assert len(parameters.cameras) == len(
    parameters.camera_names), "The number of cameras must be equal in 'cameras' and 'camera_names'"
assert NECK_ID in parameters.joint_list, f"Joint {NECK_ID} (neck) is mandatory"
