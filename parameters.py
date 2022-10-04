import sys

from collections import namedtuple

FORMAT = 'COCO'  # 'COCO'  'BODY_25'

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

USE_PANOPTIC = True

TrackerParameters = namedtuple('TrackerParameters', [
    'tag_size',
    'cameras',
    'camera_names',
    'fx',
    'fy',
    'cx',
    'cy',
    'kd0',
    'kd1',
    'kd2',
    'p1',
    'p2',
    'joint_list',
    'numbers_per_joint',
    'transformations_path',
    'used_cameras',
    'used_joints',
    'no3d',
    'draw',
    'format',
    'neck_id',
    'leftshoulder_id',
    'rightshoulder_id',
    'tracker_sends_only_one_skeleton',
    'graph_alternative',
    'camera_colours',
    'old_data_to_remove',
    'image_width',
    'image_height'
])

#
#  PARAMETERS
#
if USE_PANOPTIC:
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
        numbers_per_joint=14,  # 1 (joint detected?) 2 (x)  3 (y)  4 (over the threshold?)  5 (certainty)
        transformations_path='../tm_panoptic.pickle',
        used_cameras=['trackera', 'trackerb', 'trackerc', 'trackerd', 'trackere'],
        used_joints = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        no3d=True,
        draw=True,
        format=FORMAT,
        neck_id=NECK_ID,
        leftshoulder_id=LEFTSHOULDER_ID,
        rightshoulder_id=RIGHTSHOULDER_ID,
        tracker_sends_only_one_skeleton=True,
        graph_alternative='3',
        camera_colours={'trackera': (255, 0, 0), 'trackerb': (0, 255, 0), 'trackerc': (0, 0, 255),
                        'trackerd': (127, 127, 0), 'trackere': (0, 127, 127)},
        old_data_to_remove=0.3
    )
else:
    F = 848. / 1280.
    parameters = TrackerParameters(
        image_width=848,
        image_height=480,
        cameras=[0, 1, 2, 3],
        camera_names=['trackera', 'trackerb', 'trackerc', 'trackerd'],
        tag_size=0.452,  # 0.313,
        # fx  =              [424.86101478, 424.08621444, 425.35061892, 425.43102505],
        # fy  =              [425.31913919, 424.35178743, 425.92296923, 425.84745869],
        # cx =               [420.07507557, 420.53503877, 421.15377069, 420.93675768],
        # cy =               [237.26322826, 239.09107219, 245.26624895, 240.66948926],
        kd0=[0., 0., 0., 0.],
        kd1=[0., 0., 0., 0.],
        kd2=[0., 0., 0., 0.],
        p1=[0., 0., 0., 0.],
        p2=[0., 0., 0., 0.],

        fx=[634.0370 * F, 633.6757 * F, 636.5411 * F, 635.4050 * F],
        fy=[633.5662 * F, 633.0649 * F, 636.1349 * F, 634.5941 * F],
        cx=[631.7626 * F, 635.7685 * F, 638.4467 * F, 638.3454 * F],
        cy=[355.3067 * F, 358.7285 * F, 370.3130 * F, 362.9503 * F],

        joint_list=JOINT_LIST,
        numbers_per_joint=14,  # 1 (joint detected?) 2 (x)  3 (y)  4 (over the threshold?)  5 (certainty)
        transformations_path='../tm.pickle',
        used_cameras=['trackera', 'trackerb', 'trackerc', 'trackerd'],
        used_joints = [x for x in range(18)],
        no3d=True,
        draw=True,
        format=FORMAT,
        neck_id=NECK_ID,
        leftshoulder_id=LEFTSHOULDER_ID,
        rightshoulder_id=RIGHTSHOULDER_ID,
        tracker_sends_only_one_skeleton=True,
        graph_alternative='3',
        camera_colours={'trackera': (255, 0, 0), 'trackerb': (0, 255, 0), 'trackerc': (0, 0, 255),
                        'trackerd': (127, 127, 0)},
        old_data_to_remove=0.3
    )

#
#  ASSERTS
#
assert len(parameters.cameras) == len(
    parameters.camera_names), "The number of cameras must be equal in 'cameras' and 'camera_names'"
assert NECK_ID in parameters.joint_list, f"Joint {NECK_ID} (neck) is mandatory"
