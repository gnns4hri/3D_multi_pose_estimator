# Multi person and multi camara 3D pose estimator

Implementation of paper - [Multi-person 3D pose estimation from unlabelled data](Put a valid url here)

## Performance

| Threshold/mm |   25  |   50  |   75  |  100  |  125  |  150  |
|:------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|   Recall/%   | 12.96 | 92.08 | 97.50 | 98.70 | 99.26 | 99.49 |
|  Precision/% | 12.97 | 92.15 | 97.56 | 98.77 | 99.33 | 99.56 |
|     AP/%     |  3.31 | 87.06 | 96.08 | 97.96 | 98.83 | 99.20 |

| Times/ms  |       |
|:---------:|:-----:|
| $t_{pp}$  | 40.89 |
| ${t3Dg}$  | 36.13 |
| $t_{3Di}$ | 10.62 |

The Mean Per Joint Precision (MPJPE) is 36.66 mm. 

## Installation

To start using the system you just need to clone this repository on your local machine:

``` shell
git clone https://github.com/gnns4hri/3D_multi_pose_estimator.git
```

## Training

First step to train the system is to download the [Panoptic training dataset](url to the jsons here) 
and save it in the project directory.
Then the two networks (matching network and 3D estimator) can be trained separately:

#### Commands for training the skeleton matching network
``` shell
cd skeleton_matching
python3 train_skeleton_matching.py training_jsons dev_jsons
```

#### Commands for training the pose estimator network
``` shell
cd pose_estimator
python3 train_pose_estimator.py training_jsons dev_jsons
```

## Testing

You need to download the test set of [Panoptic dataset](url to the jsons here) and save it in the project directory.
Then it is necessary to have the two trained models in the root directory of the project.
Once everything is set up, from a terminal we move to the test directory:

``` shell
cd test
```

For getting the metrics and visualize the results from our model you will have to run this two scripts:

``` shell
python3 metrics_from_model.py test_files_path
python3 show_results_from_model.py test_files_path
```

On the other hand, if you want to check the results for only using the triangulation, the scripts are the following:

``` shell
python3 metrics_from_triangulation.py test_files_path
python3 show_results_from_triangulation.py test_files_path
```

## Citation

```
@article{,
  title={},
  author={},
  journal={},
  year={2022}
}
```