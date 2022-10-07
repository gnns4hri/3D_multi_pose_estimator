# Multi person and multi camara 3D pose estimator

Implementation of the paper - [Multi-person 3D pose estimation from unlabelled data](Put a valid url here)

## Performance

| Threshold/mm |   25  |   50  |   75  |  100  |  125  |  150  |
|:------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|   Recall/%   | 12.96 | 92.08 | 97.50 | 98.70 | 99.26 | 99.49 |
|  Precision/% | 12.97 | 92.15 | 97.56 | 98.77 | 99.33 | 99.56 |
|     AP/%     |  3.31 | 87.06 | 96.08 | 97.96 | 98.83 | 99.20 |

| Times/ms  |       |
|:---------:|:-----:|
| $t_{pp}$  | 40.89 |
| $t_{3Dg}$  | 36.13 |
| $t_{3Di}$ | 10.62 |

The Mean Per Joint Precision (MPJPE) is 36.66 mm. 

## Installation

To start using the system you just need to clone this repository on your local machine:

``` shell
git clone https://github.com/gnns4hri/3D_multi_pose_estimator.git
```
Install the dependencies in *requirements.txt*

## Dataset generation

The first step to train the system is to generate the dataset. If you want to train the models using the [CMU Panoptic dataset](http://domedb.perception.cs.cmu.edu/), download the sequences following the instructions of the [PanopticStudio Toolbox](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox).
Use the Panoptic toolbox for uncompressing the HD images of the sequences (only the images from the HD cameras 3, 6, 12, 13 and 23 are required).
Once the images have been uncompressed, you can generate a json file for each sequence using the scripts in *panoptic_conversor*.
For that,download the backbone model for CMU Panoptic dataset available from the [VoxelPose project](https://github.com/microsoft/voxelpose-pytorch) and place it in the *panoptic_conversor* directory.
To generate each training json file, run the following commands:

``` shell
cd panoptic_conversor
python3 get_joints_from_panoptic_model.py PANOPTIC_SEQUENCE_DIRECTORY
```

To generate a json file for test, run *get_joints_from_panoptic_model_multi.py* instead of the previous script. Both scripts generate a json file and a Python pickle file with the transformation matrices of the cameras. This pickle file is necessary for obtaining metrics of the models.
 
## Training

Once the dataset has been generated,  the two networks (matching network and 3D estimator) can be trained separately:

#### Commands for training the skeleton matching network
``` shell
cd skeleton_matching
python3 train_skeleton_matching.py training_jsons dev_json
```
The *dev_json* file can be created from several json files using the script *merge_jsons.py* in *utils*.

#### Commands for training the pose estimator network
``` shell
cd pose_estimator
python3 train_pose_estimator.py training_jsons 
```

## Testing

You need to download the test set of [Panoptic dataset](url to the jsons here) (json and pickle files) and save it in the project directory.
Then it is necessary to have the two trained models in a directory called *models* of the root directory of the project.
Once everything is set up, from a terminal we move to the test directory:

``` shell
cd test
```

For getting the metrics and visualize the results from our model you will have to run this two scripts:

``` shell
python3 metrics_from_model.py test_files test_tm_files_directory
python3 show_results_from_model.py test_files
```

On the other hand, if you want to check the results using triangulation instead of the pose estimation model, the scripts are the following:

``` shell
python3 metrics_from_triangulation.py test_files tm_files_directory
python3 show_results_from_triangulation.py test_files
```

The four scripts can be run with more than one json file.

## Citation

```
@article{,
  title={},
  author={},
  journal={},
  year={2022}
}
```