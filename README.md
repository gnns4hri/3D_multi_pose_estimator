# Multi person and multi camara 3D pose estimator

Implementation of the paper - [Multi-person 3D pose estimation from unlabelled data](https://arxiv.org/abs/2212.08731)

## Performance

| Threshold/mm |   25  |   50  |   75  |  100  |  125  |  150  |
|:------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|   Recall/%   | 36.72 | 96.33 | 98.26 | 99.20 | 99.43 | 99.53 |
|  Precision/% | 36.75 | 96.41 | 98.34 | 99.28 | 99.51 | 99.61 |
|     AP/%     |  18.66 | 93.82 | 97.13 | 98.71 | 99.12 | 99.27 |

| Times/ms  |       |
|:---------:|:-----:|
| $t_{pp}$  (time for persons' proposal) | 40.89 |
| $t_{3Dg}$  (time for 3D pose estimation)| 36.13 |
| $t_{3Di}$ (time for 3D pose estimation per human)| 10.62 |

The Mean Per Joint Precision (MPJPE) is 29.79 mm. 

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

You need to download the test set of [Panoptic dataset](https://www.dropbox.com/sh/9nqgf58foh7f2h0/AAC8iT_VWHKbJDw-UYaS0Emqa?dl=0) (json and pickle files) and save it in the project directory.
Then it is necessary to have the two trained models in a directory called *models* of the root directory of the project.
The files containing our trained models can be downloaded from [here](https://www.dropbox.com/sh/0fkfe5vvtex9zaa/AACvrfrTDaGgDxAWCJi-lTBna?dl=0).
Once everything is set up, from a terminal, move to the test directory:

``` shell
cd test
```

For getting the metrics and visualize the results from our model you will have to run these two scripts:

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
@misc{https://doi.org/10.48550/arxiv.2212.08731,
  doi = {10.48550/ARXIV.2212.08731},
  
  url = {https://arxiv.org/abs/2212.08731},
  
  author = {Rodriguez-Criado, Daniel and Bachiller, Pilar and Vogiatzis, George and Manso, Luis J.},
    
  title = {Multi-person 3D pose estimation from unlabelled data},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}

```