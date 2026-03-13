rm -f ./models/skeleton_matching.tch
rm -f ./models/pose_estimator.pytorch
rm -f ./skeleton_matching/skeleton_matching.tch
rm -f ./pose_estimator.pytorch

find /home/ljmanso/Nextcloud/Gatis/ -name \*ch -exec rm {} \;

find . -name Merged\* -exec rm {} \;


