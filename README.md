# Correspondences of the Third Kind: Camera Pose Estimation from Object Reflection

This repository provides an inplementation of our paper [Correspondences of the Third Kind: Camera Pose Estimation from Object Reflection](https://vision.ist.i.kyoto-u.ac.jp/research/3rdcorr/) in ECCV 2024. If you use our code and data please cite our paper.

Please note that this is research software and may contain bugs or other issues – please use it at your own risk. If you experience major problems with it, you may contact us, but please note that we do not have the resources to deal with all issues.

```
@InProceedings{Yamashita_2024_ECCV,
    author    = {Kohei Yamashita and Vincent Lepetit and Ko Nishino},
    title     = {{Correspondences of the Third Kind: Camera Pose Estimation from Object Reflection}},
    booktitle = {European Conference on Computer Vision (ECCV)},
    month     = {Oct},
    year      = {2024}
}
```

## Prerequisites

Please see ``./singularity/deepsharm.def`` for required modules. You can build your singularity container by
```
cd singularity
build.sh
```


## Usage

### Demo

Please download the pretrained models (weights.zip) from [here]() and unzip it in the root directory of this project. 

#### Experiments on synthetic images

1. Download the nLMVS-Synth dataset (Test set) from [here](https://github.com/kyotovision-public/nLMVS-Net) (or synthetic images of a cow model from [here]()). Update ``confs/test_depth_grid_nlmvss.json`` and ``confs/test_joint_opt_nlmvss.json`` according to the location of the downloaded data.

2. Run joint shape, camera pose, and reflectance map recovery:
```
Usage: python run_joint_est_nlmvss.py ${OBJECT_INDEX}
Example: python run_joint_est_nlmvss.py 152
```

The final results are saved to ``./run/test_shape_from_pose_nlmvss_3/${OBJECT_INDEX}``. ``plot_cams.py`` can visualize the camera pose results:
```
Usage: python plot_cams.py ${RESULT_PATH}
Example: python plot_cams.py ./run/test_shape_from_pose_nlmvss_3/152
```

#### Experiments on real image pairs

1. Download the real image data from [here](). Update ``confs/test_depth_grid_real.json`` and ``confs/test_joint_opt_real.json`` according to the location of the downloaded data.

2. Run joint shape, camera pose, and reflectance map recovery:
```
Usage: python run_joint_est_real.py ${OBJECT_INDEX}
Example: python run_joint_est_real.py 03
```

The final results are saved to ``./run/test_shape_from_pose_real_3/${OBJECT_INDEX}``.

(Optional) You can leverage DepthAnythingV2 as regularizer for the geometry recovery by adding ``--depth-anything`` flag
```
python run_joint_est_real.py --depth-anything 03
```
For this, please download the code and data of DepthAnythingV2 by using ``download_depth_anything_v2.sh``
```
bash download_depth_anything_v2.sh
```


#### Experiments on real image pairs of scenes with different objects

1. Download the real image data from [here](). Update ``confs/test_joint_opt_real_scene.json`` according to the location of the downloaded data.

2. Run joint shape, camera pose, and reflectance map recovery:
```
Usage: python run_joint_est_real_scene.py ${OBJECT_INDEX}
Example: python run_joint_est_real_scene.py 03
```

The final results are saved to ``./run/test_shape_from_pose_real_scene_3/${OBJECT_INDEX}``.

### Training

Please download the training data (rmap-fea-ext.zip) from [here]() and uncompress the zip file. We assume that the training data are in ``${HOME}/data/tmp``.

You can train the feature extractors for the correspondence detection using scripts in the ``training`` directory.

Training of the feature extractor for normal maps:
```
python training/train_normal_fea_ext.py
```

Training of the feature extractor for reflectance maps:
```
python training/train_rmap_fea_ext.py
```


### Empirical analysis of the number of correspondences required for the pose recovery

1. Generate synthetic correspondences:
```
python empirical_analysis/test_inv_gbr_data_gen.py
```

2. Run the experiments:
```
python empirical_analysis/run_test_inv_gbr.py
```

Evaluation:
```
python empirical_analysis/eval_test_inv_gbr.py
python empirical_analysis/plot_inv_gbr_results.py
```
