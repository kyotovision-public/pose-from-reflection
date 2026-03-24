# DeepShaRM: Multi-View Shape and Reflectance Map Recovery Under Unknown Lighting

This repository provides an implementation of our paper DeepShaRM: Multi-View Shape and Reflectance Map Recovery Under Unknown Lighting in 3DV 2024. If you use our code and data please cite our paper.

Please note that this is research software and may contain bugs or other issues – please use it at your own risk. If you experience major problems with it, you may contact us, but please note that we do not have the resources to deal with all issues.

```
@InProceedings{Yamashita_2024_3DV,
    author    = {Kohei Yamashita and Shohei Nobuhara and Ko Nishino},
    title     = {{DeepShaRM: Multi-View Shape and Reflectance Map Recovery Under Unknown Lighting}},
    booktitle = {International Conference on 3D Vision (3DV)},
    month     = {Mar},
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

### Inference
Please download the pretrained models (weights.zip) from [here](https://drive.google.com/drive/folders/1ipsx8mZ9ZEf_x4sGVfEwIaor0hm8sgAs?usp=sharing) and unzip them in the root directory of this project. 

Please also download the nLMVS-Synth and nLMVS-Real datasets from [here](https://github.com/kyotovision-public/nLMVS-Net). Update config files in the ``confs`` dir according to the location of the downloaded data.

#### Multiview shape and reflectance map recovery from posed images

Experiments on nLMVS-Synth:
```
Example: python deepsharm/optimize_sdf.py --config ./confs/deepsharm_nlmvss.json ${OBJECT_INDEX}
Usage: python deepsharm/optimize_sdf.py --config ./confs/deepsharm_nlmvss.json 152
```

Experiments on nLMVS-Real:
```
Example: python deepsharm/optimize_sdf.py --config ./confs/deepsharm_nlmvsr10.json ${OBJECT_INDEX}
Usage: python deepsharm/optimize_sdf.py --config ./confs/deepsharm_nlmvsr10.json 91
```

Experiments on nLMVS-Real (5 views):
```
Example: python deepsharm/optimize_sdf.py --config ./confs/deepsharm_nlmvsr5nv.json ${OBJECT_INDEX}
Usage: python deepsharm/optimize_sdf.py --config ./confs/deepsharm_nlmvsr5nv.json 91
```

#### Reflectance map estimation from an image and a normal map

```
python deepsharm/test_rm_net.py 152
```

### Training
Please download the training data from [here]() and uncompress the zip file. We assume the training data are in ``${HOME}/data/tmp``.

You can train the networks (SfS-Net and RM-Net) by using the following scripts.

Training of SfS-Net (surface normal estimation):
```
python deepsharm/train_sfsnet.py
```


Training of RM-Net (reflectance map estimation):
```
python deepsharm/train_rm_net.py
```