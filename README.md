# 3D-Guided Face Manipulation of 2D Images for the Prediction of Post-Operative Outcome after Cranio-Maxillofacial Surgery
*This repository is still work in progress!!!*

This repository contains the code and the download links to run the generator G on samples of the AFLW2000 dataset.

## Requirements
To install and run our scripts you need:
	- Matlab (If you don't have a licence, there might be a workaround (see below))
	- A python environment like anaconda
	- A Nvidia GPU with CUDA/PyTorch (If you want to train the model, you might need up to 11GB VRAM; 8GB might work too if you lower the batch-size)
	- Time to install and download all the submodules, data and config files that this repo requires :)


## Install
### Install repo and packages
Choose a directory to install this repo. Then:
```
git clone --recurse-submodules https://github.com/KIT-IBT/3D-Guided-Face-Mani.git
cd 3D-Guided-Face-Mani
```
Set up python path:
```
export PYTHONPATH=$PYTHONPATH$:$(pwd)/3D-Guided-Face-Mani
```

Install packages. This repo was tested using the following versions. However, more recent versions might just work fine, too.
```
conda create -n face-mani python=3.7
conda activate face-mani

pip install numpy==1.17.0 \
	   	scipy==1.3.0 \
		tqdm==4.34.0 \
		opencv-python==4.1.0.25 \
		cython==0.29.12 \
		scikit-image==0.15.0

```
Install PyTorch via conda as described [here](https://pytorch.org/). Tested version: PyTorch==1.6.0, Cuda==10.2, torchvision=0.7.0
Example:
```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```


### Install submodules
#### 3DDFA_Release
Go to the [3DDFA website](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm) and download the 3DDFA code and extract it.
Then, copy the folder to "submodules" e.g.
```
cp -R ~/Downloads/3DDFA_Release submodules/
cd submodules/3DDFA_Release
```
1. Downloading the Basel Face Model (BFM) on "http://faces.cs.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model"                                                                                       
2. Copy the "01_MorphableModel.mat" file of the BFM to Matlab/ModelGeneration/  
3. Run the Matlab/ModelGeneration/ModelGenerate.m to generate the shape model "Model_Shape.mat"
*Note: If you don't have a Matlab licence, you might consider re-writing the ModelGenerate.m file in python (script has only a few lines)!*

#### 3DDFA
*Note: 3DDFA and Face3D share identical code for rendering and 3DMM processing*
```
mv submodules/3DDFA submodules/PyTorch3DDFA
```

```
cd ../3DDFA/utils/cython
python setup.py build_ext -i
```
#### Face3D
```
cd ../../../face3d/face3d/mesh/cython/
python setup.py build_ext -i
```

### Models and configs
The model and parameter configs files are storaged in /data/configs
```
cd ../../../../..
cp submodules/3DDFA_Release/Matlab/ModelGeneration/Model_Shape.mat \
	submodules/3DDFA_Release/Matlab/Model_Expression.mat \
	submodules/3DDFA/visualize/tri.mat \
   data/configs/
```


### Download datasets
Go to the [3DDFA website](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm) and download the AFLW2000-3D dataset. Then extract it into "data/datasets/AFLW2000"

# Run
### Create AFLW2000 dataset
```
python write_dataset.py
```

### Run inference
```
python inference.py <path to image.jpg> <chin_1 or nose_1> <n or p>
```
# DOI
[![DOI](https://zenodo.org/badge/291075337.svg)](https://zenodo.org/badge/latestdoi/291075337)

# Citation

<!---
submodules: 
- face3d
- 3DDFA-Pytorch

paste into configs:
- FaceProfilingRelease_v1.1/ModelGeneration/Model_Shape.mat
- FaceProfilingRelease_v1.1/Model_Expression.mat
- PyTorch_3DDFA/visualize/tri.mat
- shape_mods
-->
# 3D-Face-Manipu
