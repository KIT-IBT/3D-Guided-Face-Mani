# 3D-Guided Face Manipulation of 2D Images for the Prediction of Post-Operative Outcome after Cranio-Maxillofacial Surgery
This repository yields the code to the corresponding paper *3D-Guided Face Manipulation of 2D Images for the Prediction of Post-Operative Outcome after Cranio-Maxillofacial Surgery*  
(Submitted to IEEE Transactions on Image Processing). 

*Work in progress!!!*


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
cd ../PyTorch3DDFA/utils/cython
python setup.py build_ext -i
```
#### Face3D
```
cd ../../../face3d/face3d/mesh/cython/
python setup.py build_ext -i
```

This is a bit hacky: Create uncommited init.py's to let python see the
submodules as packages
```
cd ../../../../..
touch submodules/PyTorch3DDFA/__init__.py submodules/face3d/__init__.py
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

Download models from [here](LINK TO IEEE) and copy them to "data/configs"


### Download and create datasets
#### AFLW2000
Go to the [3DDFA website](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm) and download the AFLW2000-3D dataset. Then extract it into "data/datasets/AFLW2000"

#### 300W-LP (only used for training!)
Go to the [3DDFA website](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm) and download the 300W-LP dataset.
Then extract it into "data/datasets/300W_LP"

Next, create a small validation dataset to track the training progress:
```
mkdir data/datasets/300W_LP/validset data/datasets/300W_LP/validset_Flip 
mv data/datasets/300W_LP/LFPW/LFPW_image_test_000[0-4]* data/datasets/300W_LP/validset/
mv data/datasets/300W_LP/LFPW_Flip/LFPW_image_test_000[0-4]* data/datasets/300W_LP/validset_Flip/
```

Create dataset with PNCC renderings and shape modifications. This should run
without errors if you set up all configs and models correctly!
This might take up to several days. You can speed up this process by creating multiple runs
on different subdirectories of the 300W-LP (change "write_dataset.py"
accordingly.)

```
python write_dataset.py 300W-LP 
```

#### Render synthetic dataset (only used for training)
1. To use random background images, download the IndoorCVPR dataset from [here](http://web.mit.edu/torralba/www/indoor.html) and extract the folder into "data/datasets/Images"
2. Follow the instructions in "submodules/face3d/examples/Data/BFM/readme.md" (*Note: If you already downloaded the BFM installed the 3DDFA_Release (see above), then you can skip a few steps. The important thing is that you create the file "BFM.mat" that we need to render random textures from the BFM model*)
3. Copy the BFM file to the configs
```
cp submodules/face3d/examples/Data/BFM/Out/BFM.mat data/configs/
```

4. ToDo: Let user create stats_list
5. Render synthetic dataset (50k with chin modifications, 50k with nose modifications, 10k+10k with white background). This might take a few hours:
```
python render_synthetic_dataset.py
```


## Run
### Run inference
To test the model on the AFLW2000 dataset, you have multiple options

#### 1. Test different modifications
E.g. 'nose_1' or 'chin_1' and different scalar multipliers ('p', 'n', or a floating number to linearly scale the size of the chin e.g. '-200000'):
```
python run_inference.py data/datasets/AFLW2000/image00006.jpg -mult n -mod chin_1 -o data/output_1
python run_inference.py data/datasets/AFLW2000/image00006.jpg -mult p -mod chin_1 -o data/output_1
python run_inference.py data/datasets/AFLW2000/image00006.jpg -mult -200000 -mod chin_1 -o data/output_1
python run_inference.py data/datasets/AFLW2000/image00006.jpg -mult p -mod nose_1 -o data/output_1
```
The results can be seen in data/output_1 

*Note: The network works best for frontal faces combined with chin modifications.  
However, the results for nose modifications and large pose rotations are often
not very realistic or accurate. Feel free to try to come up with a better model by improving the
training strategy or the network architecture*


#### 2. Visualize "modification-grids" for all images of the AFLW2000 dataset
This will create a prediction grid for every image of the 2000 images in the AFLW2000
dataset
```
python run_inference.py data/datasets/AFLW2000 -o data/output_2
```

The prediction grid is structured as follows:  

| --- | --- | --- |
| sn+sc | sc | ln+sc |
| sn | None| ln |
| sn+lc | lc | ln+lc |
| --- | --- | --- |

with sn=smaller nose, ln=larger nose, sc=smaller chin, lc=larger chin.

# DOI
[![DOI](https://zenodo.org/badge/291075337.svg)](https://zenodo.org/badge/latestdoi/291075337)

# Citation
ToDo
