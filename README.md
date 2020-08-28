# 3D-Guided Face Manipulation of 2D Images for the Prediction of Post-Operative Outcome after Cranio-Maxillofacial Surgery

*This repository is still work in progress!!!*

This repository contains the code and the download links to run the generator G on samples of the AFLW2000 dataset.

## Install
```
git clone https://github.com/KIT-IBT/3D-Guided-Face-Mani.git
```

### The models and parameters
The model and parameter configs files are storaged in /data/configs


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
