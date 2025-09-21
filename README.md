# DeformSplat
This is the official code for `Rigidity-Aware 3D Gaussian Deformation from a Single Image`.

# Enviroment setup
```
conda create --name deformsplat python=3.10
conda deformsplat gsplat

pip install torch==2.4.1+cu118 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ninja numpy jaxtyping rich
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu118
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

pip install -r requirements.txt
 
wandb login

```

# Command
## Run DeformSplat (Diva360)

```
cd gsplat
#                             GPU  object_name  frame_from  frame_to cam_idx  wandb_group
bash scripts/deform_diva360.sh 0   penguin      0217        0239     00       DeformSplat
```


If you want to run all data in Diva360,
```
bash scripts/run_all_diva360.sh
```