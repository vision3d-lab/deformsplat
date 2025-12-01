<!-- <!-- # DeformSplat -->
<!-- <div align="center"> -->
<h1>DeformSplat</h1>

<a href="https://arxiv.org/pdf/2509.22222"><img src="https://img.shields.io/badge/arXiv-2503.11651-b31b1b" alt="arXiv"></a>
<a href="https://vision3d-lab.github.io/deformsplat/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>

This is the official code for `Rigidity-Aware 3D Gaussian Deformation from a Single Image` represented on SIGGRAPH ASIA 25.

<!-- </div> -->

<p align="center">
  <video src="teaser.mp4" controls autoplay loop muted width="100%"></video>
</p>

# 1. Enviroment setup
```
conda create --name deformsplat python=3.10
conda deformsplat deformsplat

pip install torch==2.4.1+cu118 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ninja numpy jaxtyping rich
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu118
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
pip install -r requirements.txt
 ```

(Optional) We recommand to login with wandb to visualize result.
```
wandb login
```

# 2. Data

## 2.1 Download preprocessed data
Plase download preprocessed diva360 and DFA data:

```
mkdir data && cd data

# diva360
gdown 1lRq22wxogt8x8TdCdd6AMZRieElrjO1w && unzip "*.zip"

# dfa
gdown  && unzip "*.zip"

cd ..

```

Then your file will be like below:

```
data
|-- diva360_processed
|   |-- penguin_0217
|   |-- penguin_0239
|   |-- blue_car_0142
|   |-- ...
|
|-- dfa_processed

```

## 2.2 (Optional) Download original data 


Or you can download original dataset and preprocess it.

Then, you can test arbitrary frame other than those used in the paper.

<details>
<summary>Download Diva360 (~1.8TB) </summary>

Download the data from the link below and arrange the files as shown.  
[Download Link](https://www.dropbox.com/scl/fo/j68f78vlzt9q2z334294u/AETtMJY0yOOeHyby6SgJLhk/processed_data?rlkey=7e59p9e9ex8lakyuslrhg9afj&subfolder_nav_tracking=1&dl=0)

```
data
|-- diva360
|   |-- penguin
|   |   |-- frames_1
|   |   |-- segmented_gt
|   |   |-- segmented_ngp
|   |   |-- transforms_circle.json
|   |   |-- ...
|   |   
|   |-- blue_car
|   |-- ...
```

</details>

<details>
<summary>Download DFA (~200GB)</summary>

Download the data from the link below and arrange the files as shown.  
[Download Link](https://www.dropbox.com/scl/fo/j68f78vlzt9q2z334294u/AETtMJY0yOOeHyby6SgJLhk/processed_data?rlkey=7e59p9e9ex8lakyuslrhg9afj&subfolder_nav_tracking=1&dl=0)

```
data
|-- diva360
|   |-- penguin
|   |   |-- frames_1
|   |   |-- segmented_gt
|   |   |-- segmented_ngp
|   |   |-- transforms_circle.json
|   |   |-- ...
|   |   
|   |-- blue_car
|   |-- ...
```

</details>

. 


# 3. Run DeformSplat
## 3.1 Run with Diva360

Please run the code below to test penguin example:
```
cd deformsplat
#                             GPU  object_name  frame_from  frame_to cam_idx
bash scripts/deform_diva360.sh 0   penguin      0217        0239     00     
```


If you want to run all Diva360 data from the [paper](https://arxiv.org/pdf/2509.22222), plase run below:

```
bash scripts/run_all_diva360.sh
```

For executing other frames from our paper, please download the original data (see 2.2) and change `frame_from`, `frame_to`, and `cam_idx`.


## 3.2 Run with  DFA 

If you want to run all data in Diva360, please execute below
```
cd gsplat
#                             GPU  object_name  frame_from  frame_to cam_idx
bash scripts/deform_diva360.sh 0   penguin      0217        0239     00     
```


If you want to run all data in Diva360, plase run below
```
bash scripts/run_all_diva360.sh
```

# 4. Acknowledge



# 📝 Citations

```
@misc{kim2025deformsplat,
    title={Rigidity-Aware 3D Gaussian Deformation from a Single Image}, 
    author={Jinhyeok Kim and Jaehun Bang and Seunghyun Seo and Kyungdon Joo},
    year={2025},
    eprint={2509.22222},
    archivePrefix={arXiv},
    primaryClass={cs.GR},
    url={https://arxiv.org/abs/2509.22222}, 
}
```