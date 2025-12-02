<!-- <!-- # DeformSplat -->
<!-- <div align="center"> -->
<h1>DeformSplat</h1>

<a href="https://arxiv.org/abs/2509.22222"><img src="https://img.shields.io/badge/arXiv-2503.11651-b31b1b" alt="arXiv"></a>
<a href="https://vision3d-lab.github.io/deformsplat/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>

This is the official code for `Rigidity-Aware 3D Gaussian Deformation from a Single Image` represented on SIGGRAPH ASIA 25.

<p align="center">
  <img src="assets/teaser.gif" width="100%">
</p>

# 1. Enviroment setup
```
conda create -y --name deformsplat python=3.10
conda activate deformsplat

pip install torch==2.4.1+cu118 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ninja numpy jaxtyping rich
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu118
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
pip install --no-build-isolation git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5
pip install -r requirements.txt
 ```

(Optional) We recommand to login with wandb to visualize result.
```
wandb login
```

# 2. Run Diva360

## 2.1 Data Setup
Please download preprocessed diva360 data:

```
mkdir data
cd data

# diva360 data (~200MB)
gdown 1lRq22wxogt8x8TdCdd6AMZRieElrjO1w 
unzip diva360_processed.zip

cd ..
```

Then your file structure will be:

```
data
|-- diva360_processed
|   |-- penguin_0217
|   |-- penguin_0239
|   |-- blue_car_0142
|   |-- ...
```


(Optional) To test arbitrary frames other than those used in our paper, you can download the original dataset (~1.8TB).

<details>
<summary>Download Diva360 (~1.8TB) </summary>

Download the data from the link below and arrange the files as shown.  
- [Diva360 Download Link](https://www.dropbox.com/scl/fo/j68f78vlzt9q2z334294u/AETtMJY0yOOeHyby6SgJLhk/processed_data?rlkey=7e59p9e9ex8lakyuslrhg9afj&subfolder_nav_tracking=1&dl=0)


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

After Download original link, you can preprocess the data with run command (see 2.2).
## 2.2 Run

Please run the code below to test penguin example:
```
#                             GPU  object_name  frame_from  frame_to cam_idx
bash scripts/deform_diva360.sh 0   penguin      0217        0239     00     
```

If you want to run all Diva360 data from our [paper](https://arxiv.org/pdf/2509.22222), please run below:

```
bash scripts/run_all_diva360.sh
```

For executing other frames from our paper, please download the original data (see 2.1) and change `frame_from`, `frame_to`, and `cam_idx`.


# 3. Run DFA

## 3.1 Data Setup

Please download preprocessed DFA data:

```
cd data

gdown 1Qdu0s-gogtQKyAa3BZ8BrDMCk8kdzzCf
unzip DFA_processed.zip

cd ..
```

Then your file structure will be:

```
data
|-- DFA_processed
|   |-- beagle_dog(s1)
|   |-- cat(walk_final)
|   |-- duck(walk)
|   |-- lion(Walk)
|   |-- ...
```

(Optional) To test arbitrary frames other than those used in our paper, you can download the original dataset (~200GB).

<details>
<summary>Download original DFA (~200GB)</summary>

Download the data from the link below and arrange the files as shown.  
- [DFA Download Link](https://shanghaitecheducn-my.sharepoint.com/personal/luohm_shanghaitech_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fluohm%5Fshanghaitech%5Fedu%5Fcn%2FDocuments%2Fdatasets%2FArtemis%2Fdataset&ga=1) 

```
data
|-- dfa
|   |-- beagle_dog
|   |-- bear
|   |-- cat
|   |-- duck
|   |-- ...
```

After Download original link, you can preprocess the data with run command (see 3.2).

</details>

## 3.2 Run

Please run the code below to test wolf example:
```
#                          GPU       object_name     frame_from  frame_to cam_idx
bash scripts/deform_dfa.sh ${GPU}   "wolf(Howling)"  10          60       24
```

If you want to run all DFA data from our [paper](https://arxiv.org/pdf/2509.22222), please run below:

```
bash scripts/run_all_dfa.sh
```

For executing other frames from our paper, you can change `frame_from`, `frame_to`, and `cam_idx`.


# 🙏 Acknowledgements

This work builds upon the fantastic research and open-source contributions from the community. We extend our sincere thanks to the authors of the following projects:

- [gsplat](https://github.com/nerfstudio-project/gsplat)
- [RoMA](https://github.com/Parskatt/RoMa)
- [torch-splat](https://github.com/hbb1/torch-splatting)
- [Diva360](https://github.com/brown-ivl/DiVa360)
- [Artemis](https://github.com/HaiminLuo/Artemis)

# 📜 Citation

If you find this work helpful, please consider citing our paper:

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