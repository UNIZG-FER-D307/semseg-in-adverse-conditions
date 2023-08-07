# Semantic Segmentation in Adverse Weather Conditions

## To Do
- [ ] Add technical report
- [ ] Update README.md
 - Add predictions generated with the best model and ensemble
 - Add results obtained with different architectures 
 - Add tables with final results and (pseudo)labeled data used
 - Add environment installation instructions
- [ ] Upload saved checkpoints
- [ ] Refactor code
 - Add `config.json` from vistas and create `datasets` folder (remove from `.gitgnore`)
 - Add singlescale/multiscale inference support 

## Environment Installation

Before running the acdc-semseg project, make sure you have [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) installed on your system. Follow the steps below to set up the required environment:

1. Create and activate the conda environment using Python 3.8:

   ```bash
   conda create --name acdc-semseg python=3.8
   conda activate acdc-semseg
   ```

2. Install the necessary packages using `pip`:

   ```bash
   pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
   pip install -r requirements.txt
   ```

3. Install additional third-party packages, [open_clip](https://github.com/mlfoundations/open_clip), and [timm](https://github.com/huggingface/pytorch-image-models) from their official repositories:

   ```bash
   git clone https://github.com/mlfoundations/open_clip.git
   cd open_clip
   pip install -e .
   cd ..

   git clone https://github.com/huggingface/pytorch-image-models.git
   cd pytorch-image-models
   pip install -e .
   cd ..
   ```

**Note:** The project has been developed and tested with CUDA 11.7, Pytorch 2.0, and Pytorch-Lightning 2.0.6. However, using Pytorch >=1.10 should also work fine.

 ## Running scripts
 ## Architecture Validation
 ## Final Results
 ## Visualizations

#### Snow
| Original Image | Single Model Prediction (ms) | Ensemble Prediction (ms) |
|:--------------:|:----------------------:|:------------------:|
| ![Image 1](/imgs/orig_imgs/snow/GP010176_frame_000643_rgb_anon.png) | ![Prediction 1](/imgs/pred_single/snow/GP010176_frame_000643_colors.png) | ![Ensemble 1](/imgs/pred_ens/snow/GP010176_frame_000643_colors.png) |
| ![Image 2](/imgs/orig_imgs/snow/GP010494_frame_000001_rgb_anon.png) | ![Prediction 2](/imgs/pred_single/snow/GP010494_frame_000001_colors.png) | ![Ensemble 2](/imgs/pred_ens/snow/GP010494_frame_000001_colors.png) |
| ![Image 3](/imgs/orig_imgs/snow/GP010176_frame_000418_rgb_anon.png) | ![Prediction 3](/imgs/pred_single/snow/GP010176_frame_000418_colors.png) | ![Ensemble 3](/imgs/pred_ens/snow/GP010176_frame_000418_colors.png) |
#### Rain
| Original Image | Single Model Prediction (ms) | Ensemble Prediction (ms) |
|:--------------:|:----------------------:|:------------------:|
| ![Image 4](/imgs/orig_imgs/rain/GOPR0573_frame_000246_rgb_anon.png) | ![Prediction 4](/imgs/pred_single/rain/GOPR0573_frame_000246_colors.png) | ![Ensemble 4](/imgs/pred_ens/rain/GOPR0573_frame_000246_colors.png) |
| ![Image 5](/imgs/orig_imgs/rain/GP020573_frame_000353_rgb_anon.png) | ![Prediction 5](/imgs/pred_single/rain/GP020573_frame_000353_colors.png) | ![Ensemble 5](/imgs/pred_ens/rain/GP020573_frame_000353_colors.png) |
| ![Image 6](/imgs/orig_imgs/rain/GP010571_frame_000784_rgb_anon.png) | ![Prediction 6](/imgs/pred_single/rain/GP010571_frame_000784_colors.png) | ![Ensemble 6](/imgs/pred_ens/rain/GP010571_frame_000784_colors.png) |
#### Night

| Original Image | Single Model Prediction (ms) | Ensemble Prediction (ms) |
|:--------------:|:----------------------:|:------------------:|
| ![Image 7](/imgs/orig_imgs/night/GOPR0355_frame_000162_rgb_anon.png) | ![Prediction 7](/imgs/pred_single/night/GOPR0355_frame_000162_colors.png) | ![Ensemble 7](/imgs/pred_ens/night/GOPR0355_frame_000162_colors.png) |
| ![Image 8](/imgs/orig_imgs/night/GOPR0356_frame_000162_rgb_anon.png) | ![Prediction 8](/imgs/pred_single/night/GOPR0356_frame_000162_colors.png) | ![Ensemble 8](/imgs/pred_ens/night/GOPR0356_frame_000162_colors.png) |
| ![Image 9](/imgs/orig_imgs/night/GP010594_frame_000034_rgb_anon.png) | ![Prediction 9](/imgs/pred_single/night/GP010594_frame_000034_colors.png) | ![Ensemble 9](/imgs/pred_ens/night/GP010594_frame_000034_colors.png) |
#### Fog
| Original Image | Single Model Prediction (ms) | Ensemble Prediction (ms) |
|:--------------:|:----------------------:|:------------------:|
| ![Image 10](/imgs/orig_imgs/fog/GP010475_frame_000300_rgb_anon.png) | ![Prediction 10](/imgs/pred_single/fog/GP010475_frame_000300_colors.png) | ![Ensemble 10](/imgs/pred_ens/fog/GP010475_frame_000300_colors.png) |
| ![Image 11](/imgs/orig_imgs/fog/GP010477_frame_000127_rgb_anon.png) | ![Prediction 11](/imgs/pred_single/fog/GP010477_frame_000127_colors.png) | ![Ensemble 11](/imgs/pred_ens/fog/GP010477_frame_000127_colors.png) |
| ![Image 12](/imgs/orig_imgs/fog/GP010478_frame_000167_rgb_anon.png) | ![Prediction 12](/imgs/pred_single/fog/GP010478_frame_000167_colors.png) | ![Ensemble 12](/imgs/pred_ens/fog/GP010478_frame_000167_colors.png) |

