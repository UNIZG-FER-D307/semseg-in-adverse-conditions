# Semantic Segmentation in Adverse Conditions
|||
|:--------------:|:------------------:|
| ![Image 1](/imgs/orig_imgs/snow/GP010176_frame_000643_rgb_anon.png)  | ![Ensemble 1](/imgs/pred_ens/snow/GP010176_frame_000643_colors.png) |
| ![Image 4](/imgs/orig_imgs/rain/GOPR0573_frame_000246_rgb_anon.png) | ![Ensemble 4](/imgs/pred_ens/rain/GOPR0573_frame_000246_colors.png) |
| ![Image 7](/imgs/orig_imgs/night/GOPR0355_frame_000162_rgb_anon.png) | ![Ensemble 7](/imgs/pred_ens/night/GOPR0355_frame_000162_colors.png) |
| ![Image 10](/imgs/orig_imgs/fog/GP010475_frame_000300_rgb_anon.png)  | ![Ensemble 10](/imgs/pred_ens/fog/GP010475_frame_000300_colors.png) |


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

 ## Running
We provide three scripts:

- `SNCN_train_city.py`: This script is used for training SwiftNet+ConvNext on the `Cityscapes` training subset and evaluating it on the `Cityscapes` validation split.
- `generate_pseudolabels.py`: This script generates pseudolabels for unlabeled data, saving the generated pseudolabels in the `TODO` folder, along with colorized versions of the pseudolabels in the same directory.
- `generate_predictions.py`: This script generates predictions, saving the resulting segmentation maps in the `TODO` folder, along with colorized versions of the segmentation maps.

#### Training
To run training for ConvNeXt-tiny+SwiftNet-pyramid on the `Cityscapes` training subset and evaluate it on the `Cityscapes` validation subset (using GPUs with indices 0 and 1, with a batch size of 4 per GPU), execute the following command:

```bash
python SNCN_train_city.py -sv pyr -bv tiny --gpus 0 1 --batch_size_per_gpu 4
```

In `SNCN_train_city.py`, you can adjust other arguments as described in the script itself.

 #### Generating Pseudolabels
 #### Generating Predictions
 ## Experiments
 ### Architecture Validation

The training datasets we utilized to select the most suitable architecture are: *Cityscapes* (Cordts et al., 2016), *ACDC* (Sakaridis et al., 2021), *Wilddash2* (Zendel et al., 2022), and 50 annotated images from the *Dark Zurich* dataset (Sakaridis et al., 2019). We trained models on combination of these datasets and evaluated their performances on the validation subset of the *ACDC* dataset. Various backbone architectures we experimented with include: *Convnext* (Liu et al., 2022), *Convnext v2* (Woo et al., 2023), and *Swin Transformer* (Liu et al., 2021). For the upsampling path, we employed: *SwiftNet* with feature pyramid (*SwiftNet-pyr*), single-scale *SwiftNet* (*SwiftNet-ss*), and *UperNet* (Xiao et al., 2018).

##### Backbone size - Tiny

 | **Backbone**                        | **Upsampling Path** | **Parameters** | **mIoU (%)** |
|:----------------------------------|:---------------------|:--------------|:------------|
| **convnext-tiny-384-22k_1k**        | SwiftNet-pyr-256      | 33.3M          | 78.89        |
|                                    | SwiftNet-ss-256       | 30.9M          | **79.09**    |
|                                    | UperNet-256           | 37.8M          | 77.58        |
||
| **convnextv2-tiny-fcmae-384-22k_1k**| SwiftNet-pyr-256      | 32.5M          | 78.23        |
|                                    | SwiftNet-ss-256       | 30.2M          | 76.53        |
|                                    | UperNet-256           | 37.1M          | 77.68        |
||
| **swin-tiny-p4-w7-224-22k**         | SwiftNet-pyr-256      | 32.2M          | 77.03        |
|                                    | SwiftNet-ss-256       | 29.9M          | 76.65        |
|                                    | UperNet-256           | 36.7M          | 76.67        |

**Table 1:** Results obtained using the **tiny** versions of the backbones. The first three rows show results using the *Convnext* backbone pretrained on *ImageNet-22k* and fine-tuned on the *ImageNet-1k* dataset at a resolution of $384$, along with the corresponding upsampling path. The next three rows present results for the *Convnext v2* backbone, trained on the same datasets and resolution as *Convnext*. The last three rows display results obtained using the *Swin Transformer* pretrained on *ImageNet-22k* at a resolution of $224$. All experiments were conducted with a batch size of $10$, an initial learning rate of $4\mathrm{e}{-4}$ for the upsampling path, and an initial learning rate of $1\mathrm{e}{-4}$ for the backbone. The upsampling path dimension was set to $256$. Each experiment was run on two NVIDIA RTX A4500 graphics cards with 20GiB of available memory. Models are evaluated on the ACDC validation set.

##### Backbone size - Base

The second experiment we conducted involved training the *base* versions of the same backbones used in the previous experiment. Additionally, apart from the backbones pretrained on the *ImageNet* dataset, we carried out experiments with the *Convnext* backbone pretrained on the *LAION-5B* subset (Schuhmann et al., 2022). Furthermore, we also tested the *Convnext* backbone pretrained on both the *LAION-5B* subset and the *ImageNet-12k* dataset ([repo](https://github.com/rwightman/imagenet-12k)), fine-tuned on the *ImageNet-1k* dataset. Pretrained parameter values for the *Convnext* pretrained on the *LAION-5B* subset are available in the repository ([open_clip repo](https://github.com/mlfoundations/open_clip)), while the weights for the backbone that was additionally fine-tuned on the *ImageNet* dataset are accessible through the programming library `timm`.

| **Backbone**                                | **Upsampling Path** | **Parameters** | **mIoU (%)** |
|:------------------------------------------| ------------------- | -------------- | ------------ |
| *convnext-base-384-22k_1k*                  | SwiftNet-pyr-256    | 93.6M          | **81.88**    |
| *convnext-base-384-22k_1k*                  | SwiftNet-ss-256     | 91.1M          | 80.33        |
| *convnextv2-base-fcmae-384-22k_1k*          | SwiftNet-pyr-256    | 92.7M          | 80.72        |
| *convnext-base-laiona-augreg-384-12k_1k*    | SwiftNet-pyr-256    | 93.6M          | 79.95        |
| *convnext-base-laiona-augreg-320*           | SwiftNet-pyr-256    | 93.6M          | 80.60        |
| *swin-base-p4-w12-384-22k*                  | SwiftNet-pyr-256    | 91.9M          | 79.52        |
| *swin-base-p4-w12-384-22k_lr/10*            | SwiftNet-pyr-256    | 91.9M          | 79.68        |
| *swin-base-p4-w12-384-22k*                  | UperNet-512         | 120M           | 79.52        |

**Table 2:** Results obtained using the **base** versions of the backbones. The first column represents the used backbone. The first part of the name in the first column denotes the backbone version, while the remainder represents the dataset and image resolution during training. For example, '*convnext-base-laiona-augreg-384-12k_1k*' corresponds to an experiment with the *base* version of the *Convnext* backbone. This backbone was pretrained on the subset of the *LAION-5B* dataset and on *ImageNet-12k*, fine-tuned on the *ImageNet-1k* dataset, with a training image resolution of $384$. All experiments were conducted with a batch size of $12$, an initial learning rate of $4\mathrm{e}{-4}$ for the upsampling path, and an initial learning rate of $1\mathrm{e}{-4}$ for the backbone. The upsampling path dimension was $256$, except in the experiment using *UperNet*, where the dimension was $512$. Each experiment was run on 4 NVIDIA Tesla V100 GPUs with 32GiB of available memory. Models are evaluated on the ACDC validation set.

##### Backbone size - Large


 ### Final Results
 ### Visualizations

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

