# Semantic Segmentation in Adverse Conditions
[![ACDC](https://img.shields.io/badge/Ranked_%234%20-ACDC_Supervised_Semantic_Segmentation-blue)](https://acdc.vision.ee.ethz.ch/benchmarks#semanticSegmentation)
[![ACDC](https://img.shields.io/badge/Ranked_%232%20-ACDC_Uncertainty_Aware_Semantic_Segmentation-blue)](https://acdc.vision.ee.ethz.ch/benchmarks#uncertaintyAwareSemanticSegmentation) 

|||
|:--------------:|:------------------:|
| ![Image 1](./docs/imgs/orig_imgs/snow/GP010176_frame_000643_rgb_anon.png)  | ![Ensemble 1](./docs/imgs/pred_ens/snow/GP010176_frame_000643_colors.png) |
| ![Image 4](./docs/imgs/orig_imgs/rain/GOPR0573_frame_000246_rgb_anon.png) | ![Ensemble 4](./docs/imgs/pred_ens/rain/GOPR0573_frame_000246_colors.png) |
| ![Image 7](./docs/imgs/orig_imgs/night/GOPR0355_frame_000162_rgb_anon.png) | ![Ensemble 7](./docs/imgs/pred_ens/night/GOPR0355_frame_000162_colors.png) |
| ![Image 10](./docs/imgs/orig_imgs/fog/GP010475_frame_000300_rgb_anon.png)  | ![Ensemble 10](./docs/imgs/pred_ens/fog/GP010475_frame_000300_colors.png) |


## To Do
- [ ] Add technical report
 - [x] Add predictions generated with the best model and ensemble
 - [x] Add results obtained with different architectures 
 - [x] Add tables with final results and (pseudo)labeled data used
 - [x] Add environment installation instructions
 - [x] Upload saved checkpoints
 - [x] Add `config.json` from vistas and create `datasets` folder (remove from `.gitgnore`)
 - [ ] Add singlescale/multiscale inference support 
 - [ ] Add ensemble evaluation support
 ## Method
Our method involves collecting a large number of labeled and unlabeled scene driving images. We used labeled images itself as a source of supervision, and for unlabeled images, we generated hard pseudolabels based on the model's softmax confidence. First, we trained the model using only labeled images. Then, based on thresholded softmax confidence, we generated pseudolabels for the unlabeled images. We iteratively repeated this procedure.

**Labeled datasets** we use: *ACDC*, *Cityscapes*, *Dark Zurich* validation subset, *Foggy Zurich*, *Foggy Driving*, *Wilddash*, *BDD10k* Train/Val, *Mapillary Vistas*, *Foggy Cityscapes*, *Rain Cityscapes*

**Unlabeled datasets** for which we generate pseudolabels: *Dark Zurich* train, 10k *NightOwls* images, *NightCity*, *BDD100k* {Rain, Snow, Fog} subsets, 5.6k *CADCD* images
 ### Experiments 
First, we confirmed the architecture we intend to use. The results of this validation process are detailed in the [architecture validation](./docs/ARCHITECTURE_VALIDATION.md) document. Following architecture validation, we conducted 8 training runs, with different training datasets employed in each run. The final model sent to the evaluation server was a simple ensemble comprised of models from 4 distinct experiments (Experiments 2, 5, 6, and 7 as listed in the [final experiments table](./docs/FINAL_EXPERIMENTS.md#results)). For a more comprehensive overview of the hyperparameters and datasets employed in each of the experiments, please refer to the associated [document](./docs/FINAL_EXPERIMENTS.md).

At the end of the competition, our model was ranked 2nd with **83.82** mIoU. 

![Benchmark-Results](benchmark_acdc.png)

 #### Checkpoints
 Pretrained checkpoints are available [here](https://ferhr-my.sharepoint.com/:f:/g/personal/imartinovic_fer_hr/EgHe_gpUZppOtITLcVHQEuwBnlNK93h_BtpRjpdDMYcHTA?e=G1ohSa). After downloading, you should have the following directory structure:

```
acdc-semseg/               # Project root directory
├── ckpts/                 # Checkpoints directory
│   ├── model-single/
│     ├── sn-pyr_cn-lg_ud-320_best.ckpt
│   └── models-ensemble/
│     ├── sn-pyr_cn-lg_ud-320_ens1.ckpt
│     ├── sn-pyr_cn-lg_ud-320_ens2.ckpt
│     ├── sn-pyr_cn-lg_ud-320_ens3.ckpt
│     └── sn-pyr_cn-lg_ud-320_ens4.ckpt    
├── ...                    # Other project files and directories
```
 ### Predictions visualization

#### Snow
| Original Image | Single Model Prediction (ms) | Ensemble Prediction (ms) |
|:--------------:|:----------------------:|:------------------:|
| ![Image 1](./docs/imgs/orig_imgs/snow/GP010176_frame_000643_rgb_anon.png) | ![Prediction 1](./docs/imgs/pred_single/snow/GP010176_frame_000643_colors.png) | ![Ensemble 1](./docs/imgs/pred_ens/snow/GP010176_frame_000643_colors.png) |
| ![Image 2](./docs/imgs/orig_imgs/snow/GP010494_frame_000001_rgb_anon.png) | ![Prediction 2](./docs/imgs/pred_single/snow/GP010494_frame_000001_colors.png) | ![Ensemble 2](./docs/imgs/pred_ens/snow/GP010494_frame_000001_colors.png) |
| ![Image 3](./docs/imgs/orig_imgs/snow/GP010176_frame_000418_rgb_anon.png) | ![Prediction 3](./docs/imgs/pred_single/snow/GP010176_frame_000418_colors.png) | ![Ensemble 3](./docs/imgs/pred_ens/snow/GP010176_frame_000418_colors.png) |
#### Rain
| Original Image | Single Model Prediction (ms) | Ensemble Prediction (ms) |
|:--------------:|:----------------------:|:------------------:|
| ![Image 4](./docs/imgs/orig_imgs/rain/GOPR0573_frame_000246_rgb_anon.png) | ![Prediction 4](./docs/imgs/pred_single/rain/GOPR0573_frame_000246_colors.png) | ![Ensemble 4](./docs/imgs/pred_ens/rain/GOPR0573_frame_000246_colors.png) |
| ![Image 5](./docs/imgs/orig_imgs/rain/GP020573_frame_000353_rgb_anon.png) | ![Prediction 5](./docs/imgs/pred_single/rain/GP020573_frame_000353_colors.png) | ![Ensemble 5](./docs/imgs/pred_ens/rain/GP020573_frame_000353_colors.png) |
| ![Image 6](./docs/imgs/orig_imgs/rain/GP010571_frame_000784_rgb_anon.png) | ![Prediction 6](./docs/imgs/pred_single/rain/GP010571_frame_000784_colors.png) | ![Ensemble 6](./docs/imgs/pred_ens/rain/GP010571_frame_000784_colors.png) |
#### Night

| Original Image | Single Model Prediction (ms) | Ensemble Prediction (ms) |
|:--------------:|:----------------------:|:------------------:|
| ![Image 7](./docs/imgs/orig_imgs/night/GOPR0355_frame_000162_rgb_anon.png) | ![Prediction 7](./docs/imgs/pred_single/night/GOPR0355_frame_000162_colors.png) | ![Ensemble 7](./docs/imgs/pred_ens/night/GOPR0355_frame_000162_colors.png) |
| ![Image 8](./docs/imgs/orig_imgs/night/GOPR0356_frame_000162_rgb_anon.png) | ![Prediction 8](./docs/imgs/pred_single/night/GOPR0356_frame_000162_colors.png) | ![Ensemble 8](./docs/imgs/pred_ens/night/GOPR0356_frame_000162_colors.png) |
| ![Image 9](./docs/imgs/orig_imgs/night/GP010594_frame_000034_rgb_anon.png) | ![Prediction 9](./docs/imgs/pred_single/night/GP010594_frame_000034_colors.png) | ![Ensemble 9](./docs/imgs/pred_ens/night/GP010594_frame_000034_colors.png) |
#### Fog
| Original Image | Single Model Prediction (ms) | Ensemble Prediction (ms) |
|:--------------:|:----------------------:|:------------------:|
| ![Image 10](./docs/imgs/orig_imgs/fog/GP010475_frame_000300_rgb_anon.png) | ![Prediction 10](./docs/imgs/pred_single/fog/GP010475_frame_000300_colors.png) | ![Ensemble 10](./docs/imgs/pred_ens/fog/GP010475_frame_000300_colors.png) |
| ![Image 11](./docs/imgs/orig_imgs/fog/GP010477_frame_000127_rgb_anon.png) | ![Prediction 11](./docs/imgs/pred_single/fog/GP010477_frame_000127_colors.png) | ![Ensemble 11](./docs/imgs/pred_ens/fog/GP010477_frame_000127_colors.png) |
| ![Image 12](./docs/imgs/orig_imgs/fog/GP010478_frame_000167_rgb_anon.png) | ![Prediction 12](./docs/imgs/pred_single/fog/GP010478_frame_000167_colors.png) | ![Ensemble 12](./docs/imgs/pred_ens/fog/GP010478_frame_000167_colors.png) |

 ## Running

### Environment Installation

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

We provide three scripts:

- [SNCN_train_city.py](SNCN_train_city.py): This script is used for training SwiftNet+ConvNext on the `Cityscapes` training subset and evaluating it on the `Cityscapes` validation split.
- [generate_pseudolabels.py](generate_pseudolabels.py): This script generates pseudolabels for unlabeled data, saving the generated pseudolabels in the `args.output_pseudo` folder, along with colorized versions of the pseudolabels in the same directory.
- [generate_predictions.py](generate_predictions.py): This script generates predictions for custom images, saving the resulting predictions in the `args.output_pred` folder, along with colorized versions of the segmentation maps.

 ### Datasets

Cityscapes and DarkZurich Val subset are sufficient to run [training](SNCN_train_city.py) and [pseudolabel generation](generate_pseudolabels.py) scripts.

Cityscapes can be downloaded [here](https://www.cityscapes-dataset.com/). Please ensure the following directory structure:
```
cityscapes/
├── gtFine/
│   ├── train/
│   ├── test/
│   └── val/
└── leftImg8bit/
    ├── train/
    ├── test/
    └── val/
```
To create labelTrainIds from labelIds you can use [this repository](https://github.com/mcordts/cityscapesScripts).

ACDC dataset can be downloaded [here](https://acdc.vision.ee.ethz.ch/). Please ensure the following directory structure:
```
acdc/
├── gt/
│   ├── fog/
│   ├── night/
|   ├── rain/
│   └── snow/
└── rgb_anon/
    ├── fog/
    ├── night/
    ├── rain/
    └── snow/
```
DarkZurich can be downloaded [here](https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/). For pseudolabel generation demo, validation subset of Dark Zurich with 50 images is sufficient. Please ensure the following directory structure (`val` is sufficient for running [generation script](generate_pseudolabels.py)):

```
Dark_Zurich/
├── gt/
│   └── val/
│       └── night/
│           └── GOPR0356/
└── rgb_anon/
    ├── val/
    │   └── night/
    │       └── GOPR0356/
    └── train/
        ├── day/
        ├── twilight/
        └── night/
```


#### Training
To run training for ConvNeXt-tiny+SwiftNet-pyramid on the `Cityscapes` training subset and evaluate it on the `Cityscapes` validation subset (using GPUs with indices 0 and 1, with a batch size of 4 per GPU), execute [SNCN_train_city](./SNCN_train_city.py) script:

```bash
python SNCN_train_city.py -sv pyr -bv tiny --gpus 0 1 --batch_size_per_gpu 4
```

In `SNCN_train_city.py`, you can adjust other arguments as described in the script itself.

 #### Generating Pseudolabels
 To generate pseudolabels with trained ConvNeXt+SwiftNet model for some specific dataset (currently `DarkZurich night val subset`), download [checkpoints](#checkpoints), and then execute [generate pseudolabels](./generate_pseudolabels.py) script:
 ```bash
 python generate_pseudolabels.py -sv pyr -bv large --gpus 0 --upsample_dims 320 --ckpt_path ckpts/model_single/model_last-epoch=98-val_mIoU=86.50.ckpt
 ```
 #### Generating Predictions
 To generate predictions with trained ConvNeXt+SwiftNet model for custom images, download [checkpoints](#checkpoints), and then execute [generate_predictions](./generate_predictions.py) script:
 ```bash
 python generate_predictions.py -sv pyr -bv large --gpus 0 --upsample_dims 320 --ckpt_path ckpts/model_single/model_last-epoch=98-val_mIoU=86.50.ckpt --img_dir path/to/own/directory/with/images
 ```