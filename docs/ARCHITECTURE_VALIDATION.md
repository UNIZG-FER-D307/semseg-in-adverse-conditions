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