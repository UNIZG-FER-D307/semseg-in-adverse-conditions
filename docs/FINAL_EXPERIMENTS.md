 #### Final experiments
 
To obtain the final model, we conducted 8 distinct experiments, utilizing the *Convnext large* backbone and *SwiftNet-320-pyr* in each experiment. The experiments varied based on the datasets used for training. The first column indicates the name of the dataset for which we generated pseudolabels, while the last column represents the number of images in each of these datasets. The intermediate columns represent the experiment index, and a checkmark (✔) denotes the usage of a dataset in that particular experiment, while a smaller black cross (✗) indicates its exclusion.

##### Labeled datasets used throughout the experiments

| Labeled Dataset               | 1. | 2. | 3. | 4. | 5. | 6. | 7. | 8. | Number of Images |
|-----------------------        |---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|------------------:|
| *ACDC*                          | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | 1600             |
| *Cityscapes* train              | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | 2975             |
| *Dark Zurich* val               | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | 50               |
| *Foggy Zurich*                  | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | 40               |
| *Wilddash2*                     | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | 4256             |
| *BDD10k* train                  | ✔             | ✔             | ✗             | ✔             | ✔             | ✔             | ✗             | ✔             | 7000             |
| *BDD10k* val                    | ✔             | ✔             | ✗             | ✔             | ✔             | ✔             | ✗             | ✔             | 1000             |
| *Mapillary Vistas* train        | ✗             | ✔             | ✗             | ✔             | ✔             | ✔             | ✗             | ✗             | 18000            |
| *Mapillary Vistas* val          | ✔             | ✔             | ✗             | ✔             | ✔             | ✔             | ✗             | ✗             | 2000             |
| *Mapillary Vistas* snow train    | ✔             | ✗             | ✔             | ✔             | ✔             | ✗             | ✔             | ✗             | 652              |
| *Mapillary Vistas* snow val      | ✔             | ✗             | ✔             | ✔             | ✔             | ✗             | ✔             | ✗             | 71               |
| *Foggy Driving*                 | ✔             | ✔             | ✗             | ✔             | ✔             | ✔             | ✗             | ✗             | 33               |
| *Foggy Cityscapes* train        | ✗             | ✗             | ✗             | ✔             | ✔             | ✔             | ✗             | ✗             | 8878             |
| *Foggy Cityscapes* val          | ✗             | ✗             | ✗             | ✔             | ✔             | ✔             | ✗             | ✗             | 1500             |
| *Rain Cityscapes*               | ✗             | ✗             | ✗             | ✔             | ✔             | ✗             | ✗             | ✗             | 9425             |
| *Cityscapes* val                | ✗             | ✗             | ✗             | ✗             | ✗             | ✔             | ✗             | ✗             | 500              |

##### Pseudolabeled datasets used throughout the experiments

| Pseudolabeled Dataset |  1. |  2. |  3. | 4. | 5. | 6. | 7. | 8. | Number of Images |
|-----------------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|------------------:|
| *Dark Zurich Night*    | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | 2416             |
| *Dark Zurich Twilight* | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | 2920             |
| *NightOwls train*      | ✔             | ✗             | ✗             | ✔             | ✔             | ✗             | ✗             | ✗             | 10000            |
| *NightOwls val*        | ✔             | ✔             | ✗             | ✔             | ✔             | ✗             | ✗             | ✗             | 2593             |
| *NightCity train*      | ✔             | ✗             | ✗             | ✔             | ✔             | ✗             | ✗             | ✗             | 2997             |
| *NightCity val*        | ✗             | ✗             | ✗             | ✔             | ✔             | ✗             | ✗             | ✗             | 1300             |
| *BDD100k-Rain train*   | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | ✗             | 5070             |
| *BDD100k-Rain val*     | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | ✗             | 738              |
| *Seeing Through Fog*   | ✔             | ✔             | ✔             | ✔             | ✔             | ✗             | ✔             | ✗             | 12997            |
| *BDD100k-Fog train*    | ✔             | ✗             | ✗             | ✔             | ✔             | ✔             | ✗             | ✗             | 130              |
| *BDD100k-Snow train*   | ✔             | ✗             | ✗             | ✔             | ✔             | ✔             | ✗             | ✗             | 5549             |
| *BDD100k-Snow val*     | ✔             | ✗             | ✗             | ✔             | ✔             | ✔             | ✗             | ✗             | 769              |
| *CADCD*                 | ✔             | ✔             | ✔             | ✔             | ✔             | ✗             | ✔             | ✗             | 5600             |
| *Dark Zurich Day*      | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | ✔             | 3041             |

##### Results

| | 1. | 2. | 3. | 4. | 5. | 6. | 7. | 8. |
|-----------------------------|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
||
| Labeled Training Images      | 19677        | 36954        | 9644         | 57480        | 57480        | 47832        | 9644         | 16921        |
| Pseudolabeled Training Images| 54820        | 35375        | 32782        | 56120        | 56120        | 20633        | 32782        | 8377         |
| Batch Size (Per GPU)         | 5            | 5            | 5            | 5            | 5            | 5            | 5            | 5            |
| GPUs Used                    | 4            | 4            | 8            | 4            | 8            | 4            | 8            | 8            |
| Uniform Sampling             | ✔            | ✔            | ✗            | ✔            | ✔            | ✔            | ✔            | ✗            |
| Color Jittering           | ✔            | ✔            | ✔            | ✗            | ✔            | ✔            | ✔            | ✔            |
| Initial Learning Rate        | $4\mathrm{e}{-4}$ | $4\mathrm{e}{-4}$ | $4\mathrm{e}{-4}$ | $4\mathrm{e}{-4}$ | $5\mathrm{e}{-4}$ | $4\mathrm{e}{-4}$ | $4\mathrm{e}{-4}$ | $4\mathrm{e}{-4}$ |
| Number of Epochs             | 80           | 80           | 80           | 60           | 100          | 80           | 100          | 80           |
||
| **mIoU (\%)**                | 85.48        | 86.27        | 85.83        | 86.00        | **86.50**    | 85.65        | 86.05        | 82.80        |

Experiments were conducted using NVIDIA A100 GPUs with 40GB of available memory. For experiments on 4 GPUs, a batch size of 16 was used. For experiments on 8 GPUs, a batch size of 40 was utilized, resulting in 5 mini-batches per GPU. The learning rate for the *Convnext large* backbone was set to $\frac{1}{4}$ of the initial learning rate used for the upsampling path. We employed the *Adam* optimizer, cosine annealing schedule, and applied color jittering, scale jittering, and horizontal flipping with uniform class sampling. Image crops were of dimensions $1024\times 1024$ pixels.