# GBiNet: Ghost Bilateral Network

This repository contains the official Pytorch implementation of training & evaluation code and the pretrained models for Ghost Bilateral Network (GBiNet) .

GBiNet is a efficient semantic segmentation method based on BiSeNetV2, as shown in Figure 1.


We use [MMSegmentation v0.29.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.29.0) as the codebase.

GBiNet is implemented on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/segformer). 

PaddySeg dataset is also formalized in MMSegmentation-VOC-dataset style.

## Installation

For install and data preparation, please refer to the guidelines in [MMSegmentation v0.29.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.29.0).

## Evaluation

Download [PaddySeg_ExtraFiles - Google Drive](https://drive.google.com/drive/folders/1NnFOPRP20jvi3EHetyB1fUaqo4c4aJqQ?usp=sharing)

Example: evaluate ```GBiNet_t32dx2_r4``` on ```pdseg```:

```shell
# Inference cost testing 
python tools/get_flops.py configs/GBiNet/GBiNet_r2_2x4_60k_pdseg_dsw.py --shape 819 546

# Inference speed testing
python tools/benchmark.py configs/GBiNet_t/GBiNet_t32dx2_r4_2x4_60k_pdseg.py ..\weights\GBiNet_t32dx2_r4_2x4_60k_pdseg_dsw_iter_54000.pth --repeat-times 5

# Segmentation performance testing
python tools/test.py configs/GBiNet_t/GBiNet_t32dx2_r4_2x4_60k_pdseg_dsw.py ..\weights\GBiNet_t32dx2_r4_2x4_60k_pdseg_dsw_iter_54000.pth --eval mIoU
```

## Training

Example: train ```GBiNet_r2``` on ```pdseg```:

```shell
python tools/train.py configs/GBiNet/GBiNet_r2_2x4_60k_pdseg_dsw.py --seed 2022
```

## License

Please check the LICENSE file. GBiNet may be used non-commercially, meaning for research or evaluation purposes only. 
