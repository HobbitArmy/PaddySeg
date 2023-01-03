# Automated Rice Phenology Stage Mapping using UAV Images and Deep Learning

This is the code implementations of  methods and models proposed in paper *Automated Rice Phenology Stage Mapping using UAV Images and Deep Learning* . 

### Abstract

In the paper, we have designed a straightforward and effective workflow, where patches are split from the image incrementally and sampled sparsely (ISS) to eliminate computational redundancy. These patches were fed into the segmentation model that generates traits mask. Based on UAV photogrammetry, each pixel could be direct geo-located (DGL).  And the final rice phenology mapping could be achieved in QGIS with interpolation of the coupled trait values and locations. This phenology mapping system could aid decision-making towards automatic management of the rice field.

### Flow

pds01_DirectGL.py is for Direct Geo-Locating (DGL) of a UAV image at the vertical overhead view.

pds02_PatchSparseSampling.py is for Incremental Sparse Sampling (ISS) where the original image is split into patches and selected sparsely.

pds03_Patch2Distribution.py for: 1. predict trait-masks of the patches; 2. down-sample and generate the trait-location pairs csv file.

in which GBiNet model is used for prediction, and detail of the model is arranged in MMSegmentation style under PaddySeg/GBiNet/

### Extra Files

The trained model and 4 sample images are also available at: 

[PaddySeg_ExtraFiles - Google Drive](https://drive.google.com/drive/folders/1NnFOPRP20jvi3EHetyB1fUaqo4c4aJqQ?usp=sharing)

[PaddySeg_ExtraFiles - Baidu Netdisk](https://pan.baidu.com/s/1VV_8Tn3tJtONhDew1_6Jrw?pwd=mw1x)
