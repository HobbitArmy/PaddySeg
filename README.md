# Automated Rice Phenology Stage Mapping using UAV Images and Deep Learning

This is the code implementations of  methods and models proposed in paper *Automated Rice Phenology Stage Mapping using UAV Images and Deep Learning* . 

### Abstract

Accurate monitoring of rice phenology is critical for crop management, cultivars breeding and yield estimating. In the paper, we have designed a straightforward and effective workflow, where patches are split from the image incrementally and sampled sparsely (ISS) to eliminate computational redundancy. These patches were fed into the segmentation model that generates traits mask. Based on UAV photogrammetry, each pixel could be direct geo-located (DGL).  And the final rice phenology mapping could be achieved in QGIS with interpolation of the coupled trait values and locations. This phenology mapping system could aid decision-making towards automatic management of the rice field.

**Figure 2. The construction process of PaddySeg Dataset:**
![Figure 2. The construction process of PaddySeg Dataset](img/Figure%202.jpg)
### Flow

pds01_DirectGL.py is for Direct Geo-Locating (DGL) of a UAV image at the vertical overhead view.

**Figure 6. Direct Geo-Locating Based on (a) Exterior Orientation and (b) Central Projection at Nadir View:**
![Figure 6. Direct Geo-Locating Based on (a) Exterior Orientation and (b) Central Projection at Nadir View](img/Figure%206.jpg)

pds02_PatchSparseSampling.py is for Incremental Sparse Sampling (ISS) where the original image is split into patches and selected sparsely.

**Figure 8. Incremental Sparse Sampling and System Workflow:**
![Figure 8. Incremental Sparse Sampling and System Workflow](img/Figure%208.jpg)

pds03_Patch2Distribution.py for: 1. predict trait-masks of the patches; 2. down-sample and generate the trait-location pairs csv file.

**Figure 12. Rice Phenology Mapping at DJD-5 Experimental Field: a. Drone Waypoints of Image Capture; b. Sparse Sampled Patch-Boxes Distribution; c. Distribution Map of Rice Phenology with Interpolation:**
![Figure 12. Rice Phenology Mapping at DJD-5 Experimental Field: a. Drone Waypoints of Image Capture; b. Sparse Sampled Patch-Boxes Distribution; c. Distribution Map of Rice Phenology with Interpolation](img/Figure%2012.jpg)

GBiNet model is used for prediction, and detail of the model is arranged in MMSegmentation style under PaddySeg/GBiNet/

**Figure 5. Overview of Ghost Bilateral Network (GBiNet). The network has 4 main parts: the Detail Branch, Semantic Branch, Aggregation Layer, and GCN Segmentation Head. Each box is an operation block, and the arrow connection represents the feature maps flow with numbers above showing the ratios of map size to the input size:**
![Figure 5. Overview of Ghost Bilateral Network (GBiNet). The network has 4 main parts: the Detail Branch, Semantic Branch, Aggregation Layer, and GCN Segmentation Head. Each box is an operation block, and the arrow connection represents the feature maps flow with numbers above showing the ratios of map size to the input size. ](img/Figure%205.jpg)
### Extra Files

The trained model and 4 sample images are also available at: 

[PaddySeg_ExtraFiles - Google Drive](https://drive.google.com/drive/folders/1NnFOPRP20jvi3EHetyB1fUaqo4c4aJqQ?usp=sharing)

[PaddySeg_ExtraFiles - Baidu Netdisk](https://pan.baidu.com/s/1VV_8Tn3tJtONhDew1_6Jrw?pwd=mw1x)
