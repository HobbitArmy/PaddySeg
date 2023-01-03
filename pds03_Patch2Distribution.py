"""
-*- coding: utf-8 -*-
@Author  : luxy
Application of Direct Geo-Locating and Patch Sparse Sampling

To calculate trait-distribution with sparse-sampled-patch and model-inferred-mask:
1. get sparse-patch, and infer the trait-mask;
2. for image_i in images:
    2.1 grid each mask to smaller boxes and average the trait-value;
    2.2 geo-locate these boxes;
    2.3 collect value-location pair;
3. draw the trait-distribution

!! Need Download for testing:
1. images data and put them at img/ (extra files at: )
2. model weights and put them at GBiNet/weights (extra files at: )

@Time    : 2022/12/3 11:00
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
from GBiNet.mmseg.apis import inference_segmentor, init_segmentor
from pds02_PatchSparseSampling import geo_coord_list_calc
from skimage.measure import block_reduce
import pandas as pd


# 1. get sparse-patch, and infer the trait-mask
def Patch2Mask(model, img_dir, out_dir=None, tar_mask_size=(1638, 1092)):
    """
    :param model: the mmsegmentation model
    :param img_dir:directory of input image
    :param out_dir:output directory
    :param tar_mask_size: target mask size
    :return:
    """
    if not out_dir: out_dir = img_dir + '/mask_pred/'
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    img_name_list = [img for img in os.listdir(img_dir) if img.endswith('.JPG')]
    for img_name in tqdm(img_name_list):
        pred_mask = inference_segmentor(model, img_dir + '/' + img_name)[0]
        mask_img = Image.fromarray(pred_mask.astype(np.uint8)).resize(tar_mask_size, Image.Resampling.LANCZOS)
        # mask_img_array = pred_mask[0].astype(np.uint8)
        mask_img.save(out_dir + '/' + img_name.split('.')[0] + '.png')
    return True


# 2. Patch-Mask to geo-distribution [91x91] 1638x1092 -> 18x12
def mask2GeoDistribution(img_dir, patch_dir, box_size=(91, 91)):
    """
    calculate traits geo-distribution from trait-mask
    :param img_dir:directory of input image
    :param patch_dir:directory of patchSparseSampling-out_dir in pds02_PatchSparseSampling.py
    :param box_size: filter size of average pooling
    :return:
    """
    mask_dir = patch_dir + '/mask_pred/'
    mask_name_list = [mask for mask in os.listdir(mask_dir) if mask.endswith('.png')]
    trait_dist_collect = []
    # Patch edge pixel-location (generate by pds02_PatchSparseSampling.py )
    PEPL_array = np.load(patch_dir + '/inter_parms/PEPL_array1.npy')
    for i in tqdm(range(len(mask_name_list))):
        mask_head, img_suffix = mask_name_list[i].split('.')
        image_head, mask_idx = mask_head.split('-')
        # 2.1 grid each mask to smaller boxes and average the trait-value;
        mask = Image.open(mask_dir + mask_name_list[i])
        mask_ds = block_reduce(np.asarray(mask), box_size, np.mean)  # Avg Pooling

        # 2.2 geo-locate these boxes;
        w, h = mask.size
        row, col = mask_ds.shape
        ls_w, ls_h = np.int32(np.linspace(0, w, col + 1)), np.int32(np.linspace(0, h, row + 1))
        # pixel-coord of patches center
        box_centers = np.array([[int((ls_w[j] + ls_w[j + 1]) / 2), int((ls_h[i] + ls_h[i + 1]) / 2)]
                                for i in range(row) for j in range(col)])
        box_pixel_coord = box_centers + PEPL_array[i][:2]
        box_geo_coord = geo_coord_list_calc(img_dir + '/' + image_head + '.JPG', box_pixel_coord)[1]
        mask_ds = mask_ds.reshape((mask_ds.size, 1))  # average_pooling
        trait_distribution = np.concatenate((box_geo_coord, mask_ds), axis=1)
        trait_dist_collect.append(trait_distribution)
    trait_sum = np.concatenate(trait_dist_collect, axis=0)
    patches_name = [mask_path.split('.')[0] for mask_path in os.listdir(patch_dir) if mask_path.endswith('.JPG')]
    trait_sum_img = pd.DataFrame(
        [[trait_sum[i][0], trait_sum[i][1], patches_name[i]] for i in range(len(patches_name))])
    trait_sum_img.to_csv(patch_dir1 + '/traits_sum.csv')
    print('traits_sum.csv is saved under img/patches/')
    plt.scatter(trait_sum[:, 1], trait_sum[:, 0], s=0.1, marker='.', )
    return trait_sum


def traits_plot(patch_dir, trait_sum):
    plt.figure(figsize=(12, 8))
    ax = plt.axes()
    s2 = plt.scatter(trait_sum[:, 1], trait_sum[:, 0], s=2, marker='.', color=(0.6, 0.6, 1, 0.5))

    IGL_array = np.load(patch_dir + '/inter_parms/IGL_array1.npy')
    PGL_array = np.load(patch_dir + '/inter_parms/PGL_array1.npy')
    elm_d = np.load(patch_dir + '/inter_parms/elm_d1.npy')

    s0 = plt.scatter(IGL_array[:, 1], IGL_array[:, 0], c='b', marker='*')
    s1 = plt.scatter(PGL_array[:, 1], PGL_array[:, 0], c='r', marker='.', s=10)

    x0, y0 = IGL_array[:-1, 1], IGL_array[:-1, 0]
    d_x = [IGL_array[i + 1][1] - IGL_array[i][1] for i in range(len(IGL_array) - 1)]
    d_y = [IGL_array[i + 1][0] - IGL_array[i][0] for i in range(len(IGL_array) - 1)]
    # ax.quiver(x0, y0, d_x, d_y, color=(0, 0, 1, 0.6), width=0.003, angles='xy',
    #           scale_units='xy', scale=1.1, headwidth=2.5, headlength=4.8)
    ax.quiver(x0, y0, d_x, d_y, color=(1, 0, 0, 0.5), width=0.002, angles='xy',
              scale_units='xy', scale=1.1, headwidth=2.5, headlength=4.5)

    plt.title('Sparse Sampling with elm_d:%.1f' % elm_d)
    plt.legend((s0, s1, s2), ('IGL', 'PGL', 'BGL'), loc='best')
    # plt.axis('off')


if __name__ == '__main__':
    # 1. Patch to Mask_pred
    # 1.1 init segmentor
    config_file = r'GBiNet/configs/GBiNet_t/GBiNet_t32dx2_r4_2x4_60k_pdseg_dsw.py'
    # Need Download the Weights files at:
    checkpoints_file = r'GBiNet/weights/GBiNet_t32dx2_r4_2x4_60k_pdseg_dsw_iter_54000.pth'
    model1 = init_segmentor(config_file, checkpoint=checkpoints_file, device='cuda')
    # 1.2 infer the prediction mask
    patch_dir1 = r'img/patches/'
    Patch2Mask(model1, patch_dir1)

    # 2. Mask_pred to Traits-distribution (output img/patches/traits_sum.csv)
    img_dir1 = r'img/'  # dir to original images for geo-location
    trait_sum1 = mask2GeoDistribution(img_dir1, patch_dir1, box_size=(91, 91))

    # 3. Plot
    traits_plot(patch_dir1, trait_sum1)
