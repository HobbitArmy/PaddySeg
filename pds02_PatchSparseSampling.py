"""
-*- coding: utf-8 -*-
@Author  : luxy

To save inference-compute-costs with incremental-sparse patch sampling:
(Considering the DGL-acc, the outer edge patches of each image are removed)
1. initialize patch_list (PGL_list: [IMAGE_NAME-PATCH_IDX] [PATCH_GEO_LOCATION]) with the first image;
2. for image_i in images:
         2.1 get geo-locations (GL_i) of every patches in image_i;
         2.2 calculate the pair-distance-matrix between GL_i and GL_l;
         2.3 eliminate all GL_i that too near (< 0.8 patch-distance-unit);
         2.4 append others remained in GL_i to PGL_list;
3. return GL_l, and plot patch distribution map


!! Need Download for testing:
1. images data and put them at img/ (extra files at: )

@Time    : 2022/12/2 16:26
"""

import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from haversine import inverse_haversine, Direction, Unit, haversine, haversine_vector
from pyexiv2 import Image as Image2
from tqdm import tqdm


def geo_coord_list_calc(img_path, pix_coord_list, ):
    """
    Calculate geo-location of a list of point in one image (given-pixel-coordinate)
    :param img_path:        path of the image
    :param pix_coord_list:  image-pixel-coordinate of point
    :return: geo_coord_list:    WGS84 coordinate of point in list
    """
    img1 = Image2(img_path, encoding='gbk')
    img1_xmp1, img1_exif = img1.read_xmp(), img1.read_exif()
    lat_, lon_ = img1_exif['Exif.GPSInfo.GPSLatitude'].split(), img1_exif['Exif.GPSInfo.GPSLongitude'].split()
    lat, lon = eval(lat_[0]) + eval(lat_[1]) / 60 + eval(lat_[2]) / 3600, eval(lon_[0]) + eval(lon_[1]) / 60 + eval(
        lon_[2]) / 3600
    kappa = -float(img1_xmp1['Xmp.drone-dji.FlightYawDegree'])
    cos_theta, sin_theta = np.cos(np.radians(kappa)), np.sin(np.radians(kappa))
    cs_trans_M = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    relative_height = eval(img1_xmp1['Xmp.drone-dji.RelativeAltitude'])
    focal_len = eval(img1_exif['Exif.Photo.FocalLength'])
    h = eval(img1_exif['Exif.Photo.PixelYDimension'])
    w = eval(img1_exif['Exif.Photo.PixelXDimension'])
    sensor_size = np.array([35.000, 23.3276])
    # %% point geo-coord calculation
    geo_coord_list, geo_kappa_list = [], []
    for tar_pix_coord in pix_coord_list:
        tar_mm_vec = np.array([w / 2, h / 2]) - np.array([tar_pix_coord])
        # pixel-vector from image-center to selected-point
        tar_mm_vec[:, 0] = -tar_mm_vec[:, 0]
        # image-coordinate/geo-coordinate are left/right -handed system (Y axis is flipped around the X axis)
        tar_geo_coord_vector = np.dot(cs_trans_M, tar_mm_vec.T)  # project to the North and East
        tar_geo_distance = (sensor_size / [w, h]) * (relative_height / focal_len) * tar_geo_coord_vector
        # convert to geo-lcoation using collinear relationship
        tar_coord = [
            inverse_haversine([lat, lon], tar_geo_distance.T[0, 1], Direction.NORTH, unit=Unit.METERS)[0],
            inverse_haversine([lat, lon], tar_geo_distance.T[0, 0], Direction.EAST, unit=Unit.METERS)[1]]
        geo_coord_list.append(tar_coord)
        geo_kappa_list.append(kappa)
    return (lat, lon), np.array(geo_coord_list), np.array(geo_kappa_list)


# Sparse sample the patch
def patchSparseSampling(img_dir, patch_num_rc: tuple = (5, 5), elm_d: float = 0.8,
                        rmv_edge: int = 1, show_plot=True, out_dir='./img/patches/'):
    """
    Sparse sampling patches of images in img_dir,
    :param img_dir:         directory of images
    :param patch_num_rc:    number of rows and cols in one image
    :param elm_d:           elm_d-times of patch-distance-unit is the nearest distance to be kept
    :param rmv_edge:        number of edge-patches to remove
    :param show_plot:       whether to show distribution map of remained-patches
    :param out_dir:        directory to save
    :return: (PATCH_list, PGL_array, PEPL_array): patch index; patch geo-location; patch edge pixel-location
    """
    # 1. initialize patch_list
    imgs_name = [img.split('.')[0] for img in os.listdir(img_dir) if img.endswith('.JPG')]
    if len(imgs_name) == 0:
        print('No image exists.')
        return False
    IGL_list, PGL_list, PGK_list = [], [], [],  # image geo-location; patch geo-location; patch geo-kappa
    PEPL_list = []  # patch edge pixel-location
    PATCH_list = []  # patch index;
    img = Image.open(img_dir + imgs_name[0] + '.JPG')
    w, h = img.size
    # boundary of patches in width and height:
    ls_w, ls_h = np.linspace(0, w, patch_num_rc[1] + 1), np.linspace(0, h, patch_num_rc[0] + 1)
    rmd_r, rmd_c = patch_num_rc[0] - rmv_edge * 2, patch_num_rc[1] - rmv_edge * 2  # patches number
    # pixel-coord of patches center
    patch_centers = np.array([[int((ls_w[j + rmv_edge] + ls_w[j + rmv_edge + 1]) / 2),
                               int((ls_h[i + rmv_edge] + ls_h[i + rmv_edge + 1]) / 2)]
                              for i in range(0, rmd_r) for j in range(0, rmd_c)])
    # geo-coord of patches
    geo_image_center, geo_patch_center, geo_patch_kappa = geo_coord_list_calc(img_dir + imgs_name[0] + '.JPG',
                                                                              patch_centers)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for file in glob.glob(out_dir+"/*.JPG"):
        os.remove(file)
    # initial patches
    ls_w, ls_h = np.int32(ls_w), np.int32(ls_h)
    IGL_list.append(geo_image_center)
    for i in range(0, rmd_r):  # row
        for j in range(0, rmd_c):  # col
            PATCH_list.append(imgs_name[0] + '-%i%i' % (i + rmv_edge, j + rmv_edge))
            PGL_list.append(geo_patch_center[i * rmd_c + j])
            PGK_list.append(geo_patch_kappa[i * rmd_c + j])
            PEPL_list.append([ls_w[j + rmv_edge], ls_h[i + rmv_edge], ls_w[j + rmv_edge + 1], ls_h[i + rmv_edge + 1]])
            # left,top,right,bottom
            img_ij = img.crop(PEPL_list[-1])  # left,top,right,bottom
            img_ij.save(out_dir + '/%s%s' % (PATCH_list[-1], '.JPG'))
    # distance between adjacent patches (short one: in image Up-Bottom direction)
    degree_distance = np.linalg.norm(PGL_list[0] - PGL_list[patch_num_rc[1] - rmv_edge * 2])
    unit_patch_distance = haversine(PGL_list[0], PGL_list[patch_num_rc[1] - rmv_edge * 2], unit=Unit.METERS)

    # 2. for image_i in images
    for image_i_name in tqdm(imgs_name[1:]):
        # geo-coord of patches
        img = Image.open(img_dir + image_i_name + '.JPG')
        geo_image_center, geo_patch_center, geo_patch_kappa = geo_coord_list_calc(img_dir + image_i_name + '.JPG',
                                                                                  patch_centers)
        IGL_list.append(geo_image_center)
        # calculate geo_patch_centers(base) and existing patches distance
        geo_distance_matrix = haversine_vector(np.array(PGL_list), geo_patch_center, comb=True, unit=Unit.METERS)
        elm_idx = np.where(geo_distance_matrix < unit_patch_distance * elm_d)[0]
        for idx in set(range(len(geo_distance_matrix))) - set(elm_idx):  # 差集
            i, j = idx // rmd_c, idx % rmd_c  # i - row, j - col
            PATCH_list.append(image_i_name + '-%i%i' % (i + rmv_edge, j + rmv_edge))
            PGL_list.append(geo_patch_center[idx])
            PGK_list.append(geo_patch_kappa[idx])
            PEPL_list.append([ls_w[j + rmv_edge], ls_h[i + rmv_edge], ls_w[j + rmv_edge + 1], ls_h[i + rmv_edge + 1]])
            # crop patch and save
            img_ij = img.crop(PEPL_list[-1])  # left,top,right,bottom
            img_ij.save(out_dir + '/%s%s' % (PATCH_list[-1], '.JPG'))

    # 3. return GL_l, and plot patch distribution map
    PGL_array, IGL_array = np.array(PGL_list), np.array(IGL_list)
    PEPL_array = np.array(PEPL_list)  # # patch edge pixel-location
    if show_plot:
        plt.figure(figsize=(8, 6))
        ax = plt.axes()
        s2 = plt.scatter(IGL_array[:, 1], IGL_array[:, 0], c='b', marker='*')
        s1 = plt.scatter(PGL_array[:, 1], PGL_array[:, 0], c='r', marker='.', s=10)

        x0, y0 = PGL_array[:, 1], PGL_array[:, 0]
        d_x = [degree_distance * np.sin(np.radians(-kappa)) for kappa in PGK_list]
        d_y = [degree_distance * np.cos(np.radians(-kappa)) for kappa in PGK_list]
        ax.quiver(x0, y0, d_x, d_y, color=(1, 0, 0, 0.4), width=0.002, angles='xy',
                  scale_units='xy', scale=1.1, headwidth=2.5, headlength=4.5)

        plt.title('Sparse Sampling, elm_d:%.1f' % elm_d)
        plt.legend((s2, s1), ('IGL', 'PGL'), loc='best')
        plt.margins(0.2)

        plt.tight_layout()
    inter_parms_path = out_dir + '/inter_parms/'
    if not os.path.exists(inter_parms_path):
        os.makedirs(inter_parms_path)
    np.save(inter_parms_path + '/IGL_array1.npy', IGL_array)  # image geo-location
    np.save(inter_parms_path + '/PGL_array1.npy', PGL_array)  # patch geo-location
    np.save(inter_parms_path + '/PEPL_array1.npy', PEPL_array)  # patch edge pixel-location
    np.save(inter_parms_path + '/elm_d1.npy', elm_d)  # eliminate d1-times of patch-distance-unit
    return {'PATCH_list': PATCH_list, 'PGL_array': PGL_array, 'PEPL_array': PEPL_array, 'IGL_array': IGL_array}


if __name__ == '__main__':
    img_dir1 = r'img/'
    out_dir1 = img_dir1 + '/patches/'
    PATCH1_info = patchSparseSampling(img_dir1, patch_num_rc=(5, 5), elm_d=0.8, rmv_edge=1,
                                      show_plot=True, out_dir=out_dir1)

