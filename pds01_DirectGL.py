# -*- coding: utf-8 -*-
"""
Created on 2022/11/22 11:03
To calculate the geodesic-distance between GCP and Observed-point in mono-photogrammetry,

1. Find images containing target GCP (using camera location and position {CAM_loc, CAM_pos})
2. Manually-Select observed-GCP-point in graph (collect point-of-interest pixel-coordinate {POI_coor})
3. Estimate the POI location {OBS_loc} and geodesic-distance {Geodesic_dis}.


!! Need Download for testing:
1. images data and put them at img/ (extra files at: )

@author: LU
"""
import os
from pyexiv2 import Image
from PIL import Image as Image2
import numpy as np
from haversine import inverse_haversine, Direction, Unit, haversine
import matplotlib.pyplot as plt
from geographiclib.geodesic import Geodesic
from tqdm import tqdm


class monoPhoto:
    """ About a single ortho-photo TAKEN FROM P1 camera"""

    def __init__(self, imgPath, focal_length=34.9023, sensor_sizes=np.array([35.0000, 23.3276])):
        """
        :param imgPath: path of one P1-image
        """
        self.img_path = imgPath
        self.img_dir, self.img_name = os.path.split(imgPath)
        self.param_dict = self._getParam(imgPath, focal_length, sensor_sizes)
        globals().update(self.param_dict)

    def _getParam(self, img_path, focal_length, sensor_sizes):
        """
        Parameters init.
        :param img_path:
        :return:
        """
        img1 = Image(img_path, encoding='gbk')
        img1_xmp1, img1_exif = img1.read_xmp(), img1.read_exif()
        lat_, lon_ = img1_exif['Exif.GPSInfo.GPSLatitude'].split(), img1_exif['Exif.GPSInfo.GPSLongitude'].split()
        lat, lon = eval(lat_[0]) + eval(lat_[1]) / 60 + eval(lat_[2]) / 3600, eval(lon_[0]) + eval(lon_[1]) / 60 + eval(
            lon_[2]) / 3600
        kappa = -float(img1_xmp1['Xmp.drone-dji.FlightYawDegree'])
        cos_theta, sin_theta = np.cos(np.radians(kappa)), np.sin(np.radians(kappa))
        cs_trans_M = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])  # 坐标投影到正北和正东方向(geo_pixel坐标)
        param_dict = {'img_name': self.img_name.split('.')[0],
                      'img_loc': [lat, lon],
                      'kappa': kappa,
                      'relative_height': eval(img1_xmp1['Xmp.drone-dji.RelativeAltitude']),
                      'focal_len': focal_length,  # eval(img1_exif['Exif.Photo.FocalLength']),   #
                      'h': eval(img1_exif['Exif.Photo.PixelYDimension']),
                      'w': eval(img1_exif['Exif.Photo.PixelXDimension']),
                      'sensor_size': sensor_sizes,  # np.array([35.900, 24.000]),    #
                      'cs_trans_M': cs_trans_M}
        return param_dict

    def gcp_coord_calc(self, gcp_loc, s_ratio=0.8):
        """
        Calculate pixel-coordinate of gcp_loc(geo-location).
        :param gcp_loc: WGS84 geo-location of point to be checked
        :param s_ratio: image-edge-shrink ratio
        :return:pixel-coord, if: in_region; else: False
        """
        dis_geo = haversine(img_loc, gcp_loc, unit=Unit.METERS)
        dis_pix = (dis_geo / (sensor_size / [w, h]) / (relative_height / focal_len)).max()
        # pixel distance of GCP and Image-center
        azi_cor = Geodesic.WGS84.Inverse(img_loc[0], img_loc[1], gcp_loc[0], gcp_loc[1], )['azi1']
        # (azimuth) Angle that reverse img to the North
        angle_tar = azi_cor + kappa  # Angle that reverse to the image Y-axis(👆)
        if angle_tar >= 180:
            angle_tar -= 360
        elif angle_tar <= -180:
            angle_tar += 360
        cent2tar_vec = np.array(
            (dis_pix * np.sin(np.radians(angle_tar)), -dis_pix * np.cos(np.radians(angle_tar))))
        # pixel-vector from image-center to GCP
        tar_pix_coord = cent2tar_vec + np.array([w / 2, h / 2])
        if abs(cent2tar_vec[0]) < w / 2 * s_ratio and abs(cent2tar_vec[1]) < h / 2 * s_ratio:
            return tar_pix_coord
        else:
            print('! tar_pix_coord out of boundary:', np.around(tar_pix_coord), '/',
                  (w / 2 * s_ratio, h / 2 * s_ratio))
            return False

    def interactDirectGeoLocate(self, gcp_loc=None, s_ratio=0.8):
        if gcp_loc is None:
            gcp_loc = img_loc

        def geo_coord_calc(tar_pix_coord, ):
            """
            Calculate geo-location of selected point(given-pixel-coordinate)

            tar_pix_coord: image-pixel-coordinate of selected point
            :return tar_coord: WGS84 coordinate of selected point
             """
            # %% point geo-coord calculation
            tar_mm_vec = np.array([w / 2, h / 2]) - np.array([tar_pix_coord])
            # pixel-vector from image-center to select-point
            tar_mm_vec[:, 0] = -tar_mm_vec[:, 0]
            # image-coordinate/geo-coordinate are left/right -handed system (Y axis is flipped around the X axis)
            tar_geo_coord_vector = np.dot(cs_trans_M, tar_mm_vec.T)  # project to the North and East
            tar_geo_distance = (sensor_size / [w, h]) * tar_geo_coord_vector * (relative_height / focal_len)
            # convert to geo-lcoation using collinear relationship
            tar_coord = [
                inverse_haversine(img_loc, tar_geo_distance.T[0, 1], Direction.NORTH, unit=Unit.METERS)[0],
                inverse_haversine(img_loc, tar_geo_distance.T[0, 0], Direction.EAST, unit=Unit.METERS)[1]]
            return tar_coord

        img1 = Image2.open(self.img_path)
        fig, ax = plt.subplots()
        im = plt.imshow(img1)
        gcp_pix_coord = self.gcp_coord_calc(gcp_loc, s_ratio=s_ratio)
        plt.scatter(gcp_pix_coord[0], gcp_pix_coord[1])
        plt.title('%s, GCP-Loc: %.8f,%.8f' % (img_name, gcp_loc[0], gcp_loc[1]))
        plt.axis('off')

        # annotate: add arrow
        annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        # Add Param
        def update_annot(event):
            pos = [event.xdata, event.ydata]
            geo_pos = geo_coord_calc(list(pos))
            annot.xy = pos
            annot.set_text('%.8f\n%.8f' % (geo_pos[0], geo_pos[1]))
            annot.get_bbox_patch().set_facecolor(np.random.rand(3))
            annot.get_bbox_patch().set_alpha(0.4)
            print('%s   %.8f,%.8f   %.1f    %i,%i   %.8f,%.8f   %.2f' % (img_name, img_loc[0], img_loc[1], kappa,
                                                                         int(pos[0]), int(pos[1]), geo_pos[0],
                                                                         geo_pos[1],
                                                                         haversine(geo_pos, gcp_loc, unit=Unit.METERS)))
            return geo_pos

        # update hover
        def hover(event):
            geo_pos = update_annot(event=event)
            annot.set_visible(True)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", hover)
        plt.show()


def calcNearest(img_dir, tar_loc, ):
    """
    Calculate the K-nearest neighbors for tar_loc
    :param img_dir: directory of available imgs
    :param tar_loc: geo-location of target point
    :return: sorted imgs index according to distance
    """
    imgs_name = [img for img in os.listdir(img_dir) if img.endswith('.JPG')]
    img_num, gcp_num = len(imgs_name), len(tar_loc)
    geo_coord = np.zeros((img_num, 2))
    geo_distance_matrix = np.zeros((img_num, gcp_num))
    for i in tqdm(range(img_num)):
        img1 = Image(img_dir + imgs_name[i])
        img1_xmp1, img1_exif = img1.read_xmp(), img1.read_exif()
        lat_, lon_ = img1_exif['Exif.GPSInfo.GPSLatitude'].split(), img1_exif['Exif.GPSInfo.GPSLongitude'].split()
        geo_coord[i] = eval(lat_[0]) + eval(lat_[1]) / 60 + eval(lat_[2]) / 3600, \
                       eval(lon_[0]) + eval(lon_[1]) / 60 + eval(lon_[2]) / 3600
        for j in range(gcp_num):
            geo_distance_matrix[i][j] = haversine(geo_coord[i], tar_loc[j], unit=Unit.METERS)
    near_imgs_sorted = [geo_distance_matrix[:, j].argsort() for j in range(gcp_num)]
    return near_imgs_sorted


if __name__ == '__main__':
    # testing image and ground control point
    img_path = r"img/DJI_20220727102043_0046.JPG"
    gcp_loc = [30.07485181, 119.923803]

    mp1 = monoPhoto(img_path, focal_length=34.9023, sensor_sizes=np.array([35.0000, 23.3276]))
    shrink_ratio = 0.8      # image-edge-shrink ratio
    mp1.gcp_coord_calc(gcp_loc=gcp_loc, s_ratio=shrink_ratio)  # test if gcp in image view field
    # interactively direct geo-locating
    mp1.interactDirectGeoLocate(gcp_loc=gcp_loc, s_ratio=shrink_ratio)
