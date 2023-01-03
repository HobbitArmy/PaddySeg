# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset
import mmcv
from PIL import Image
import numpy as np


@DATASETS.register_module()
class PaddySeg(CustomDataset):
    """Pascal PaddySeg dataset.

    Args:
        split (str): Split txt file for Pascal PaddySeg.
    """

    CLASSES = ('others', 'seedling', 'jointing', 'heading', 'filling')
    PALETTE = [[0, 0, 0], [255, 255, 127], [0, 85, 255], [255, 105, 253], [255, 0, 0]]

    # CLASSES = ('water', 'traffic', 'building', 'farmland',
    # 'grassland', 'woodland', 'ground', 'others')
    # PALETTE = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    # [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0]]

    def __init__(self, split, **kwargs):
        super(PaddySeg, self).__init__(
            img_suffix='.JPG', seg_map_suffix='.png',
            reduce_zero_label=False, split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None

    # @staticmethod
    # def _convert_to_label_id(result):
    #     """Convert trainId to id for cityscapes."""
    #     if isinstance(result, str):
    #         result = np.load(result)
    #     import cityscapesscripts.helpers.labels as CSLabels
    #     result_copy = result.copy()
    #     for trainId, label in CSLabels.trainId2label.items():
    #         result_copy[result == trainId] = label.id
    #
    #     return result_copy

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):
            # if to_label_id:
            #     result = self._convert_to_label_id(result)
            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            output = Image.fromarray(result.astype(np.uint8)).convert('P')
            # import cityscapesscripts.helpers.labels as CSLabels
            # palette = np.zeros((len(CSLabels.id2label), 3), dtype=np.uint8)
            # for label_id, label in CSLabels.id2label.items():
            #     palette[label_id] = label.color

            # output.putpalette(palette)
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for Cityscapes
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """
        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)

        return result_files
