import os.path as osp
import mmcv
import numpy as np
import torch

import random

from ..builder import PIPELINES

@PIPELINES.register_module()
class SequenceShuffle(object):
    """

        Selects patches with high motion in the images of a frame window with length 'num_frames', as described in [Unsupervised Representation Learning by Sorting Sequences].

        Args:
        video_seqs (dict[dict]): dicitonary of input video sequences. Keys correspond to the video identifier (name); values are also dictionaries containing both their frames and corresponding optical flow fields between neighboring frames.
        num_frames (int): number of frames in a given time-window
        num_crops (int): number of crops in which the frames are split in order to select crops with the highest motion magnitudes.

        Returns:
        sorted_video_seq (list[list]): list of N-frame sequences (frame-windows), sorted by motion (optical flow-based)
    
    """

    def __init__(self, window_size=4):
        self.window_size = window_size


    def __call__(self, results):
        # MODIFY DESCRIPTION
        """Call functions to calculate motion and select crops with the highest motion in frames

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            results (dict), with the following modifications:
            'img' (list[torch.Tensor]): now stores the list of crops corresponding to the original images
            'crop_size' (list[int]): the dimensions of the cropped regions
        """

        new_order = []
        new_order_optflow = []
        positional_letters = ['a', 'b', 'c', 'd']
        shuffling_order = results['str_cls'][1:len(results['str_cls'])-1].split(',')

        # print(shuffling_order)

        # print("RANDOM CROP IMG SHAPE (1): ", results['img'][0].shape)
        # print("RANDOM CROP IMG SHAPE (2): ", results['img'][1].shape)
        # print("RANDOM CROP IMG SHAPE (3): ", results['img'][2].shape)
        # print("RANDOM CROP IMG SHAPE (4): ", results['img'][3].shape)

        for letter in shuffling_order:
            idx = positional_letters.index(letter)
            new_order.append(results['img'][idx])
            new_order_optflow.append(results['optflows'][idx])
        
        results['img'] = new_order
        results['optflows'] = new_order_optflow

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str