import os.path as osp
import mmcv
import numpy as np
import torch

from ..builder import PIPELINES

@PIPELINES.register_module()
class MotionAwareCropSelection(object):
    """

        Selects patches with high motion in the images of a frame window with length 'num_frames', as described in [Unsupervised Representation Learning by Sorting Sequences].

        Args:
        video_seqs (dict[dict]): dicitonary of input video sequences. Keys correspond to the video identifier (name); values are also dictionaries containing both their frames and corresponding optical flow fields between neighboring frames.
        num_frames (int): number of frames in a given time-window
        num_crops (int): number of crops in which the frames are split in order to select crops with the highest motion magnitudes.

        Returns:
        sorted_video_seq (list[list]): list of N-frame sequences (frame-windows), sorted by motion (optical flow-based)
    
    """

    def __init__(self, window_size=4, num_crops=4):
        self.window_size = window_size
        self.num_crops = num_crops


    def __call__(self, results):
        """Call functions to calculate motion and select crops with the highest motion in frames

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            results (dict), with the following modifications:
            'img' (list[torch.Tensor]): now stores the list of crops corresponding to the original images
            'crop_size' (list[int]): the dimensions of the cropped regions
        """

        # calculating the kernel sizes, according to the num_crops required
        # notice that we aim at extracting the same number of crops in vertical and horizontal directions
        kh, kw = results['img_shape'][0]//(self.num_crops//2), results['img_shape'][1]//(self.num_crops//2)
        
        print("AQUIEEE", kh, kw)
        print("\n\n\n")
        # print(results)
        print("\n\n\n")

        dh, dw = kh, kw # stride values are equal to kernel values (crop dimension) so that they don't overlap

        # computes magnitudes from optical flow displacements in x and y axis
        extract_magnitude = lambda f: torch.sqrt(torch.as_tensor(f[..., 0])**2+torch.as_tensor(f[..., 1])**2) # computes magnitude from optical flow
        
        # crop extraction
        unfold = lambda img: img.unfold(0, kh, dh).unfold(1, kw, dw) # returns a tensor with (i,j,C,kh,kw) dimension, where i and j are indices to the extracted crops
        
        magnitudes = list(map(extract_magnitude, results['optflows'])) # optflows (list[np.ndarray] of size window_size)
        
        # stack the magnitude maps together
        magnitudes = torch.stack(magnitudes) # a stack of window-size maps : (window_size, h, w), where h, w are the original dimensions

        # sum the magnitudes, in order to get rid of the additional dimension generated by the stack operation
        magnitudes = torch.sum(magnitudes, dim=0)

        print("MAGNITUDES SHAPE: ", magnitudes.shape)
        print("\n\n\n")


        # generates patches (windows) from the input images
        patches_mags = list(map(unfold, magnitudes)) # returns a list of (window_size) tensors with shape (i,j,kh,kw)
        unfold_shape = patches_mags[0].size()

        
        #transform crops into list
        crops_magnitude = [patches_mags[i,j,...] for i in range(unfold_shape[0]) for j in range(unfold_shape[1])]
        
        # sum the different crops in order to select the one with highest motion
        sum_magnitude = [torch.sum(crop) for crop in crops_magnitude]
        selected_crop = torch.argmax(sum_magnitude)
        # according to the selected crop, we have different indices to access the return from unfold
        if selected_crop == 0:
            i, j = 0, 0
        elif selected_crop == 1:
            i, j = 0, 1
        elif selected_crop == 2:
            i, j = 1, 0
        else:
            i, j = 1, 1

        # Finally, we select the corresponding crop in all four frames of the window
        # cropping imgs
        unfold_imgs = lambda img: img.unfold(1, kh, dh).unfold(2, kw, dw) # remember that channels come first in images; the result will be of shape (C,i,j,kh,kw)
        cropped_imgs = list(map(unfold_imgs, results['img'])) # list of (C,i,j,kh,kw) elements
        # selecting the appropriate crop, according to our previous reasoning 
        list_selected_crops = [crop[:,i,j,...] for crop in cropped_imgs]
                    
                    
        # Finally!!! We replace the list of images under 'imgs' by their selected crops :D
        results['img'] = list_selected_crops
        results['crop_size'] = [kh,kw]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str