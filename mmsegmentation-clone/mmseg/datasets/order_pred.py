from .builder import DATASETS
from .custom import CustomDataset

import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg.utils import get_root_logger
from PIL import Image
import random
import os

import torch


@DATASETS.register_module()
class OrderPredDataset(CustomDataset):
    """OrderPred Dataset.

    Intended to be used in order prediction tasks for unsupervised pretraining, aiming at improving the temporal
    awareness of a given model.
    
    Args:

    """

    CLASSES = ('{a,b,c,d}', '{a,c,b,d}', '{a,c,d,b}', '{a,b,d,c}', '{a,d,b,c}', '{a,d,c,b}',
               '{b,a,c,d}', '{b,a,d,c}', '{b,c,a,d}', '{b,d,a,c}', '{c,a,b,d}',
               '{c,b,a,d}')
    
    # the classes, but represented as one-hot tensors for loss calculation
    ONE_HOT_CLASSES = torch.nn.functional.one_hot(torch.arange(len(CLASSES))).to(torch.float)

    num_samples_per_class = {c:0 for c in CLASSES}

    PALETTE = None

    def __init__(self, img_suffix='.png',
                 seg_map_suffix='.png',
                 window_size=4,
                 **kwargs):
        super(OrderPredDataset, self).__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
        self.img_dirs = self.img_dir
        self.optflow_dirs = kwargs['optflow_dir']
        self.window_size = window_size

        # if not(isinstance(self.img_dirs, list)):
        #     self.img_dirs = [self.img_dirs]
        
        # if not(isinstance(self.optflow_dirs, list)):
        #     self.optflow_dir = [self.optflow_dir]

        # print("AQUI")
        # print(self.img_dirs)
        # print(self.optflow_dirs)
        # print("\n\n")

        # load annotations
        self.img_infos = self.load_annotations(self.img_dirs, self.optflow_dirs)


    # here, we only set the filenames
    def load_annotations(self, img_dirs, optflow_dirs):
        """Load image and optical flow infos from directory.
        Args:
            img_dir (str): Path to image directory
            optflow_dir (str): Path to optical flow directory corresponding to the images
            
        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []

        for img_dir, optflow_dir in zip(img_dirs, optflow_dirs):

            
            filenames = sorted(os.listdir(img_dir))
            optflow_filenames = sorted(os.listdir(optflow_dir))

            # loop over filenames, considering the window size passed as argument
            for i in range(len(filenames)//self.window_size): # guarantees that we don't access windows shorter than the required window_size
                window_list = filenames[(i)*self.window_size:(i+1)*self.window_size]
                optflow_list = [osp.join(optflow_dir, img.replace('leftImg8bit.png', 'opt_flow.flo')) for img in window_list]

                optflow_exist = [osp.exists(f) for f in optflow_list]

                # test if all images have a corresponding optflow
                if not all(optflow_exist):
                    continue


                img_info = dict(filename=window_list[0]) # the name of first frame in the window
                img_info['video_name'] = img_dir # we use the original img_dir path as the video name
                img_info['img_filenames'] = window_list
    

                img_info['optflow_filenames'] = optflow_list
                if i == len(filenames)//self.window_size - 1: # in the last sequence, we need to make an additional verification, because the last image of the video sequence doesn't have an associated optical flow 
                    idx = filenames.index(window_list[self.window_size-1]) # find index corresponding to last image in the window
                    if idx == len(filenames)-1:
                        img_info['optflow_filenames'][self.window_size-1] = None
                
                # chooses the sequence order for shuffling
                # ensuring that we get a balance class distribution
                # class_idx = random.randint(0,len(self.CLASSES)-1)
                
                str_cls = min(self.num_samples_per_class, key=self.num_samples_per_class.get)
                img_info['str_cls'] = str_cls # the class name (for visualization and debugging)
                # although labels are now one-hot encoded tensors representing the sequence order, we keep the name as 'gt_semantic_seg' to avoid modifying the downstream code
                class_idx = list(self.CLASSES).index(str_cls)
                img_info['ann'] = self.ONE_HOT_CLASSES[class_idx] # the one-hot encoded class

                self.num_samples_per_class[str_cls] += 1

                img_infos.append(img_info)

        img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        
        return img_infos