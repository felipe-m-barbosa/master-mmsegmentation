# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import mmcv
import numpy as np
import torch
import cityscapesscripts.helpers.labels as CSLabels

from ..builder import PIPELINES

import mmcv

import time

@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        # MODIFY FROM HERE
        if 'video_name' in results['img_info']: 
            images_list = []
            optflows_list = []
            for idx in range(len(results['img_info']['img_filenames'])):
                filename = osp.join(results['img_info']['video_name'], results['img_info']['img_filenames'][idx])
                optflow_filename = results['img_info']['optflow_filenames'][idx]

                img_bytes = self.file_client.get(filename)
                img = mmcv.imfrombytes(
                    img_bytes, flag=self.color_type, backend=self.imdecode_backend)
                if self.to_float32:
                    img = img.astype(np.float32)
                
                images_list.append(torch.as_tensor(img))
                
                optflow = mmcv.flowread(optflow_filename) if optflow_filename is not None else None
                optflows_list.append(optflow)
            
            # remember that img is a list of images, actually
            results['img'] = images_list # we keep the singular in order to avoid changing the downstream code
            
            # print("LOADING IMG SHAPE (1): ", results['img'][0].shape)
            # print("LOADING IMG SHAPE (2): ", results['img'][1].shape)
            # print("LOADING IMG SHAPE (3): ", results['img'][2].shape)
            # print("LOADING IMG SHAPE (4): ", results['img'][3].shape)

            results['optflows'] = optflows_list

            results['gt_semantic_seg'] = results['ann_info']

            results['img_shape'] = img.shape
            results['ori_shape'] = img.shape
            # Set initial values for default meta_keys
            results['pad_shape'] = img.shape
            results['scale_factor'] = 1.0
            num_channels = 1 if len(img.shape) < 3 else img.shape[2]
            results['img_norm_cfg'] = dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False)

        else:

            if results.get('img_prefix') is not None:
                filename = osp.join(results['img_prefix'],
                                    results['img_info']['filename'])
            else:
                filename = results['img_info']['filename']
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            if self.to_float32:
                img = img.astype(np.float32)

            results['filename'] = filename
            results['ori_filename'] = results['img_info']['filename']
            results['img'] = img
            results['img_shape'] = img.shape
            results['ori_shape'] = img.shape
            # Set initial values for default meta_keys
            results['pad_shape'] = img.shape
            results['scale_factor'] = 1.0
            num_channels = 1 if len(img.shape) < 3 else img.shape[2]
            results['img_norm_cfg'] = dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False)
        
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        # load depth annotations, if needed
        if results['img_info'].get('gt_depth', None) is not None:
            filename = results['img_info']['gt_depth']['filename']
        
            img_bytes = self.file_client.get(filename)
            gt_depth = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend)

            results['gt_depth'] = gt_depth


        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        
        # convert label ids to trainIds
        lbl_ids = gt_semantic_seg[:,:,0] # isolate red channel (it could be any channel)


        train_ids = np.zeros_like(lbl_ids)
        for lbl_id, label in CSLabels.id2label.items():
          train_ids[lbl_ids==lbl_id] = label.trainId

        gt_semantic_seg = train_ids

        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


# CUSTOM PIPELINES
# MMFLOW
def read_flow(name: str) -> np.ndarray:
    """Read flow file with the suffix '.flo'.
    This function is modified from
    https://lmb.informatik.uni-freiburg.de/resources/datasets/IO.py
    Copyright (c) 2011, LMB, University of Freiburg.
    Args:
        name (str): Optical flow file path.
    Returns:
        ndarray: Optical flow
    """

    with open(name, 'rb') as f:

        header = f.read(4)
        if header.decode('utf-8') != 'PIEH':
            raise Exception('Flow file header does not contain PIEH')

        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()

        flow = np.fromfile(f, np.float32, width * height * 2).reshape(
            (height, width, 2))

    return flow



@PIPELINES.register_module()
class newLoadImageFromFile(object):
    """Load an image from file.
    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).
    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.
        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.
        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        
        # sequence images
        try:
            s1_filename = results['img_info']['s1']['filename']
            s1_img_bytes = self.file_client.get(s1_filename)
            s1_img = mmcv.imfrombytes(
                s1_img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            
            s2_filename = results['img_info']['s2']['filename']
            s2_img_bytes = self.file_client.get(s2_filename)
            s2_img = mmcv.imfrombytes(
                s2_img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except:
            s1_filename = None
            s2_filename = None
        
        # optflow
        if not isinstance(results, list):
            if results['img_info'].get(['optflow'], None) is not None:
                optflow_filename = results['img_info']['optflow']['filename']
                optflow_img = read_flow(optflow_filename)
            else:
                optflow_filename = None
                optflow_img = None
        else:
            optflow_filename = None
            optflow_img = None


        if self.to_float32:
            img = img.astype(np.float32)

            if s1_filename is not None:
                s1_img = s1_img.astype(np.float32)
            
            if s2_filename is not None:
                s2_img = s2_img.astype(np.float32)
            
            if optflow_filename is not None:
                optflow_img = optflow_img.astype(np.float32)


        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        
        if s1_filename is not None:
            results['s1_filename'] = s1_filename
            results['s1_ori_filename'] = results['img_info']['s1']['filename']
            results['s1_img'] = s1_img
            results['s1_img_shape'] = s1_img.shape
            results['s1_ori_shape'] = s1_img.shape
            # Set initial values for default meta_keys
            results['s1_pad_shape'] = s1_img.shape
            results['s1_scale_factor'] = 1.0
            results['s1_img_norm_cfg'] = dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False)
            

            results['s2_filename'] = s2_filename
            results['s2_ori_filename'] = results['img_info']['s2']['filename']
            results['s2_img'] = s2_img
            results['s2_img_shape'] = s2_img.shape
            results['s2_ori_shape'] = s2_img.shape
            # Set initial values for default meta_keys
            results['s2_pad_shape'] = s2_img.shape
            results['s2_scale_factor'] = 1.0
            results['s2_img_norm_cfg'] = dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False)

        if optflow_filename is not None:
            results['optflow_filename'] = optflow_filename
            results['optflow_ori_filename'] = optflow_filename
            results['optflow'] = optflow_img
            results['optflow_shape'] = optflow_img.shape if optflow_img is not None else None
            results['optflow_ori_shape'] = optflow_img.shape if optflow_img is not None else None
            # Set initial values for default meta_keys
            results['optflow_pad_shape'] = optflow_img.shape if optflow_img is not None else None
            results['optflow_scale_factor'] = 1.0
            results['optflow_norm_cfg'] = dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


# NOT NEEDED, SINCE IT IS IDENTICAL TO THE ORIGINAL LOAD ANNOTATIONS
@PIPELINES.register_module()
class newLoadAnnotations(object):
    """Load annotations for semantic segmentation.
    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.
        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.
        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        

        # print(results.keys())

        # load depth annotations, if needed
        if results.get('depth_info', None) is not None:
            filename = results['depth_info']['filename']
        
            img_bytes = self.file_client.get(filename)
            gt_depth = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend)

            results['gt_depth'] = gt_depth
        

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        
        # convert label ids to trainIds
        lbl_ids = gt_semantic_seg[:,:,0] # isolate red channel (it could be any channel)

        # first, convert labelid to train id
        train_ids = np.zeros_like(lbl_ids)
        for lbl_id, label in CSLabels.id2label.items():
          train_ids[lbl_ids==lbl_id] = label.trainId

        gt_semantic_seg = train_ids

        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str