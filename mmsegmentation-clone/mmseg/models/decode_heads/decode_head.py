# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy
from torchmetrics import JaccardIndex
import torch.nn.functional as F


class BaseDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        out_channels (int): Output channels of conv_seg.
        threshold (float): Threshold for binary segmentation in the case of
            `out_channels==1`. Default: None.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict | Sequence[dict]): Config of decode loss.
            The `loss_name` is property of corresponding loss function which
            could be shown in training log. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
             e.g. dict(type='CrossEntropyLoss'),
             [dict(type='CrossEntropyLoss', loss_name='loss_ce'),
              dict(type='DiceLoss', loss_name='loss_dice')]
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255.
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 out_channels=None,
                 threshold=None,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv_seg'))):
        super(BaseDecodeHead, self).__init__(init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index

        self.ignore_index = ignore_index
        self.align_corners = align_corners

        if out_channels is None:
            if num_classes == 2:
                warnings.warn('For binary segmentation, we suggest using'
                              '`out_channels = 1` to define the output'
                              'channels of segmentor, and use `threshold`'
                              'to convert seg_logist into a prediction'
                              'applying a threshold')
            out_channels = num_classes

        if out_channels != num_classes and out_channels != 1:
            raise ValueError(
                'out_channels should be equal to num_classes,'
                'except binary segmentation set out_channels == 1 and'
                f'num_classes == 2, but got out_channels={out_channels}'
                f'and num_classes={num_classes}')

        if out_channels == 1 and threshold is None:
            threshold = 0.3
            warnings.warn('threshold is not defined for binary, and defaults'
                          'to 0.3')
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.threshold = threshold

        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, **kwargs):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the arhitecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        if self.__class__.__name__ == 'FCNDepthHead':
            seg_logits, depth_pred = self(inputs)

            losses = self.losses(seg_logits, gt_semantic_seg, depth_pred=depth_pred, gt_depth=kwargs['gt_depth'])

        else:
            seg_logits = self(inputs)
        
            # verify if sequence images where passed
            if 's1' in kwargs: # we assume that s1 and s2 always appear together
                s1_logits = self(kwargs['s1'])
                s2_logits = self(kwargs['s2'])

                losses = self.losses(seg_logits, gt_semantic_seg, s1_logits=s1_logits, s2_logits=s2_logits, opt_flow=kwargs['opt_flow'], s1=kwargs['s1'], s2=kwargs['s2'])

            else:
                if isinstance(inputs, list): # order prediction task (NOT SURE IF THIS IS SUFFICIENT -> CHECK)
                    losses = self.losses(seg_logits, gt_semantic_seg, is_order_pred=True)
                else:
                    losses = self.losses(seg_logits, gt_semantic_seg)

        
        return losses


    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label, **kwargs):
        """Compute segmentation loss."""
        loss = dict()
        # print('Seg logit shape: ', seg_logit.shape)
        # print('Seg label shape: ', seg_label.shape)
        
        # THIS IS UNNECESSARY IN ORDER PREDICTION TASK
        if not(kwargs.get('is_order_pred', False)):
            seg_logit = resize(
                input=seg_logit,
                size=seg_label.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        
        # print('Seg logit shape: ', seg_logit.shape)

        if 's1_logits' in kwargs:
            s1_logits = resize(
                input=kwargs['s1_logits'],
                size=seg_label.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        
        if 's1' in kwargs:
            s1 = resize(
                input=kwargs['s1'][0],
                size=seg_label.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        if 's2_logits' in kwargs:
            s2_logits = resize(
                input=kwargs['s2_logits'],
                size=seg_label.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        
        if 's2' in kwargs:
            s2 = resize(
                input=kwargs['s2'][0],
                size=seg_label.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)


        if 'gt_depth' in kwargs:
            gt_depth = kwargs['gt_depth']
            depth_pred = kwargs['depth_pred']

            # print("losses - gt_depth shape: ", gt_depth.shape)
            # print("losses - depth_pred shape: ", depth_pred.shape)
            
            # resize preds and gts, if necessary, to the same dimensions
            dim_gt_depth = gt_depth.shape[2]*gt_depth.shape[3] # H*W
            dim_depth_pred = depth_pred.shape[2]*depth_pred.shape[3] # H*W

            # print('dim_gt_depth: ', dim_gt_depth)
            # print('dim_depth_pred: ', dim_depth_pred)

            # print('gt_depth dtype: ', gt_depth.dtype)
            # print('depth_pred dtype: ', depth_pred.dtype)


            if dim_gt_depth > dim_depth_pred:
                depth_pred = resize(
                input=depth_pred,
                size=gt_depth.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            elif dim_depth_pred > dim_gt_depth:
                gt_depth = resize(
                input=gt_depth,
                size=depth_pred.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)


        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None

        if not ('is_order_pred' in kwargs):
            seg_label = seg_label.squeeze(1) # PQ ISSO? NO CASO DA DETECÇÃO DE ORDEM NÃO PRECISA...

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            # temporal consistency loss
            if loss_decode.loss_name == 'loss_tc':

                # considering optical flow from t+1 to t (backward)
                # input_2 = torch.argmax(kwargs['s2_logits'], dim=1)
                input_1 = s1_logits
                input_2 = s2_logits

                tmp_loss = loss_decode(
                    input_1,
                    input_2,
                    weight=seg_weight,
                    ignore_index=self.ignore_index,
                    opt_flow = kwargs['opt_flow'],
                    gt_labels=seg_label,
                    s1 = s1,
                    s2 = s2)
            elif loss_decode.loss_name == 'loss_depth':
                tmp_loss = loss_decode(
                    depth_pred,
                    gt_depth
                ) # are there any missing arguments?
                
            else:
                input_1 = seg_logit
                input_2 = seg_label

                tmp_loss = loss_decode(
                    input_1,
                    input_2,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = tmp_loss
            else:
                loss[loss_decode.loss_name] += tmp_loss

        # print("SEG LOGITS SHAPE: ", seg_logit.shape)

        loss['acc_seg'] = accuracy(
            seg_logit, seg_label, ignore_index=self.ignore_index)

        if seg_logit.shape[1] != 12: # if not order prediction task

            jaccard = JaccardIndex(task='multiclass', num_classes=self.num_classes, ignore_index=255).to('cuda')
            seg_logit = F.softmax(seg_logit, dim=1) # loss function explodes if not applying softmax to logits
            # print(seg_logit.shape)
            
            loss['miou'] = jaccard(seg_logit, seg_label) # what you put in loss dict will be shown by the TextLogger

        return loss
