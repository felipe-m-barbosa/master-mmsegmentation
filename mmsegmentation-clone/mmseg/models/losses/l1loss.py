# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss



@LOSSES.register_module()
class L1Loss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_l1',
                 avg_non_ignore=False,
                 ignore_value=0.0):
        super(L1Loss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.avg_non_ignore = avg_non_ignore
        self.ignore_value = ignore_value
        

        self.cls_criterion = nn.L1Loss()
        self._loss_name = loss_name

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                label,
                **kwargs):
        """Forward function."""


        # print("LOSS - PRED SHAPE: ", cls_score.shape)
        # print("LOSS - TARGET SHAPE: ", label.shape)

        # print(kwargs.keys())

        if self._loss_name != "loss_depth":
            label = torch.argmax(label, dim=1) # this is supposed to work

        # print(f"label dim {label.shape}")
        # print(f"cls_score dim {cls_score.shape}")

        if self._loss_name == "loss_depth":
            # where depth gt is 0 (invalid), also turn predicted depths into 0,
            # so that they do not have impact on the loss
            cls_score = torch.where(label == self.ignore_value, 0, cls_score)

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label)
        
        loss_cls = weight_reduce_loss(
        loss_cls, weight=None, reduction=self.reduction, avg_factor=None)

        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
