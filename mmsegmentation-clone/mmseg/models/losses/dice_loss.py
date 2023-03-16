# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/LikeLy-Journey/SegmenTron/blob/master/
segmentron/solver/loss.py (Apache-2.0 License)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv

from ..builder import LOSSES
from .utils import get_class_weight, weighted_loss

# not necessary, since grid_sample already implements this (change mode from 'bilinear' to 'nearest')
# def nearest_sample(x): # such as described in [Unsupervised temporal consistency metric for video segmentation in highly-automated driving]
#     return torch.floor(x+0.5)

def warp(x, flo, inp1, inp2):
    """
    Args:
        x: the input prediction, to be warped using the optical flow
        flo: optical flow field, with shape (B, H, W, C)
            C accounts for the number of channels, 
              which correspond to the pixel displacements in x and y directions
    """

    B,C,H,W = x.size()

    xx = torch.arange(0,W).view(1,-1).repeat(H,1) # H lines, each with W elements (int)
    yy = torch.arange(0,H).view(-1,1).repeat(1,W) # W columns, each with H elements (int)
     
    xx = xx.view(1,H,W,1).repeat(B,1,1,1) # change shape and repeat over the batch dimension
    yy = yy.view(1,H,W,1).repeat(B,1,1,1) # change shape and repeat over the batch dimension

    grid = torch.cat((xx,yy),3).float() # concatenates the given sequence of seq tensors in the given dimension
    # grid.shape: (B,H,W,2), where 2 corresponds to [x_idx, y_idx]

    if x.is_cuda:
        grid = grid.cuda()
        flo = flo.cuda()

    # vgrid = Variable(grid) + flo # sums the flow field displacements over x and y
    vgrid = grid + flo # adds the flow field displacements over x and y

    # print(flo)

    # NEAREST IMPLEMENTATION IS MISSING, SUCH AS DESCRIBED IN [An Unsupervised Temporal Consistency (TC) Loss to Improve the Performance of Semantic Segmentation Networks]

    ## scale grid to [-1,1]
    vgrid[:,:,:,0] = 2.0*vgrid[:,:,:,0].clone()/max(W-1,1)-1.0 # x
    vgrid[:,:,:,1] = 2.0*vgrid[:,:,:,1].clone()/max(H-1,1)-1.0 # y
     
    #  x = x.permute(0,3,1,2)
    x = x.type(torch.float32)

    #  print(f"Img type: {x.dtype}")
    #  print(f"Grid type: {vgrid.dtype}")

    # WARPING
    output = torch.nn.functional.grid_sample(x, vgrid) # by default, grid sample works in bilinear mode (nearest is also possible)
    
    # VALIDITY MASK
    # this implementation only accounts for misaligned borders
    # that is, occlusions caused by regions in the image borders
    # mask = torch.autograd.Variable(torch.ones(x.size()))

    # if x.is_cuda:
    #     mask = mask.cuda()

    # mask = torch.nn.functional.grid_sample(mask, vgrid)
    
    # mask[mask<0.9999]=0
    # mask[mask>0]=1

    # OCCLUSION MASK, SUCH AS IN [An Unsupervised Temporal Consistency (TC) Loss to Improve the Performance of Semantic Segmentation Networks]
    mask = torch.exp(-torch.norm(inp1 - inp2, p=1, dim=1))
    mask = mask.unsqueeze(1).repeat(1,output.shape[1],1,1)

    # VISIBILITY MASK, SUCH AS IN [Learning Blind Video Temporal Consistency]

    output = output*mask
    output = output.type(torch.float32)

    return output, mask


@weighted_loss
def dice_loss(pred,
              target,
              valid_mask,
              smooth=1,
              exponent=2,
              class_weight=None,
              ignore_index=255):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            dice_loss = binary_dice_loss(
                pred[:, i],
                target[..., i],
                valid_mask=valid_mask,
                smooth=smooth,
                exponent=exponent)
            if class_weight is not None:
                dice_loss *= class_weight[i]
            total_loss += dice_loss
    return total_loss / num_classes


@weighted_loss
def binary_dice_loss(pred, target, valid_mask, smooth=1, exponent=2, **kwargs):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth

    return 1 - num / den


@LOSSES.register_module()
class DiceLoss(nn.Module):
    """DiceLoss.

    This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.

    Args:
        smooth (float): A float number to smooth loss, and avoid NaN error.
            Default: 1
        exponent (float): An float number to calculate denominator
            value: \\sum{x^exponent} + \\sum{y^exponent}. Default: 2.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        ignore_index (int | None): The label index to be ignored. Default: 255.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_dice'.
    """

    def __init__(self,
                 smooth=1,
                 exponent=2,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_dice',
                 **kwargs):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        # warp prediction from t to t+1 (used in tc_loss)

        if 'tc' in self._loss_name:
            opt_flow = kwargs['opt_flow']

            # using mmcv built-in flow warp
            # print("OPTFLOW SHAPE: ", opt_flow.shape)
            preds = []
            for p, of in zip(pred, opt_flow):
                p = p.squeeze(0)
                of = of.squeeze(0)
                print('P SHAPE: ', p.shape)
                print('OF SHAPE: ', of.shape)
                print(type(p))
                print(type(of))
                pw = mmcv.flow_warp(p, of)
                p = p.unsqueeze(0)
                preds.append(pw)

            pred = torch.stack(preds, 0)

            # pred, _ = warp(pred, opt_flow, inp1=kwargs['s1'], inp2=kwargs['s2'])

        pred = F.softmax(pred, dim=1)
        target = F.softmax(target, dim=1)

        if 'tc' in self._loss_name:
            target = torch.argmax(target, dim=1)
        
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()

        loss = self.loss_weight * dice_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            reduction=reduction,
            avg_factor=avg_factor,
            smooth=self.smooth,
            exponent=self.exponent,
            class_weight=class_weight,
            ignore_index=self.ignore_index)
        return loss

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
