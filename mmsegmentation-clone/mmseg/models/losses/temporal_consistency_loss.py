# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss

import mmcv

def warp(x, flo):
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

    ## scale grid to [-1,1]
    vgrid[:,:,:,0] = 2.0*vgrid[:,:,:,0].clone()/max(W-1,1)-1.0 # x
    vgrid[:,:,:,1] = 2.0*vgrid[:,:,:,1].clone()/max(H-1,1)-1.0 # y
     
    #  x = x.permute(0,3,1,2)
    x = x.type(torch.float32)

    #  print(f"Img type: {x.dtype}")
    #  print(f"Grid type: {vgrid.dtype}")

    # WARPING
    output = torch.nn.functional.grid_sample(x, vgrid)
    
    # VALIDITY MASK

    # this implementation only accounts for misaligned borders
    # that is, occlusions caused by regions in the image borders
    mask = torch.autograd.Variable(torch.ones(x.size()))

    if x.is_cuda:
        mask = mask.cuda()

    mask = torch.nn.functional.grid_sample(mask, vgrid)
    
    mask[mask<0.9999]=0
    mask[mask>0]=1
    
    output = output*mask
    output = output.type(torch.float32)

    return output, mask


def temporal_miou(preds, 
                  targets, 
                  gt_labels,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=255,
                  avg_non_ignore=False):

    """
        Args:
            preds: 
            targets: 
    """


    non_ignore_mask = (gt_labels != ignore_index).long()
    non_ignore_mask = non_ignore_mask.unsqueeze(1)

    # print("aqui")
    # print(gt_labels.shape)
    # print('preds_shape: ', preds.shape)
    # print('targets shape: ', targets.shape)
    # print(non_ignore_mask.shape)

    targets = targets * non_ignore_mask

    # soft mIoU
    num = torch.sum(torch.abs(preds * targets), dim=(2,3))
    den = torch.sum(torch.abs(preds + targets - (preds * targets)), dim=(2,3))

    num_classes = targets.shape[1]
    # print(num_classes)
    miou = torch.sum((num/den), dim=1)/num_classes

    # miou = torch.sum((num/den))/num_classes
    # miou = miou / batch_size

    # miou = torch.sum((num/den))
    loss = 1-miou

    if weight is not None:
        weight = weight.float()

    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


# also called soft-miou in [Optimizing Intersection-Over-Union in Deep Neural Networks for Image Segmentation]
@LOSSES.register_module()
class TCLoss(nn.Module):
    """Temporal Consistency Loss (mIoU-based)

    """
    def __init__(self,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_tc',
                 avg_non_ignore=False):
        super(TCLoss, self).__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.avg_non_ignore = avg_non_ignore
        self._loss_name = loss_name

        self.cls_criterion = temporal_miou
    
    def forward(self, 
                preds, 
                targets, 
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = preds.new_tensor(self.class_weight)
        else:
            class_weight = None

        # apply softmax to preds and targets logits
        preds = F.softmax(preds, dim=1)
        targets = F.softmax(targets, dim=1)

        # prediction WARPING from frame at time t to frame at time t+1 (t -> t+1)
        opt_flow = kwargs['opt_flow']
        # print(preds.shape)
        # print(opt_flow.shape)
        # preds_to_targets, _ = warp(preds, opt_flow) # TROCAR AQUI PARA O MMCV.FLOW_WARP

        print("")

        preds_list = []
        for p, of in zip(preds, opt_flow):
            of = of.squeeze(0)
            of = of.detach().cpu().numpy()
            p = p.detach().cpu().numpy()

            print(p.shape)
            
            # warp predictions
            pw = mmcv.flow_warp(p, of)
            
            # back to torch tensor
            pw = torch.as_tensor(pw).permute(2,0,1) 

            preds_list.append(pw)
        
        preds_to_targets = torch.stack(tuple(preds_list), 0).to('cuda')

        # p = p.squeeze(0).permute(1,2,0) # channels last
        #         of = of.squeeze(0)
        #         p = p.detach().cpu().numpy()
        #         of = of.detach().cpu().numpy()
        #         # print('P SHAPE: ', p.shape)
        #         # print('OF SHAPE: ', of.shape)
        #         # print(type(p))
        #         # print(type(of))
        #         pw = mmcv.flow_warp(p, of)
        #         pw = torch.as_tensor(pw).permute(2,0,1) # back to torch tensor
        #         # pw = pw.unsqueeze(0)
        #         preds.append(pw)


        #     pred = torch.stack(tuple(preds), 0).to('cuda')


        # COMPUTE (WEIGHTED) LOSS
        loss_cls = self.loss_weight * self.cls_criterion(preds_to_targets, 
            targets,
            kwargs['gt_labels'],
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_index,
            avg_non_ignore=self.avg_non_ignore)

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