# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from mmseg.ops import resize

import time

def f_score(precision, recall, beta=1):
    """calculate the f-score value.

    Args:
        precision (float | torch.Tensor): The precision value.
        recall (float | torch.Tensor): The recall value.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.

    Returns:
        [torch.tensor]: The f-score value.
    """
    score = (1 + beta**2) * (precision * recall) / (
        (beta**2 * precision) + recall)
    return score


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray | str): Ground truth segmentation map
            or label filename.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(label, str):
        label = torch.from_numpy(
            mmcv.imread(label, flag='unchanged', backend='pillow'))
    else:
        if isinstance(label, tuple):
            label = label[0]

        label = torch.from_numpy(label)

    if label_map is not None:
        label_copy = label.clone()
        for old_id, new_id in label_map.items():
            label[label_copy == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)
    for result, gt_seg_map in zip(results, gt_seg_maps):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(
                result, gt_seg_map, num_classes, ignore_index,
                label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label


def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]:
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <IoU> ndarray: Per category IoU, shape (num_classes, ).
    """
    iou_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return iou_result


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False,
              **kwargs):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <Dice> ndarray: Per category dice, shape (num_classes, ).
    """

    dice_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return dice_result


def mean_fscore(results,
                gt_seg_maps,
                num_classes,
                ignore_index,
                nan_to_num=None,
                label_map=dict(),
                reduce_zero_label=False,
                beta=1):
    """Calculate Mean F-Score (mFscore)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.


     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Fscore> ndarray: Per category recall, shape (num_classes, ).
            <Precision> ndarray: Per category precision, shape (num_classes, ).
            <Recall> ndarray: Per category f-score, shape (num_classes, ).
    """
    fscore_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mFscore'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label,
        beta=beta)
    return fscore_result




def warp(x, flo):
    """
    Args:
        x: the input prediction, to be warped using the optical flow
        flo: optical flow field, with shape (B, H, W, C)
            C accounts for the number of channels, 
              which correspond to the pixel displacements in x and y directions
    """

    # print(x.shape)
    # time.sleep(50)
    H,W = x.shape
    B=1

    xx = torch.arange(0,W).view(1,-1).repeat(H,1) # H lines, each with W elements (int)
    yy = torch.arange(0,H).view(-1,1).repeat(1,W) # W columns, each with H elements (int)
     
    xx = xx.view(1,H,W,1).repeat(B,1,1,1) # change shape and repeat over the batch dimension
    yy = yy.view(1,H,W,1).repeat(B,1,1,1) # change shape and repeat over the batch dimension

    grid = torch.cat((xx,yy),3).float() # concatenates the given sequence of seq tensors in the given dimension
    # grid.shape: (B,H,W,2), where 2 corresponds to [x_idx, y_idx]

    x = torch.from_numpy(x)
    if x.is_cuda:
        grid = grid.cuda()
        flo = flo.cuda()
    else:
        grid = grid.to('cpu')
        flo = flo.to('cpu')

    # print(x.shape, end='\n')
    # print(flo.shape, end='\n')
    # print(grid.shape, end='\n')
    # time.sleep(50)

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
    x = x.unsqueeze(0).unsqueeze(0) # create batch and channels dimensions
    # print(f"x device: {x.device}")
    # print(f"vgrid device: {vgrid.device}", end="\n\n\n\n")
    # time.sleep(50)
    x = x.to('cuda')
    vgrid = vgrid.to('cuda')
    output = torch.nn.functional.grid_sample(x, vgrid)

    # print("output shape: ", output.shape)


    # VALIDITY MASK

    # this implementation only accounts for misaligned borders
    # that is, occlusions caused by regions in the image borders
    mask = torch.autograd.Variable(torch.ones(x.size()))
    mask = mask.to('cuda')

    if x.is_cuda:
        mask = mask.cuda()

    mask = torch.nn.functional.grid_sample(mask, vgrid)
    
    mask[mask<0.9999]=0
    mask[mask>0]=1
    
    output = output*mask
    output = output.type(torch.float32)

    output = output.squeeze(0).squeeze(0) # recover original shape of prediction

    output = output.to('cpu')
    mask = mask.to('cpu')

    output = output.numpy()
    mask = mask.numpy()

    # print(f"warp-output.shape: {output.shape}")

    return output, mask

from mmseg.core.evaluation import get_palette
import matplotlib.pyplot as plt

def verify_warp(pred, wpred, target):
    PALETTE = get_palette('cityscapes')
    print(PALETTE)
    
    if len(pred.shape) > 2:
        pred = pred.squeeze(2)
    
    if len(wpred.shape) > 2:
        wpred = wpred.squeeze(2)
    
    if len(target.shape) > 2:
        target = target.squeeze(2)
    
    # plt.imshow(pred, cmap='gray')
    # plt.show()

    # print(np.unique(pred))

    from PIL import Image

    output_pred = Image.fromarray(pred.astype(np.uint8)).convert('P')
    output_wpred = Image.fromarray(wpred.astype(np.uint8)).convert('P')
    output_target = Image.fromarray(target.astype(np.uint8)).convert('P')
    import cityscapesscripts.helpers.labels as CSLabels
    palette = np.zeros((len(CSLabels.trainId2label), 3), dtype=np.uint8)
    for label_id, label in CSLabels.trainId2label.items():
        if label_id != 255:
            palette[label_id] = label.color

    output_pred.putpalette(palette)
    output_wpred.putpalette(palette)
    output_target.putpalette(palette)

    cv_pred = np.array(output_pred)
    cv_wpred = np.array(output_wpred)
    cv_target = np.array(output_target)

    # concat = cv2.hconcat([cv_pred, cv_target])
    # concat = 0.5*cv_pred + 0.5*cv_target

    concat = cv2.vconcat([cv_pred, cv_wpred, cv_target])

    pil_concat = Image.fromarray(concat.astype(np.uint8))

    plt.imshow(pil_concat)
    plt.axis('off')
    plt.show()

    # for idx_cls in range(19):
    #     print(f"idx {idx_cls} | count: {np.sum(pred == idx_cls)} | color: {PALETTE[idx_cls]}")
    #     color_pred[pred == idx_cls] = PALETTE[idx_cls]
    
    # for idx_cls in range(19):
    #     color_target[target == idx_cls] = PALETTE[idx_cls]
    
    # color_pred = cv2.cvtColor(color_pred, cv2.COLOR_RGB2BGR)
    # color_target = cv2.cvtColor(color_target, cv2.COLOR_RGB2BGR)

    # concat = cv2.hconcat([color_pred, color_target])
    # plt.imshow(concat)
    # plt.show()




def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1,
                 tc_eval=False,
                 **kwargs):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    ret_metrics = {}

    if gt_seg_maps is not None:
        total_area_intersect, total_area_union, total_area_pred_label, \
            total_area_label = total_intersect_and_union(
                results, gt_seg_maps, num_classes, ignore_index, label_map,
                reduce_zero_label)

        # first, compute only non-TC metrics
        metrics = [m for m in metrics if 'TC' not in m]
        ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                            total_area_pred_label,
                                            total_area_label, metrics, nan_to_num,
                                            beta)
    
    # Compute Temporal Consistency metrics
    if ('TC(mDice)' in metrics) or ('TC(mIoU)' in metrics):
        optflows = kwargs.get('optflows', None)
        names = kwargs.get('names', None)
        assert (tc_eval and (optflows is not None)) or (not(tc_eval) and (optflows is None))
        
        tc_metrics = [m for m in metrics if 'TC' in m]
        
        # generating (warped) predictions and targets for temporal consistency computation
        # optical flow-based seg-pred warping
        preds = []
        # print(len(results))

        # 0 -> 1
        for r1, optf in zip(results[:-1], optflows[:-1]):

            # print(r1.shape)

            optf = optf.squeeze(0)
            optf = optf.detach().cpu().numpy()
            r1 = torch.as_tensor(r1).unsqueeze(2).type(torch.float32)
            r1 = r1.detach().cpu().numpy()
            pred = mmcv.flow_warp(r1, optf) # prediction is seg(t0)->seg'(t1), while the target is seg(t1)
            preds.append(pred)

            # pred, _ = warp(r1, optf) # prediction is seg(t0)->seg'(t1), while the target is seg(t1)
            # preds.append(pred)

        # print(f"preds.shape: {preds[0].shape}")
        # time.sleep(50)

        targets = [torch.as_tensor(r).unsqueeze(2).type(torch.float32) for r in results[1:]]
        targets = [t.detach().cpu().numpy() for t in targets]

        # orig_preds = results[:-1]


        # for p, wp, t in zip(orig_preds, preds, targets):
        #     verify_warp(p, wp, t)
        

        # 1 -> 0
        # for r2, optf in zip(results[1:], optflows[:-1]):

        #     # print(r1.shape)

        #     optf = optf.squeeze(0)
        #     optf = optf.detach().cpu().numpy()
        #     r2 = torch.as_tensor(r2).unsqueeze(2).type(torch.float32)
        #     r2 = r2.detach().cpu().numpy()
        #     pred = mmcv.flow_warp(r2, optf) # prediction is seg(t0)->seg'(t1), while the target is seg(t1)
        #     preds.append(pred)

        #     # pred, _ = warp(r1, optf) # prediction is seg(t0)->seg'(t1), while the target is seg(t1)
        #     # preds.append(pred)

        # # print(f"preds.shape: {preds[0].shape}")
        # # time.sleep(50)

        # targets = [torch.as_tensor(r).unsqueeze(2).type(torch.float32) for r in results[:-1]]
        # targets = [t.detach().cpu().numpy() for t in targets]

        # orig_preds = results[1:]


        # for p, wp, t in zip(orig_preds, preds, targets):
        #     verify_warp(p, wp, t)




        # compute losses similar to what was previously done, but for TC metrics
        total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(
            preds, targets, num_classes, ignore_index, label_map,
            reduce_zero_label)
        ret_metrics_2 = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, tc_metrics, nan_to_num,
                                        beta)
        
        # as ret_metrics2 only contains TC metrics, we need to add those to the main ret_metrics dict
        for met, val in ret_metrics_2.items():
            ret_metrics[met] = val
        

    return ret_metrics


def pre_eval_to_metrics(pre_eval_results,
                        metrics=['mIoU'],
                        nan_to_num=None,
                        beta=1):
    """Convert pre-eval results to metrics.

    Args:
        pre_eval_results (list[tuple[torch.Tensor]]): per image eval results
            for computing evaluation metric
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    # print("\n\n\n", pre_eval_results, end="\n\n\n")

    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 4

    total_area_intersect = sum(pre_eval_results[0])
    total_area_union = sum(pre_eval_results[1])
    total_area_pred_label = sum(pre_eval_results[2])
    total_area_label = sum(pre_eval_results[3])

    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)

    return ret_metrics


def total_area_to_metrics(total_area_intersect,
                          total_area_union,
                          total_area_pred_label,
                          total_area_label,
                          metrics=['mIoU'],
                          nan_to_num=None,
                          beta=1):
    """Calculate evaluation metrics
    Args:
        total_area_intersect (ndarray): The intersection of prediction and
            ground truth histogram on all classes.
        total_area_union (ndarray): The union of prediction and ground truth
            histogram on all classes.
        total_area_pred_label (ndarray): The prediction histogram on all
            classes.
        total_area_label (ndarray): The ground truth histogram on all classes.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore', 'TC(mDice)', 'TC(mIoU)']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({'aAcc': all_acc})
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics['IoU'] = iou
            ret_metrics['Acc'] = acc
        elif metric == 'mDice': # TODO: modify the function so that it handles the temporal consistency calculation
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            acc = total_area_intersect / total_area_label
            ret_metrics['Dice'] = dice
            ret_metrics['Acc'] = acc
        elif metric == 'mFscore':
            precision = total_area_intersect / total_area_pred_label
            recall = total_area_intersect / total_area_label
            f_value = torch.tensor(
                [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
            ret_metrics['Fscore'] = f_value
            ret_metrics['Precision'] = precision
            ret_metrics['Recall'] = recall
        elif metric == 'TC(mIoU)':
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics['TC(IoU)'] = iou
            ret_metrics['TC(IoUAcc)'] = acc
        elif metric == 'TC(Dice)':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            acc = total_area_intersect / total_area_label
            ret_metrics['TC(Dice)'] = dice
            ret_metrics['TC(DiceAcc)'] = acc
        

    ret_metrics = {
        metric: value.numpy()
        for metric, value in ret_metrics.items()
    }
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics
