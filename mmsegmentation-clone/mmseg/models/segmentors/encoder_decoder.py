# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor

import time


class MemoryQueue():

    def __init__(self, args):
        self.queue_size = args.stm_queue_size
        self.queue_keys = []
        self.queue_vals = []
        self.queue_idxs = []

    def reset(self):
        self.queue_keys = []
        self.queue_vals = []
        self.queue_idxs = []

    def current_size(self):
        return len(self.queue_keys)

    def update(self, key, val, idx):
        self.queue_keys.append(key)
        self.queue_vals.append(val)
        self.queue_idxs.append(idx)

        if len(self.queue_keys) > self.queue_size:
            self.queue_keys.pop(0)
            self.queue_vals.pop(0)
            self.queue_idxs.pop(0)

    def get_indices(self):
        return self.queue_idxs

    def get_keys(self):
        return torch.stack(self.queue_keys, dim=2)

    def get_vals(self):
        return torch.stack(self.queue_vals, dim=2)


class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim, val_pass):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.val_pass = val_pass
        if not self.val_pass:
            self.Value = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, x):
        val = x if self.val_pass else self.Value(x)
        return self.Key(x), val


class Memory(nn.Module):
    def __init__(self, args):
        super(Memory, self).__init__()
        self.learnable_constant = args.learnable_constant
        if self.learnable_constant:
            self.const = nn.Parameter(torch.zeros(1))

    def forward(self, m_in, m_out, q_in):  # m_in: o,c,t,h,w
        # o = batch of objects = num objects.
        # d is the dimension, number of channels, t is time
        B, D_e, T, H, W = m_in.size()
        _, D_o, _, _, _ = m_out.size()

        mi = m_in.view(B, D_e, T*H*W)
        mi = torch.transpose(mi, 1, 2)  # b, THW, emb

        qi = q_in.view(B, D_e, H*W)  # b, emb, HW

        p = torch.bmm(mi, qi) # b, THW, HW
        p = p / math.sqrt(D_e)
        if self.learnable_constant:
            p = torch.cat([p, self.const.view(1, 1, 1).expand(B, -1, H*W)], dim=1)
        p = F.softmax(p, dim=1) # b, THW, HW
        if self.learnable_constant:
            p = p[:, :-1, :]
        # For visualization later
        p_volume = None
        # p_volume = p.view(B, T, H, W, H, W)

        mo = m_out.view(B, D_o, T*H*W)

        mem = torch.bmm(mo, p) # Weighted-sum B, D_o, HW
        mem = mem.view(B, D_o, H, W)

        return mem, p, p_volume



class MemoryLocal(nn.Module):
    def __init__(self, args):
        super(MemoryLocal, self).__init__()
        self.learnable_constant = args.learnable_constant
        self.corr_size = args.corr_size
        if self.learnable_constant:
            self.const = nn.Parameter(torch.zeros(1))

    def functionCorrelation():
        pass

    def functionCorrelationTranspose():
        pass

    def forward(self, m_in, m_out, q_in):  # m_in: o,c,t,h,w
        # TODO note to verify
        # o = batch of objects = num objects. NOT batch
        # d is the dimension, number of channels, t is time
        B, D_e, T, H, W = m_in.size()
        _, D_o, _, _, _ = m_out.size()

        patch_size = self.corr_size

        p = torch.stack([self.functionCorrelation(q_in.contiguous(), m_in[:,:,t,:,:].contiguous(), patch_size) for t in range(T)], dim=2) # B, N^2, T, H, W
        p = p.reshape(B, -1, H, W) # B, T*N^2, H, W
        if self.learnable_constant:
            p = torch.cat([p, self.const.view(1, 1, 1, 1).expand(B, -1, H, W)], dim=1)
        p = F.softmax(p, dim=1)
        if self.learnable_constant:
            p = p[:, :-1, :, :]
        p = p.reshape(B, -1, T, H, W) # B, N^2, T, H, W

        p_volume = None
        # p_volume = torch.stack([self.remap_cost_volume(p[:,:,t,:,:]) for t in range(T)], dim=1)

        mem = sum([self.functionCorrelationTranspose(p[:,:,t,:,:].contiguous(), m_out[:,:,t,:,:].contiguous(), patch_size) for t in range(T)])

        return mem, p, p_volume

    
@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 #in_indices=None,
                 init_cfg=None):
        super(EncoderDecoder, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

        # self.spatial_memory = MemoryQueue(mem_args)
        # self.context_memory = MemoryQueue(mem_args)
        # self.memory_module = Memory(mem_args)
        # ALSO RECEIVE IN_INDICES INFO

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img):
        """Extract features from images."""

        # print(len(img))
        # print("\n --- \n")

        # for i in img:
        #     print(i.shape)

        # for idx, i in enumerate(img):
        #     print(f"encoder-decoder - extract_feat - img{idx}.shape: ", i.shape)

        if isinstance(img, list):
            x = []
            for i in img:
                if self.with_neck:
                    x.append(self.neck(self.backbone(i)))
                else:
                    x.append(self.backbone(i))
        else:
            x = self.backbone(img)
            if self.with_neck:
                x = self.neck(x)
        
        return x
    
    def extract_feat_seq(self, img, s1_img, s2_img):
        """Extract features from images.""" 
        x = self.backbone(img)

        # for the image sequence
        s1 = self.backbone(s1_img)
        s2 = self.backbone(s2_img)

        if self.with_neck:
            x = self.neck(x)
            # for the image sequence
            s1 = self.neck(s1)
            s2 = self.neck(s2)
        return x, s1, s2

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)

        if not 'video_name' in img_metas[0]: # not order prediction task
            out = resize(
                input=out,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg, **kwargs):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()

        if 's1' in kwargs:
            loss_decode = self.decode_head.forward_train(x, img_metas,
                                                        gt_semantic_seg,
                                                        self.train_cfg,
                                                        s1=kwargs['s1'], 
                                                        s2=kwargs['s2'], 
                                                        opt_flow=kwargs['opt_flow'])
        elif 'gt_depth' in kwargs:
            loss_decode = self.decode_head.forward_train(x, img_metas,
                                                        gt_semantic_seg,
                                                        self.train_cfg,
                                                        gt_depth=kwargs['gt_depth'])
        else:
            loss_decode = self.decode_head.forward_train(x, img_metas,
                                                        gt_semantic_seg,
                                                        self.train_cfg) # x is a list in order prediction task

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg, **kwargs):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                if 's1' in kwargs:
                    loss_aux = aux_head.forward_train(x, img_metas,
                                                    gt_semantic_seg,
                                                    self.train_cfg,
                                                    s1=kwargs['s1'], 
                                                    s2=kwargs['s2'], 
                                                    opt_flow=kwargs['opt_flow'])
                else:
                    loss_aux = aux_head.forward_train(x, img_metas,
                                                    gt_semantic_seg,
                                                    self.train_cfg)

                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            if 's1' in kwargs:
                loss_aux = self.auxiliary_head.forward_train(
                    x, img_metas, gt_semantic_seg, self.train_cfg, s1=kwargs['s1'], 
                                                                s2=kwargs['s2'], 
                                                                opt_flow=kwargs['opt_flow'])
            else:
                loss_aux = self.auxiliary_head.forward_train(
                    x, img_metas, gt_semantic_seg, self.train_cfg)

            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit


    # def reset_memory(self):
    #     self.memory_queue.reset()


    # this function can be useful for ablation studies regarding FPS 
    # def memory_range(self, seq_len):

    #     ret_range = range(seq_len)

    #     if self.memory_strategy == "all":
    #         pass
    #     if self.memory_strategy == "skip_01":
    #         ret_range = range(0, seq_len + 1, 2)
    #     if self.memory_strategy == "skip_02":
    #         ret_range = range(0, seq_len + 1, 3)
    #     if self.memory_strategy == "skip_03":
    #         ret_range = range(0, seq_len + 1, 4)
    #     if self.memory_strategy == "skip_04":
    #         ret_range = range(0, seq_len + 1, 5)
    #     if self.memory_strategy == "skip_05":
    #         ret_range = range(0, seq_len + 1, 6)
    #     if self.memory_strategy == "random":
    #         ret_range = random.sample(range(seq_len-1), self.stm_queue_size-1)
    #         ret_range.append(seq_len - 1)
    #         ret_range.sort()

    #     # assert(len(ret_range) == self.stm_queue_size)

    #     assert(seq_len - 1 in ret_range)
    #     return ret_range

    # HERE GOES THE LOGIC FOR MEMORY WRITE AND READ
    def forward_train(self, img, img_metas, gt_semantic_seg, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        losses = dict()

        if 's1_img' in kwargs:
            x, s1, s2 = self.extract_feat_seq(img, s1_img=kwargs['s1_img'], s2_img=kwargs['s2_img'])

            loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg,
                                                      s1=s1, 
                                                      s2=s2, 
                                                      opt_flow=kwargs['opt_flow'])
        else:
            x = self.extract_feat(img) # x is a list in order prediction task
            # *** THIS RETURNS A TUPLE... WE SHOULD SELECT THE DESIRED INDEX BEFORE PERFORMING THE FOLLOWING PROCESSING
            # SOMETHING LIKE:
            # x_l = x[self.in_indices[0]] (lower-level feature)
            # x_h = x[self.in_indices[1]] (higher-level feature)

            # IMPORTANT!!!
            # ---------------------------------------------------------------------------
            #   THE FOLLOWING LOGIC SHOULD BE EXTENDED TO OUR TWO DESIRED MEMORY MODULES:
            #       - CONTEXTUAL MEMORY
            #       - SPATIAL MEMORY
            # ---------------------------------------------------------------------------
            #   ALSO, REMEMBER THAT THE LOGIC FOR FEATURE SELECTION (SELECTING ONLY THE IN_INDEX)
            #   IS IMPLEMENTED IN THE DECODE_HEADS. HENCE, WE SHOULD IMPLEMENT SOMETHING SIMILAR
            #   HERE IN THE ENCODER_DECODER CLASS (SEE COMMENT MARKED WITH ***)
            # ---------------------------------------------------------------------------

            # TRAINING MEMORY LOGIC
            #   - FILL MEMORY WITH PREVIOUS FRAMES' FEATURES (SO, WE NEED A FOR LOOP)
            #
            # SPATIAL MEMORY
            # for t in range(len(x_l)-1): # the last frame is not stored in memory
            #     encoder_output = self.extract_feat(x_l[t])
            #     kM, vM = self.kv_M_r4.forward(encoder_output)

            #     idx = t
            #     self.spatial_memory.update(kM, vM, idx)
            #
            # CONTEXT_MEMORY
            # for t in range(len(x_h)-1): # the last frame is not stored in memory
            #     encoder_output = self.extract_feat(x_h[t])
            #     kM, vM = self.kv_M_r4.forward(encoder_output)

            #     idx = t
            #     self.context_memory.update(kM, vM, idx)

            
            # SAVE LAST OUTPUT (referring to frame[t-1] and only for high-level features), SINCE IT WILL BE USED IN THE TEMPORAL CONSISTENCY LOSS
            # prev_feat = encoder_output

            # NOW THAT MEMORY IS FULL, WE EXTRACT FEATURES FOR THE CURRENT FRAME
            # curr_feat = self.extract_feat(imgs[-1])
            # kQ, vQ = self.kv_Q_r4.forward(curr_feat)

            # READ SPATIAL MEMORY (returning spatial_mem)
            # spatial_mem, p, p_vol = self.memory_module.forward(
            #                 self.spatial_memory.get_keys(), self.spatial_memory.get_vals(), kQ)
            
            # READ CONTEXT MEMORY (returning context_mem)
            # context_mem, context_p, context_p_vol = self.memory_module.forward(
            #                 self.context_memory.get_keys(), self.context_memory.get_vals(), kQ)


            # FUSE QUERY VALUE AND RESULT FROM MEMORY READ (CONSIDERING BOTH SPATIAL AND CONTEXT MEMORIES)
            # fused_mem = self.memory_fusion(spatial_mem, context_mem, vQ)


            # IMPLEMENT MEMORY READ AND FUSION ALSO FOR x[t-1] (x_prev)
            #   REMEMBER THAT IT SHOULD ONLY HAVE ACCESS TO FEATURES FROM ITS PREDECESSORS
            # kQ_prev, vQ_prev = self.kv_Q_r4.forward(prev_feat)
            
            # MEMORY READ (THE LOGIC FOR READING ONLY FEATURES FROM PREVIOUS FRAMES SHOULD BE IMPLEMENTED IN THE memory_module - 'is_prev=True')
            # IT CAN ALSO BE IMPLEMENTED IN THE get_keys() and get_vals() methods from MemoryQueue (maybe by passing an argument corresponding to the number of features to retrieve)
            # spatial_mem_prev, p_prev, p_vol_prev = self.memory_module.forward(
            #                 self.spatial_memory.get_keys(), self.spatial_memory.get_vals(), kQ_prev, is_prev=True)
            
            # context_mem_prev, context_p_prev, context_p_vol_prev = self.memory_module.forward(
            #                 self.context_memory.get_keys(), self.context_memory.get_vals(), kQ_prev, is_prev=True)
            
            # FUSE QUERY VALUE AND RESULT FROM MEMORY READ
            # fused_mem_prev = self.memory_fusion(spatial_mem_prev, context_mem_prev, vQ)


            # CONTINUE TO DECODER
            # WE SHOULD SEND BOTH x[t-1] (fused_mem_prev) and x[t] (fused_mem) to the decoder
            # loss_decode = self._decode_head_forward_train(fused_mem, fused_mem_prev, img_metas, gt_semantic_seg)
            
            if 'gt_depth' in kwargs:
                loss_decode = self._decode_head_forward_train(x, img_metas,
                                                        gt_semantic_seg, gt_depth=kwargs['gt_depth'])
            else:
                loss_decode = self._decode_head_forward_train(x, img_metas,
                                                        gt_semantic_seg)

        losses.update(loss_decode)

        if self.with_auxiliary_head:
            if 's1_img' in kwargs:
                loss_aux = self._auxiliary_head_forward_train(
                    x, img_metas, gt_semantic_seg, s1=s1, s2=s2, opt_flow=kwargs['opt_flow'])
            else:
                if 'gt_depth' in kwargs:
                    loss_aux = self._auxiliary_head_forward_train(
                        x, img_metas, gt_semantic_seg, gt_depth=kwargs['gt_depth'])
                else:    
                    loss_aux = self._auxiliary_head_forward_train(
                        x, img_metas, gt_semantic_seg)

            losses.update(loss_aux)

        # ADD AUXILIARY HEAD FOR TEMPORAL CONSISTENCY
        # which is based on the outputs of the previous auxiliary head
        # (s1_logits, s2_logits)
        # ACTUALLY, NOT NEEDED ... the implementation should be done inside _auxiliary_head_forward_train


        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            # remove padding area
            resize_shape = img_meta[0]['img_shape'][:2]
            preds = preds[:, :, :resize_shape[0], :resize_shape[1]]
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)

        if not 'video_name' in img_meta[0]: # not order prediction task
            if rescale:
                # support dynamic shape for onnx
                if torch.onnx.is_in_onnx_export():
                    size = img.shape[2:]
                else:
                    # remove padding area
                    resize_shape = img_meta[0]['img_shape'][:2]
                    seg_logit = seg_logit[:, :, :resize_shape[0], :resize_shape[1]]
                    size = img_meta[0]['ori_shape'][:2]

                # print("\n\n\n")
                # print(f"Seg_logit shape: {seg_logit.shape}")
                # print(f"Size to rescale to: {size}")
                # print("\n\n\n")

                # time.sleep(50)

                seg_logit = resize(
                    seg_logit,
                    size=size,
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        if not 'video_name' in img_meta[0]: # just for non-order prediction task
            assert self.test_cfg.mode in ['slide', 'whole']
            ori_shape = img_meta[0]['ori_shape']
            assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        
        
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        if self.out_channels == 1:
            output = F.sigmoid(seg_logit)
        else:
            output = F.softmax(seg_logit, dim=1)
        
        if not 'video_name' in img_meta[0]: # just for non-order prediction task
            flip = img_meta[0]['flip']
            if flip:
                flip_direction = img_meta[0]['flip_direction']
                assert flip_direction in ['horizontal', 'vertical']
                if flip_direction == 'horizontal':
                    output = output.flip(dims=(3, ))
                elif flip_direction == 'vertical':
                    output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True, **kwargs):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale) # the return is actually the probabilities vector
        if self.out_channels == 1:
            seg_pred = (seg_logit >
                        self.decode_head.threshold).to(seg_logit).squeeze(1)
        else:
            seg_pred = seg_logit.argmax(dim=1) # the integer class
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)

        return seg_pred # list with the integer class values

    def simple_test_logits(self, img, img_metas, rescale=True):
        """Test without augmentations.

        Return numpy seg_map logits.
        """
        seg_logit = self.inference(img[0], img_metas[0], rescale)
        seg_logit = seg_logit.cpu().numpy()
        return seg_logit

    def aug_test(self, imgs, img_metas, rescale=True, **kwargs):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        if self.out_channels == 1:
            seg_pred = (seg_logit >
                        self.decode_head.threshold).to(seg_logit).squeeze(1)
        else:
            seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test_logits(self, img, img_metas, rescale=True):
        """Test with augmentations.

        Return seg_map logits. Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale

        imgs = img
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit

        seg_logit /= len(imgs)
        seg_logit = seg_logit.cpu().numpy()
        return seg_logit
