import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from mmcv.cnn.utils.weight_init import normal_init, constant_init, kaiming_init

from ..builder import HEADS
from .decode_head import BaseDecodeHead

from mmcv.cnn import ConvModule

@HEADS.register_module()
class ConvOrderPredHead(BaseDecodeHead):

    def __init__(self, output_channels, fc_dim, output_dim, norm_cfg, seq_length, **kwargs):
        """
        Args:
            in_channels = 2048
            embed_channels = 256
            output_channels = 128
            fc_dim = 16*16*128
            output_dim = 12
        """
        
        super(ConvOrderPredHead, self).__init__(**kwargs)
        
        self.embed_channels = self.channels
        self.output_channels = output_channels
        self.fc_dim = fc_dim
        self.output_dim = output_dim

        self.seq_len = seq_length

        self.bottleneck = nn.Conv2d(self.in_channels, self.embed_channels, 1) # bottleneck layer added in order to shrink feature maps, thus reducing computation and memory usage

        self.conv1 = ConvModule(
            2*self.embed_channels,
            self.embed_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        
        self.conv2 = ConvModule(
            6*self.embed_channels,
            self.output_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.fc = nn.Linear(self.fc_dim, self.output_dim)
        
        self.init_weights()


    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                normal_init(m, mean=0.01, std=1, bias=0)
                # kaiming_init(m)

                
    def forward(self, inputs):
        """
        Args:
        inputs (list[np.ndarray]|list[torch.Tensor]), expected to be of length seq_len

        Return:
        prediction
        """

        # QUAL VAI SER A DIMENSÃO DA ENTRADA?
        #       UMA LISTA DE BATCHES [(B,c,h,w), ...], em que cada posição corresponde ao batch de imagens da posição correspondente nas diferentes listas


        # FAZER O CÓDIGO PARA SE FOR BATCH ... (usa-se dim=1, eu acho.... tem q rodar para testar)

        # 'inputs' has the outputs from the backbone. Hence, we must extract the input corresponding to the in_index received as argument in class instantiation


        # selecting only the last outputs from backbone
        # inputs is of shape [[(), (), (), ()], [(), (), (), ()], [(), (), (), ()], [(), (), (), ()]]
        # we must select                    ^                 ^                 ^                 ^
        # that is, each internal list corresponds to the outputs for a given image from the input sequence of 4 frames
        # moreover, each item correspond to a batch of 8 images

        inputs = [i[self.in_index] for i in inputs]


        assert len(inputs) == self.seq_len, (f"Inputs list to OrderPredHead is expected to have {self.seq_len} elements, but got length of {len(inputs)}")
        
        inputs = [self.bottleneck(i) for i in inputs]

        # print("INPUTS SHAPE: ", inputs[0].shape, inputs[1].shape, inputs[2].shape, inputs[3].shape)

        # concatenate inputs, pair-wise
        concat1 = torch.cat((inputs[0], inputs[1]), dim=1)
        concat2 = torch.cat((inputs[0], inputs[2]), dim=1)
        concat3 = torch.cat((inputs[0], inputs[3]), dim=1)
        concat4 = torch.cat((inputs[1], inputs[2]), dim=1)
        concat5 = torch.cat((inputs[1], inputs[3]), dim=1)
        concat6 = torch.cat((inputs[2], inputs[3]), dim=1)

        # print(concat1.shape)


        # pass them through first FC layer
        f1 = self.conv1(concat1)
        f2 = self.conv1(concat2)
        f3 = self.conv1(concat3)
        f4 = self.conv1(concat4)
        f5 = self.conv1(concat5)
        f6 = self.conv1(concat6)

        # concatenate intermediate features
        final_concat = torch.cat((f1,f2,f3,f4,f5,f6), dim=1)

        # generate logits
        output = self.conv2(final_concat) # apply relu and dropout?

        output = torch.flatten(output, start_dim=1)

        output = self.fc(output)

        # print("OrderPredHead output: ", output)

        # print(output.shape)

        # generate probabilities

        # loss expects the predicted unnormalized logits
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy
        # so, comment the following        
        # output = F.softmax(output, dim=1) 

        return output

