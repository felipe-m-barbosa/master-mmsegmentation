import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from ..builder import HEADS
from .decode_head import BaseDecodeHead

@HEADS.register_module()
class OrderPredHead(BaseDecodeHead):

    def __init__(self, input_dim, embed_dim, output_dim, seq_length, pool_scales=(1, 2, 3, 6), **kwargs):
        super(OrderPredHead, self).__init__(**kwargs)
        self.fc1 = nn.Linear(2*input_dim, embed_dim) # input_dim should be 256*64*64 (add in config file)
        self.fc2 = nn.Linear(6*embed_dim, output_dim)
        self.seq_len = seq_length

    # def init_weights(self):
    #     pass

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


        inputs1 = inputs[self.in_index] # isso aqui tbm tá meio zoado ... ver o colab
        inputs2 = inputs[0]
        inputs3 = inputs[1]
        inputs4 = inputs[2]

        print("Inputs2: ", end='\n')
        for i in inputs2:
            print(i.shape)

        print("Inputs3: ", end='\n')
        for i in inputs3:
            print(i.shape)

        print("Inputs4: ", end='\n')
        for i in inputs4:
            print(i.shape)


        print("Shapes before flatten operation: ", end='\n')
        for i in inputs1:
            print(i.shape)

        assert len(inputs) == self.seq_len, (f"Inputs list to OrderPredHead is expected to have {self.seq_len} elements, but got length of {len(inputs)}")
        
        # flatten inputs
        inputs = [torch.flatten(i, start_dim=1) for i in inputs]

        print(len(inputs))

        # shapes were supposed to be equal... 
        print(inputs[0].shape) # 1048576
        print(inputs[1].shape) # 524288

        # concatenate inputs, pair-wise
        concat1 = torch.cat((inputs[0], inputs[1]), dim=1)
        concat2 = torch.cat((inputs[0], inputs[2]), dim=1)
        concat3 = torch.cat((inputs[0], inputs[3]), dim=1)
        concat4 = torch.cat((inputs[1], inputs[2]), dim=1)
        concat5 = torch.cat((inputs[1], inputs[3]), dim=1)
        concat6 = torch.cat((inputs[2], inputs[3]), dim=1)

        print(concat1.shape)


        # pass them through first FC layer
        f1 = self.fc1(concat1)
        f2 = self.fc1(concat2)
        f3 = self.fc1(concat3)
        f4 = self.fc1(concat4)
        f5 = self.fc1(concat5)
        f6 = self.fc1(concat6)

        # concatenate intermediate features
        final_concat = torch.cat((f1,f2,f3,f4,f5,f6), dim=1)

        # generate logits
        output = self.fc2(final_concat)

        print(output.shape)

        # generate probabilities

        # loss expects the predicted unnormalized logits
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy
        # so, comment the following        
        # output = F.softmax(output, dim=0) 

        return output

