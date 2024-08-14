
'''
 (C) Copyright IBM Corp. 2024.
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
           http://www.apache.org/licenses/LICENSE-2.0
     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
     limitations under the License.
 Project name: Model Urban Growth MUG
'''
import torch
import torch.nn as nn
from operator import mul
from functools import reduce
from typing import Iterable, Tuple


# Original ConvLSTM cell as proposed by Shi et al. (2015)
class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, out_channels, 
    kernel_size, padding, activation, frame_size):

        super(ConvLSTMCell, self).__init__()  

        if activation == "tanh":
            self.activation = torch.tanh 
        elif activation == "relu":
            self.activation = torch.relu
        
        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels, 
            out_channels=4 * out_channels, 
            kernel_size=kernel_size, 
            padding=padding)           

        # Initialize weights for Hadamard Products
        self.W_ci = nn.Parameter(torch.randn(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.randn(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.randn(out_channels, *frame_size))

    def forward(self, X, H_prev, C_prev):

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev)

        # Current Cell output
        C = forget_gate*C_prev + input_gate * self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C )

        # Current Hidden State
        H = output_gate * self.activation(C)

        return H, C


class ConvLSTM(nn.Module):

    def __init__(self, in_channels, out_channels, 
    kernel_size, padding, activation, frame_size):

        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels

        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels, 
        kernel_size, padding, activation, frame_size)

    def forward(self, X):

        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)

        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size,
                             self.out_channels,
                             seq_len,
                             height,
                             width,
                             device=self.convLSTMcell.W_cf.device
                             )
        
        # Initialize Hidden State
        H = torch.zeros(batch_size,
                        self.out_channels,
                        height,
                        width,
                        device=self.convLSTMcell.W_cf.device
                        )

        # Initialize Cell Input
        C = torch.zeros(batch_size,
                        self.out_channels,
                        height,
                        width,
                        device=self.convLSTMcell.W_cf.device
                        )

        # Unroll over time steps
        for time_step in range(seq_len):

            H, C = self.convLSTMcell(X[:,:,time_step], H, C)

            output[:,:,time_step] = H

        return output


class SequenceToSequence(nn.Module):

    def __init__(self, num_channels, num_kernels, kernel_size, padding, 
    activation, frame_size, num_layers):

        super().__init__()

        self.sequential = nn.Sequential()

        # Add First layer (Different in_channels than the rest)
        self.sequential.add_module(
            "convlstm1", ConvLSTM(
                in_channels=num_channels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size)
        )

        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=num_kernels)
        ) 

        # Add rest of the layers
        for l in range(2, num_layers+1):

            self.sequential.add_module(
                f"convlstm{l}", ConvLSTM(
                    in_channels=num_kernels, out_channels=num_kernels,
                    kernel_size=kernel_size, padding=padding, 
                    activation=activation, frame_size=frame_size)
                )
                
            self.sequential.add_module(
                f"batchnorm{l}", nn.BatchNorm3d(num_features=num_kernels)
                )

        # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(
            in_channels=num_kernels, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding)

    def forward(self, X):

        # Forward propagation through all the layers
        output = self.sequential(X)

        # Return only the last output frame
        output = self.conv(output[:,:,-1])
        
        return nn.Sigmoid()(output)


class Conv3DLSTM(nn.Module):
    def __init__(
            self,
            frame_size: Tuple[int, int],
            input_channels: int,
            activation: str = "relu",
            batch_normalization: bool = False,
            layers: int = 3,
            strides: Tuple[int, ...] = (2,2,3,),
            encoder_channels: Tuple[int, ...] = (32, 64, 128,),
            forecaster_channels: Tuple[int, ...] = (128, 32, 8,),
            ) -> None:
        """
        Adapted from Dan Liu, Li Diao, Liujia Xu et al (2020) 
        'Precipitation Forecast Based on Multi-Channel ConvLSTM and 3D-CNN,'
        2020 Inter Conf on Unmanned Aircraft Systems (ICUAS), 367-371.
        """
        super().__init__()

        self.encoder = nn.Sequential()
        self.forecaster = nn.Sequential()

        # argument checks
        for s in frame_size:
            if s % reduce(mul, strides, 1) != 0:
                raise ValueError("frame_size must be a multiple of the productory of strides.")
        
        if len(strides) != layers:
            raise ValueError("layers must be equal to length of strides.")
        
        if len(encoder_channels) != layers:
            raise ValueError("layers must be equal to length of encoder_channels.")
        
        if len(forecaster_channels) != layers:
            raise ValueError("layers must be equal to length of forecaster_channels.")
        
        if forecaster_channels[0] != encoder_channels[-1]:
            raise ValueError("Incompatible lists of encoder and forecaster channels.")

        # build encoder
        channels = (input_channels, *encoder_channels)
        for j in range(layers):
            self.encoder.add_module(
                f"e-conv3d-{j+1}", nn.Conv3d(
                    in_channels=channels[j],
                    out_channels=channels[j+1],
                    kernel_size=1,
                    stride=(1, strides[j], strides[j]),
                    padding="valid",
                )
            )

            self.encoder.add_module(
                f"e-convlstm-{j+1}", ConvLSTM(
                    in_channels=channels[j+1],
                    out_channels=channels[j+1],
                    kernel_size=1,
                    padding="valid",
                    activation=activation,
                    frame_size=tuple(int(s/reduce(mul, strides[:j+1])) for s in frame_size)
                )
            )

            if batch_normalization:
                self.encoder.add_module(
                    f"e-batchnorm-{j+1}", nn.BatchNorm3d(
                        num_features=channels[j+1]
                    )
                )

        # build forecaster
        channels = (encoder_channels[-1], *forecaster_channels)
        for k in range(layers):
            self.forecaster.add_module(
                f"f-conv3d-{k+1}", nn.ConvTranspose3d(
                    in_channels=channels[k],
                    out_channels=channels[k+1],
                    kernel_size=(1, strides[-k-1], strides[-k-1]),
                    stride=(1, strides[-k-1], strides[-k-1]),
                    padding=0
                )
            )

            self.forecaster.add_module(
                f"f-convlstm-{k+1}", ConvLSTM(
                    in_channels=channels[k+1],
                    out_channels=channels[k+1],
                    kernel_size=1,
                    padding="valid",
                    activation=activation,
                    frame_size=tuple(int(s/reduce(mul, strides)) * reduce(mul, strides[-k-1:]) for s in frame_size)
                )
            )

            if batch_normalization:
                self.encoder.add_module(
                    f"f-batchnorm-{k+1}", nn.BatchNorm3d(
                        num_features=channels[k+1]
                    )
                )

        self.forecaster.add_module(
            f"f-conv3d-{layers+1}", nn.ConvTranspose3d(
                in_channels=forecaster_channels[-1],
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )

    def forward(self, X):
        output = self.encoder(X)
        output = self.forecaster(output)
        return nn.Sigmoid()(output)