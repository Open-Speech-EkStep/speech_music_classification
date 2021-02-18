import math

import torch
import torch.nn as nn

class ConvolutionBlock(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int):
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class TransformerBlock(nn.Module):
    def __init__(self):
        super(TransformerBlock, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

    def forward(self, x, src_key_mask):
        out = self.transformer(src, src_key_padding_mask=src_key_mask)
        return out

class Conformer(nn.Module):
    def __init__(self, input_dim=257):
        """ Combination of Conv Block + Transformer Encoder for speech/songs classification."""

        super(Conformer, self).__init__()
        self.conv_blocks = []
        self.in_filters = [input_dim, 64, 128, 256]
        self.out_filters = [64, 128, 256, 256]
        for num_in_filter, num_out_filter in zip(self.in_filters, self.out_filters):
            self.conv_blocks.append(ConvolutionBlock(num_in_filter, num_out_filter))

        self.transformer = TransformerBlock()

    def forward(self, x):
        print("Forward Pass")
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            print(x.shape)

        out = x.permute(2, 0, 1) # [T, B, E]
        return out


def get_conv_output_sizes(lengths, num_layers=4):
    """ 
    Returns the length of the output after conv block.
    Using kernel size of 3 for conv1d and maxpool1d, and max pool stride of 2.
    """
    
    kernel_size = 3
    max_pool_stride = 2

    for _ in range(num_layers):
        lengths = [math.floor(((length - (kernel_size-1)) - (kernel_size) + max_pool_stride)/max_pool_stride) for length in lengths]
    return lengths