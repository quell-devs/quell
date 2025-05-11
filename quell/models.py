from torch import nn
import torch

class Identity(nn.Module):
    """Identity model that returns input unchanged."""
    def __init__(self):
        super().__init__()
        # dummy parameter
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x

            
class ConvLayersModel(nn.Module):
    def __init__(self, in_channels: int, filters: int, kernel_size: int, layers: int):
        super().__init__()
        self.layers = layers
        
        self.initial_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=kernel_size,
            padding_mode="reflect",
            padding="same",
        )
        self.initial_bn = nn.BatchNorm3d(filters)
        self.initial_activation = nn.PReLU()
        
        self.layers = nn.ModuleList()
        for _ in range(layers - 2):
            self.layers.append(nn.Conv3d(
                in_channels=filters,
                out_channels=filters,
                kernel_size=kernel_size,
                padding_mode="reflect",
                padding="same",
            ))
            self.layers.append(nn.BatchNorm3d(filters))
            self.layers.append(nn.PReLU())
        
        self.final_conv = nn.Conv3d(
            in_channels=filters,
            out_channels=1,
            kernel_size=1,
        )
        self.final_activation = nn.Tanh()

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_activation(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_conv(x)
        x = self.final_activation(x)
        
        return x
            

def conv(in_channels:int, filters:int, kernel_size:int, padding:int|str, activation:nn.Module, dilation:int=1, batch_norm:bool=False):
    layers = []
    layers.append(
        nn.Conv3d(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=kernel_size,
            padding_mode="reflect",
            padding=padding,
            dilation=dilation,
            bias=not batch_norm
        )
    )
    if batch_norm:
        layers.append(nn.BatchNorm3d(filters))
    if activation is not None:
        layers.append(activation)
        
    return nn.Sequential(*layers)

def conv_block(in_channels:int, filters:int, kernel_size:int, padding:int, activation:nn.Module):
    return nn.Sequential(
        conv(in_channels,filters,kernel_size,padding,activation),
        conv(filters,filters,kernel_size,padding,activation),
    )

def down_block(in_channels:int, filters:int, kernel_size:int, padding:int, activation:nn.Module):
    return nn.Sequential(
        nn.MaxPool3d(kernel_size=2, stride=2),
        conv_block(in_channels,filters,kernel_size,padding,activation),
    )

def up_block(in_channels:int, filters:int, kernel_size:int, padding:int, activation:nn.Module):
    return nn.Sequential(
        conv_block(in_channels,filters,kernel_size,padding,activation),
        nn.Upsample(
            scale_factor=2, mode="trilinear"
        ),
    )

class Unet(nn.Module):
    def __init__(self, in_channels:int, filters:int, kernel_size:int, layers:int):
        super().__init__()
        assert layers > 0, "Layers (number skip connections) must be positive integer"
        self.layers = layers
        p = kernel_size//2
        
        self.inconv = nn.Sequential(
            conv_block(in_channels,filters,kernel_size,p,nn.ReLU()),
            )
        
        self.down_blocks = nn.ModuleList()
        for i in range(layers):
            n_filters = filters*(2**i)
            self.down_blocks.append(down_block(n_filters,n_filters*2,kernel_size,p,nn.ReLU()))

        self.bottleneck = nn.Upsample(scale_factor=2, mode="trilinear")

        self.up_blocks = nn.ModuleList()
        for i in range(layers,1,-1):
            out_filters = filters*(2**(i-1))
            in_filters = filters*(2**i) + out_filters
            self.up_blocks.append(up_block(in_filters,out_filters,kernel_size,p,nn.ReLU()))

        self.outconv = nn.Sequential(
            conv(filters*2+filters,filters,kernel_size,p,nn.ReLU()),
            conv(filters,in_channels,1,0,None),
            )


    def forward(self,x):
        skip_conn = []

        x = self.inconv(x) 
        skip_conn.append(x)

        for i in range(self.layers-1):
            x = self.down_blocks[i](x)
            skip_conn.append(x)

        x = self.bottleneck(self.down_blocks[-1](x)) 

        for i in range(self.layers-1): 
            x = self.up_blocks[i](torch.cat((skip_conn.pop(), x), dim=1))

        xout = self.outconv(torch.cat((skip_conn.pop(), x), dim=1))

        return xout
    