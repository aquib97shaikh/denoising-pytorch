from torch import nn
import torch.nn.functional as F
import torch
class CrossNet(nn.Module):
    def __init__(self):
        super(CrossNet, self).__init__()

        self.cross = nn.ModuleList(
            [Block_x_8(3,90), Block_x4(186,48), Block_x_2(51,96) , Block_x4(99,24)] #[Block_x_8(1,90), Block_x4(186,48), Block_x_2(51,96) , Block_x4(99,24)]
        )
        self.pool = nn.ModuleList(
            [nn.MaxPool2d(8,8), nn.MaxPool2d(2,2), nn.MaxPool2d(4,4) , nn.MaxPool2d(1,1)]
        )
        self.last = BNReLUConv(27,3)
    def forward(self,x):
        residual = x
        for i,block in enumerate(self.cross):
            print(x.shape)
            x = block(x)
            x = torch.cat((x,self.pool[i](residual)),dim=1)
        x = self.last(x)
        return x+residual
class Block_x_8(nn.Sequential):
    def __init__(self, in_channels, channels, inplace=True):
        super(Block_x_8, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))  
        self.add_module('conv-x/2', nn.Conv2d(in_channels, in_channels+channels+1, 3, 2, 1)) 
        self.add_module('relu', nn.ReLU(inplace=inplace))  
        self.add_module('conv-x/4', nn.Conv2d(in_channels+channels+1, in_channels + 2*channels, 3, 2, 1))
        self.add_module('relu', nn.ReLU(inplace=inplace))  
        self.add_module('conv-x/8', nn.Conv2d(in_channels+2*channels, in_channels + 2*channels, 3, 2, 1))
        
class Block_x_2(nn.Sequential):
    def __init__(self, in_channels, channels, inplace=True):
        super(Block_x_2, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))  
        self.add_module('conv-x/2', nn.Conv2d(in_channels, channels, 3, 2, 1))
         
class Block_x4(nn.Sequential):
    def __init__(self,in_channels, channels, inplace=True):
        super(Block_x4, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))  
        self.add_module('conv-4x', nn.ConvTranspose2d(in_channels, channels, 4, stride=4))
    
class BNReLUConv(nn.Sequential):
    def __init__(self, in_channels, channels, inplace=True):
        super(BNReLUConv, self).__init__()
        self.add_module('bn -last', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace)) 
        self.add_module('conv - last', nn.Conv2d(in_channels, channels, 3, 1, 1)) 
