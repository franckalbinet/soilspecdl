---
title: Models
---



> Models for the SoilspecDL project



::: {#2bb84e93 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t' execution_count=5}
``` {.python .cell-code}
from functools import partial
import torch
from torch.nn import (Module, AvgPool1d, LeakyReLU,
                      AdaptiveAvgPool1d, Sequential, 
                      Dropout, Linear)

from fastai.layers import ConvLayer
from typing import Callable
```
:::


::: {#d77ba715 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t' execution_count=7}
``` {.python .cell-code}
class ConvBlockBaseline(Module):    
    def __init__(self, ni, nf):
        super().__init__()
        self.conv = Sequential(
            torch.nn.Conv1d(ni, nf, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm1d(nf),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.pool = torch.nn.AvgPool1d(3)
    
    def forward(self, x): return self.pool(self.conv(x))
```
:::


::: {#bdbb1a87 .cell execution_count=8}
``` {.python .cell-code}
x = torch.randn(32, 61, 64)
block = ConvBlockBaseline(61, 128)
out = block(x)
out.shape
```

::: {.cell-output .cell-output-display}
```
torch.Size([32, 128, 21])
```
:::
:::


::: {#654a2525 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t' execution_count=9}
``` {.python .cell-code}
class ConvBlock(Module):    
    "Convolutional block with pooling."
    def __init__(self, 
                 ni, # input channels
                 nf, # output channels
                 ks=3, # kernel size
                 stride=1, # stride
                 act_cls=partial(LeakyReLU, negative_slope=0.1)
                 ):
        super().__init__()
        self.conv = ConvLayer(ni, nf, ks=ks, stride=stride, bias=False, 
                              # norm_type='NormType.Instance',
                            #   norm_type='NormType.Batch',
                              act_cls=act_cls,
                              ndim=1)
        self.pool = AvgPool1d(3)
    
    def forward(self, x): return self.pool(self.conv(x))
```
:::


For instance:

::: {#b3290e7e .cell execution_count=10}
``` {.python .cell-code}
x = torch.randn(32, 61, 64)
block = ConvBlock(61, 128)
out = block(x)
out.shape
```

::: {.cell-output .cell-output-display}
```
torch.Size([32, 128, 21])
```
:::
:::


::: {#ead64c03 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t' execution_count=11}
``` {.python .cell-code}
class FeatureExtractor(Module):
    def __init__(self, 
                 in_channel=1, # input channels
                 out_channel=16, # base number of filters
                 conv_block_cls=ConvBlock,
                 ):
        super().__init__()
        nfs = [out_channel * 2**i for i in range(5)]
        self.layers = Sequential(*[
            conv_block_cls(ni=in_channel if i==0 else nfs[i-1], nf=nfs[i]) 
            for i in range(len(nfs))
        ])
        self.adaptive_pool = AdaptiveAvgPool1d(1)
    
    def forward(self, x): 
        x = self.layers(x)
        return self.adaptive_pool(x)
```
:::


For instance:

::: {#af40f3b7 .cell execution_count=12}
``` {.python .cell-code}
fe = FeatureExtractor(in_channel=61, out_channel=16)
x = torch.randn(32, 61, 243)
out = fe(x)
out.shape
```

::: {.cell-output .cell-output-display}
```
torch.Size([32, 256, 1])
```
:::
:::


::: {#f49cdcea .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t' execution_count=13}
``` {.python .cell-code}
class MirzaiCNN(Module):
    """
    1D Convolutional Neural Network for spectral data regression/classification.
    
    Based on the architecture proposed in:
    Albinet, F., Peng, Y., Eguchi, T., Smolders, E., Dercon, G., 2022. Prediction of exchangeable potassium in soil through mid-infrared spectroscopy and deep learning: From prediction to explainability. Artificial Intelligence in Agriculture 6, 230–241. https://doi.org/10.1016/j.aiia.2022.10.001
    """
    def __init__(self, 
                 in_channel: int=1, # input channels
                 out_channel: int=16, # base number of filters
                 conv_block_cls: Callable = ConvBlock,
                 is_classifier: bool=False, # if True, model will output probabilities
                 dropout: float=0.4 # dropout rate
                 ):
        super().__init__()
        self.backbone = FeatureExtractor(in_channel, out_channel, conv_block_cls)
        self.head = Sequential(
            Dropout(dropout), 
            Linear(256, 1))
        self.is_classifier = is_classifier
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.squeeze(-1)
        x = self.head(x)
        x = x.squeeze(-1)
        return torch.sigmoid(x) if self.is_classifier else x
```
:::


For instance:

::: {#37b4856a .cell execution_count=14}
``` {.python .cell-code}
model_reg = MirzaiCNN(in_channel=1, out_channel=16)
model_cls = MirzaiCNN(in_channel=1, out_channel=16, is_classifier=True)

x = torch.randn(32, 1, 243)
pred_reg = model_reg(x)
pred_cls = model_cls(x)

pred_reg.shape, pred_cls.shape, pred_cls.min().item(), pred_cls.max().item()
```

::: {.cell-output .cell-output-display}
```
(torch.Size([32]), torch.Size([32]), 0.22773000597953796, 0.5989965796470642)
```
:::
:::


::: {#86f4147b .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t' execution_count=15}
``` {.python .cell-code}
def cnt_params(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel():,} parameters")
    print(f"\nTotal: {sum(p.numel() for p in model.parameters()):,} parameters")
```
:::


::: {#f2ea9599 .cell execution_count=16}
``` {.python .cell-code}
cnt_params(model_reg)
```

::: {.cell-output .cell-output-stdout}
```
backbone.layers.0.conv.0.weight: 48 parameters
backbone.layers.0.conv.1.weight: 16 parameters
backbone.layers.0.conv.1.bias: 16 parameters
backbone.layers.1.conv.0.weight: 1,536 parameters
backbone.layers.1.conv.1.weight: 32 parameters
backbone.layers.1.conv.1.bias: 32 parameters
backbone.layers.2.conv.0.weight: 6,144 parameters
backbone.layers.2.conv.1.weight: 64 parameters
backbone.layers.2.conv.1.bias: 64 parameters
backbone.layers.3.conv.0.weight: 24,576 parameters
backbone.layers.3.conv.1.weight: 128 parameters
backbone.layers.3.conv.1.bias: 128 parameters
backbone.layers.4.conv.0.weight: 98,304 parameters
backbone.layers.4.conv.1.weight: 256 parameters
backbone.layers.4.conv.1.bias: 256 parameters
head.1.weight: 256 parameters
head.1.bias: 1 parameters

Total: 131,857 parameters
```
:::
:::


