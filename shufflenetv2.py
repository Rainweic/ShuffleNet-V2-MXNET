'''
@Author: Rainweic
@Date: 2020-02-01 10:44:58
@LastEditTime : 2020-02-02 00:00:06
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: /ShuffleNet-V2-MXNET/shufflenetv2.py
'''
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon import HybridBlock

def shuffle_channel(x, groups):
    n, c, h, w = x.shape
    assert c % groups == 0, "Channel:{}, Groups:{}, Channel % Group != 0, shuffle_channel can't work!".format(
        c, groups
    )
    x = nd.Reshape(x, shape=(n, groups, c // groups, h, w))
    x = nd.transpose(x, axes=(0, 2, 1, 3, 4))
    x = nd.Reshape(x, shape=(n, -1, h, w))
    return x
    

class Conv2D_BN_ReLU(HybridBlock):
    '''Conv2D -> BN -> ReLU'''

    def __init__(self, out_channels, kernel_size=1, stride=1, padding=0):
        super(Conv2D_BN_ReLU, self).__init__()
        self.conv = nn.Conv2D(out_channels, kernel_size, stride, padding, use_bias=False)
        self.bn = nn.BatchNorm()
        self.relu = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DepthwiseConv2D_BN(HybridBlock):
    '''DepthwiseConv2D -> BN'''

    def __init__(self, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseConv2D_BN, self).__init__()
        self.dconv = nn.Conv2D(out_channels, kernel_size, stride, padding, groups=out_channels, use_bias=False)
        self.bn = nn.BatchNorm()

    def hybrid_forward(self, F, x):
        x = self.dconv(x)
        x = self.bn(x)
        return x

class ShufflenetUnit1(HybridBlock):
    '''The unit of shufflenetv2 for stride=1'''
    
    def __init__(self, out_channels):
        super(ShufflenetUnit1, self).__init__()
        self.out_channels = out_channels
        self.conv1_bn_relu = Conv2D_BN_ReLU(out_channels // 2, 1, 1)
        self.dconv_bn = DepthwiseConv2D_BN(out_channels // 2, 3)
        self.conv2_bn_relu = Conv2D_BN_ReLU(out_channels // 2, 1, 1)

    def hybrid_forward(self, F, x):
        assert x.shape[1] == self.out_channels, "This feature map's in_channels ans out_channels\
             are not equal:{} vs {}".format(x.shape[1], self.out_channels)
        assert x.shape[1] % 2 == 0, "This feature map's channel can't not be splited in two."
        x1, x2 = F.split(x, axis=1, num_outputs=2)
        x1 = self.conv1_bn_relu(x1)
        x1 = self.dconv_bn(x1)
        x1 = self.conv2_bn_relu(x1)
        x = F.concat(x1, x2)
        x = shuffle_channel(x, 2)
        return x

class ShufflenetUnit2(HybridBlock):
    '''The unit of shufflenetv2 for stride=2'''

    def __init__(self, in_channels, out_channels):
        super(ShufflenetUnit2, self).__init__()
        self.out_channels = out_channels
        
        self.conv1_bn_relu = Conv2D_BN_ReLU(out_channels // 2, 1, 1)
        self.dconv_bn = DepthwiseConv2D_BN(out_channels // 2, 3, 2)
        self.conv2_bn_relu = Conv2D_BN_ReLU(out_channels - in_channels, 1, 1)

        self.shortcut_dconv_bn = DepthwiseConv2D_BN(in_channels, 3, 2)
        self.shortcut_conv_bn_relu = Conv2D_BN_ReLU(in_channels, 1, 1)

    def hybrid_forward(self, F, x):
        shortcut, x = x, x

        x = self.conv1_bn_relu(x)
        x = self.dconv_bn(x)
        x = self.conv2_bn_relu(x)

        shortcut = self.shortcut_dconv_bn(shortcut)
        shortcut = self.shortcut_conv_bn_relu(shortcut)

        x = F.concat(x, shortcut)
        x = shuffle_channel(x, 2)
        return x

class ShufflenetStage(HybridBlock):

    def __init__(self, in_channels, out_channels, num_blocks):
        super(ShufflenetStage, self).__init__()

        self.ops = nn.HybridSequential()
        for i in range(num_blocks):
            if i == 0:
                op = ShufflenetUnit2(in_channels, out_channels)
            else:
                op = ShufflenetUnit1(out_channels)
            self.ops.add(op)

    def hybrid_forward(self, F, x):
        return self.ops(x)

class ShuffleNetV2(HybridBlock):

    def __init__(self, num_classes, first_channel=24, channels_per_stage=(116, 232, 464)):
        super(ShuffleNetV2, self).__init__()
        self.num_classes = num_classes
        self.conv1_bn_relu = Conv2D_BN_ReLU(first_channel, 3, 2, 1)
        self.pool1 = nn.MaxPool2D(3, 2, 1) 
        self.stage2 = ShufflenetStage(first_channel, channels_per_stage[0], 4)
        self.stage3 = ShufflenetStage(channels_per_stage[0], channels_per_stage[1], 8)
        self.stage4 = ShufflenetStage(channels_per_stage[1], channels_per_stage[2], 4)
        self.conv5_bn_relu = Conv2D_BN_ReLU(1024, 1, 1)
        self.gap = nn.GlobalAvgPool2D()
        self.linear = nn.Dense(num_classes)

    def hybrid_forward(self, F, x):
        x = self.conv1_bn_relu(x)
        x = self.pool1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5_bn_relu(x)
        x = self.gap(x)
        x = self.linear(x)
        x = F.SoftmaxActivation(x)
        return x

def getShufflenetV2(type, num_classes):
    channels_per_stage = {
        "0.5x": (48, 96, 192),
        "1x": (116, 232, 464),
        "1.5x": (176, 352, 704),
        "2x": (244, 488, 976)
    }
    assert type in channels_per_stage.keys(), "net type: {}".format(list(net.keys()))
    net = ShuffleNetV2(num_classes, channels_per_stage=channels_per_stage[type])
    return net


    