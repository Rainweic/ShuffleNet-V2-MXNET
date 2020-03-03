'''
@Author: Rainweic
@Date: 2020-02-01 10:44:58
@LastEditTime : 2020-02-03 00:26:50
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: /ShuffleNet-V2-MXNET/shufflenetv2.py
'''
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon import HybridBlock

__ALL__ = ["get_shufflenetv2_body", "getShufflenetV2"]

class ShuffleChannel(HybridBlock):

    def __init__(self, groups):
        super(ShuffleChannel, self).__init__()
        self.groups = groups

    def hybrid_forward(self, F, x):
        x.reshape((0, -4, self.groups, -1, -2)).swapaxes(1, 2).reshape((0, -3, -2))
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
        self.shufflechannel = ShuffleChannel(2)

    def hybrid_forward(self, F, x):
        # assert x.shape[1] == self.out_channels, "This feature map's in_channels ans out_channels\
        #      are not equal:{} vs {}".format(x.shape[1], self.out_channels)
        # assert x.shape[1] % 2 == 0, "This feature map's channel can't not be splited in two."
        x1, x2 = F.split(x, axis=1, num_outputs=2)
        x1 = self.conv1_bn_relu(x1)
        x1 = self.dconv_bn(x1)
        x1 = self.conv2_bn_relu(x1)
        x = F.concat(x1, x2)
        x = self.shufflechannel(x)
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

        self.shufflechannel = ShuffleChannel(2)

    def hybrid_forward(self, F, x):
        shortcut, x = x, x

        x = self.conv1_bn_relu(x)
        x = self.dconv_bn(x)
        x = self.conv2_bn_relu(x)

        shortcut = self.shortcut_dconv_bn(shortcut)
        shortcut = self.shortcut_conv_bn_relu(shortcut)

        x = F.concat(x, shortcut)
        x = self.shufflechannel(x)
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

    def __init__(self, first_channel=24, channels_per_stage=(116, 232, 464)):
        super(ShuffleNetV2, self).__init__()
        self.shufflenetv2 = nn.HybridSequential()
        self.shufflenetv2.add(
            # conv1_bn_relu
            Conv2D_BN_ReLU(first_channel, 3, 2, 1),
            # pool1
            nn.MaxPool2D(3, 2, 1),
            # stage2
            ShufflenetStage(first_channel, channels_per_stage[0], 4),
            # stage3
            ShufflenetStage(channels_per_stage[0], channels_per_stage[1], 8),
            # stage4
            ShufflenetStage(channels_per_stage[1], channels_per_stage[2], 4),
            # conv5_bn_relu
            conv5_bn_relu=Conv2D_BN_ReLU(1024, 1, 1)
        )


    def hybrid_forward(self, F, x):
        return self.shufflenetv2(x)

def get_shufflenetv2_body(type):
    channels_per_stage = {
        "0.5x": (48, 96, 192),
        "1x": (116, 232, 464),
        "1.5x": (176, 352, 704),
        "2x": (244, 488, 976)
    }
    assert type in channels_per_stage.keys(), "net type: {}, use type=xx to get net, " \
                                              "for example: type='0.5x'".format(list(channels_per_stage.keys()))
    return ShuffleNetV2(channels_per_stage=channels_per_stage[type])

def getShufflenetV2(classes, type, **kwargs):
    shufflenetv2 = get_shufflenetv2_body(type)
    net = nn.HybridSequential()
    net.add(
        shufflenetv2,
        nn.AvgPool2D(),
        nn.Dense(classes)
    )
    return net

if __name__ == '__main__':

    # test
    net = getShufflenetV2(classes=3, type='1x')
    net.initialize()

    data = nd.ones((256, 3, 224, 224))
    net.summary(data)
    out = net(data)

    print(out.shape)