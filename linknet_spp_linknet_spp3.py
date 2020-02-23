import torch
import torch.nn as nn
from torch.autograd import Variable
#from torchvision.models import resnet
import torch.nn.functional as F
from resnet import *
#from ptsemseg.models.utils import *

class conv2DBatchNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        with_bn=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels),
                             int(n_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation,)

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod, 
                                          nn.BatchNorm2d(int(n_filters)), 
                                          nn.ReLU(inplace=True))
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class pyramidPooling(nn.Module):
    def __init__(
        self,
        in_channels,
        pool_sizes,
        model_name="pspnet",
        fusion_mode="cat",
        with_bn=True,
    ):
        super(pyramidPooling, self).__init__()

        bias = not with_bn

        self.paths = []
        for i in range(len(pool_sizes)):
            self.paths.append(
                conv2DBatchNormRelu(
                    in_channels,
                    int(in_channels / len(pool_sizes)),
                    1,
                    1,
                    0,
                    bias=bias,
                    with_bn=with_bn,
                )
            )

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes
        self.model_name = model_name
        self.fusion_mode = fusion_mode

    def forward(self, x):
        h, w = x.shape[2:]
        k_sizes = []
        strides = []
        for pool_size in self.pool_sizes:
            k_sizes.append((int(h / pool_size), int(w / pool_size)))
            strides.append((int(h / pool_size), int(w / pool_size)))
        #mobarak
        # if self.training or self.model_name != "icnet":  # general settings or pspnet
        #     k_sizes = []
        #     strides = []
        #     for pool_size in self.pool_sizes:
        #         k_sizes.append((int(h / pool_size), int(w / pool_size)))
        #         strides.append((int(h / pool_size), int(w / pool_size)))
        # else:  # eval mode and icnet: pre-trained for 1025 x 2049
        #     k_sizes = [(8, 15), (13, 25), (17, 33), (33, 65)]
        #     strides = [(5, 10), (10, 20), (16, 32), (33, 65)]
        print('PPK:',k_sizes)
        print('PPS:',strides)
        if self.fusion_mode == "cat":  # pspnet: concat (including x)
            output_slices = [x]

            for i, (module, pool_size) in enumerate(
                zip(self.path_module_list, self.pool_sizes)
            ):
                
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                
                # out = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
                if self.model_name != "icnet":
                    out = module(out)
                out = F.upsample(out, size=(h, w), mode="bilinear")
                output_slices.append(out)

            return torch.cat(output_slices, dim=1)
        else:  # icnet: element-wise sum (including x)
            pp_sum = x

            for i, (module, pool_size) in enumerate(
                zip(self.path_module_list, self.pool_sizes)
            ):
                print('in',x.size())
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                print('SPP_out1:',out.size())
                # out = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
                if self.model_name != "icnet":
                    out = module(out)
                    print('not icnet')
                print('SPP_out1:',out.size())

                out = F.upsample(out, size=(h, w), mode="bilinear")
                #print('SPP_out2:', out.size())
                #print('SPP_pp_sum:', pp_sum.size())
                #pp_sum = F.upsample(pp_sum, size=(110, 125), mode="bilinear")
                pp_sum = pp_sum + out

            return pp_sum


class bottleNeckPSP(nn.Module):
    def __init__(
        self, in_channels, mid_channels, out_channels, stride, dilation=1, with_bn=True
    ):
        super(bottleNeckPSP, self).__init__()

        bias = not with_bn

        self.cbr1 = conv2DBatchNormRelu(
            in_channels,
            mid_channels,
            1,
            stride=1,
            padding=0,
            bias=bias,
            with_bn=with_bn,
        )
        if dilation > 1:
            self.cbr2 = conv2DBatchNormRelu(
                mid_channels,
                mid_channels,
                3,
                stride=stride,
                padding=dilation,
                bias=bias,
                dilation=dilation,
                with_bn=with_bn,
            )
        else:
            self.cbr2 = conv2DBatchNormRelu(
                mid_channels,
                mid_channels,
                3,
                stride=stride,
                padding=1,
                bias=bias,
                dilation=1,
                with_bn=with_bn,
            )
        self.cb3 = conv2DBatchNorm(
            mid_channels,
            out_channels,
            1,
            stride=1,
            padding=0,
            bias=bias,
            with_bn=with_bn,
        )
        self.cb4 = conv2DBatchNorm(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=bias,
            with_bn=with_bn,
        )

    def forward(self, x):
        conv = self.cb3(self.cbr2(self.cbr1(x)))
        residual = self.cb4(x)
        return F.relu(conv + residual, inplace=True)

class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size, 1, padding, groups=groups, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_planes),)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Encoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(Encoder, self).__init__()
        self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias)
        self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        return x


class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)

        return x


class LinkNet(nn.Module):
    """
    Generate Model Architecture
    """

    def __init__(self, n_classes=21):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(LinkNet, self).__init__()
        with_bn = True
        #base = resnet.resnet18(pretrained=True)
        self.pyramid_pooling = pyramidPooling(
            1024, [14,8,4,2], model_name="icnet", fusion_mode="sum", with_bn=with_bn
        )
        base = resnet18(pretrained=True)

        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )


        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 0)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)
        self.lsm = nn.LogSoftmax(dim=1)


    def forward(self, x):
        # Initial block
        #print('pool',x.size())
        x = self.in_block(x)
        #print('pool1',x.size())
        #x = self.pyramid_pooling(x) #top
        #print('pool2', x.size())
        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        #e2 = F.upsample(e2, (56, 64), mode='bilinear')
        e3 = self.encoder3(e2)
        
        #e3 = self.pyramid_pooling(e3) #mid
        
        e4 = self.encoder4(e3)
        #print('e4',e4.size())
        print('e4',e4.size())
        e4 = self.pyramid_pooling(e4) #end
        print('aft spp',e4.size())
        #print('e4',e4.size())
        # Decoder blocks
        #d4 = e3 + self.decoder4(e4)
        d4 = e3 + self.decoder4(e4)
        #print('d4', d4.size())

        d3 = e2 + self.decoder3(d4)
        #print('d3', d3.size())
        #e1 = F.upsample(e1, (110, 126), mode='bilinear')
        #print(e1.size(), self.decoder2(d3).size())
        d2 = e1 + F.upsample(self.decoder2(d3), (e1.size(2), e1.size(3)), mode='bilinear')
        #print('d2', d2.size())
        d1 = x + self.decoder1(d2)
        #print('d1', d1.size())
        # Classifier
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        y = self.lsm(y)
        #print('final:',y.size())

        return y

class LinkNetBase(nn.Module):
    """
    Generate model architecture
    """

    def __init__(self, n_classes=2):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(LinkNetBase, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.encoder1 = Encoder(64, 64, 3, 1, 1)
        self.encoder2 = Encoder(64, 128, 3, 2, 1)
        self.encoder3 = Encoder(128, 256, 3, 2, 1)
        self.encoder4 = Encoder(256, 512, 3, 2, 1)

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 0)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Initial block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder blocks
        #d4 = e3 + self.decoder4(e4)
        d4 = e3 + self.decoder4(e4)
        d3 = e2 + self.decoder3(d4)
        #d2 = e1 + self.decoder2(d3)
        d2 = e1 + F.upsample(self.decoder2(d3), (e1.size(2), e1.size(3)), mode='bilinear')
        d1 = x + self.decoder1(d2)

        # Classifier
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        y = self.lsm(y)

        return y
