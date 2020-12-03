import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
#from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

class conv_block(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size=3,stride=1,padding=0,dilation=1,groups=1,bias=True,bn=True,relu=True):
        super(conv_block,self).__init__()
        self.input_channels=input_channels
        self.output_channels=output_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding,self.dilation,self.groups,self.bias=padding,dilation,groups,bias
        self.conv=nn.Conv2d(self.input_channels,self.output_channels,self.kernel_size,self.stride,self.padding,self.dilation,self.groups,self.bias)
        self.bn=nn.BatchNorm2d(self.output_channels)
        self.relu=nn.ReLU()

    def forward(self,inputs):
        x=inputs
        x=self.conv(x)
        if self.bn:
            x=self.bn(x)
        if self.relu:
            x=self.relu(x)
        return x

class msm_block(nn.Module):
    def __init__(self,pool_size):
        super(msm_block,self).__init__()
        self.pool_size=pool_size
        self.relu=nn.ReLU()
        self.avepool=nn.AvgPool2d(self.pool_size,stride=self.pool_size)
        self.pool_conv=nn.Conv2d(512,256,kernel_size=1,bias=False)
        self.bl=nn.UpsamplingBilinear2d(scale_factor=self.pool_size)
        self.resize_conv=nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,bias=False)

    def forward(self,inputs):
        x=inputs
        x=self.relu(x)
        x=self.avepool(x)
        x=self.pool_conv(x)
        x=self.bl(x)
        x=self.resize_conv(x)
        return x
class Feature_module(nn.Module):
    def __init__(self,input_channels=4,input_rows=128,input_cols=128):
        super(Feature_module,self).__init__()
        self.input_channels=input_channels
        self.input_rows=input_rows
        self.input_cols=input_cols
        #layers
        self.conv1=conv_block(input_channels,64,3,stride=2,padding=1,bias=False,bn=False,relu=True)
        self.conv2=conv_block(64,96,3,1,1,bias=False,bn=True,relu=True)
        self.conv3=conv_block(96,128,3,1,1,bias=False,bn=True,relu=True)
        self.maxpool1=nn.MaxPool2d(2)
        self.conv4=conv_block(128,192,padding=1,bias=False,bn=True,relu=True)
        self.conv5=conv_block(192,256,3,1,1,bias=False,bn=True,relu=True)
        self.maxpool2=nn.MaxPool2d(2)
        self.conv6=conv_block(256,256,3,1,1,bias=False,bn=True,relu=True)
        self.conv7=conv_block(256,512,3,1,1,bias=False,bn=True,relu=True)
    
    def forward(self,inputs):
        #x=F.interpolate(inputs,[self.input_rows,self.input_cols])
        x=self.conv1(inputs)
        x=self.conv2(x)
        x1=self.conv3(x)
        x=self.maxpool1(x1)
        x=self.conv4(x)
        x2=self.conv5(x)
        x=self.maxpool2(x2)
        x=self.conv6(x)
        x3=self.conv7(x)
        return x1,x2,x3
        
class MS_module(nn.Module):
    def __init__(self):
        super(MS_module,self).__init__()
        self.msm_block1=msm_block(16)
        self.msm_block2=msm_block(8)
        self.msm_block3=msm_block(4)
        self.msm_block4=msm_block(2)

    def forward(self,inputs):
        x1=self.msm_block1(inputs)
        x2=self.msm_block2(inputs)
        x3=self.msm_block3(inputs)
        x4=self.msm_block4(inputs)
        return x1,x2,x3,x4

class MSNetwork(nn.Module):
    def __init__(self,input_channels=4,nclass=2,input_rows=128,input_cols=128):
        super(MSNetwork,self).__init__()
        self.input_rows=input_rows
        self.input_cols=input_cols
        self.nclass=nclass
        self.input_channels=input_channels
        self.feature_module=Feature_module(self.input_channels,self.input_rows,self.input_cols)
        self.ms_module=MS_module()
        self.conv_up1=conv_block(1536,512,3,1,1,bias=False,bn=False,relu=True)
        self.conv_up2=conv_block(768,256,3,1,1,bias=False)
        self.conv_up3=conv_block(384,128,3,1,1,bias=False)
        self.upsampling=nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.conv_out=nn.Conv2d(128,self.nclass,3,1,1,bias=False)
        #self.dropout=nn.Dropout(0.5)

    def forward(self,inputs):
        features1,features2,features3=self.feature_module(inputs)
        ms1,ms2,ms3,ms4=self.ms_module(features3)
        #concat the output of the multiscale module
        ms=torch.cat([ms1,ms2,ms3,ms4],dim=1)
        #upsample_block1:concat-conv-upsample
        features3=F.relu(features3)
        outputs=torch.cat([features3,ms],dim=1)
        outputs=self.conv_up1(outputs)
        outputs=self.upsampling(outputs)
        #upsample_block2:concat-conv-upsample
        features2=F.relu(features2)
        outputs=torch.cat([outputs,features2],dim=1)
        outputs=self.conv_up2(outputs)
        outputs=self.upsampling(outputs)
        #upsample_block2:concat-conv-upsample
        features2=F.relu(features1)
        outputs=torch.cat([outputs,features1],dim=1)
        outputs=self.conv_up3(outputs)
        outputs=self.upsampling(outputs)
        #outputs=self.dropout(outputs)
        outputs=self.conv_out(outputs)
        return outputs


if __name__=="__main__":
    net=MSNetwork()
    net=net.cuda()
    summary(net,input_size=(4,384,384))











        





