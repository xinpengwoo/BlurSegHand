import torch
from torch import nn, cat

from models.modules.layer_utils import make_unet_deconv_layers


class UnetDecoder(nn.Module):

    def __init__(self, num_classes=1, num_filters=32, model="resnet50"): # Dropout=.2, 
        
        super().__init__()

        if model in ["resnet18", "resnet34"]: model = "resnet18-34"
        else: model = "resnet50-101"
        self.filters_dict = {
            "resnet18-34": [512, 512, 256, 128, 64],
            "resnet50-101": [2048, 2048, 1024, 512, 256]
        }
        
        self.num_classes = num_classes
        #self.Dropout = Dropout
        
        self.pool = nn.MaxPool2d(2, 2)
        self.center = make_unet_deconv_layers(self.filters_dict[model][0], num_filters * 8 * 2, 
                                         num_filters * 8)
        self.dec5 = make_unet_deconv_layers(self.filters_dict[model][1] + num_filters * 8, 
                                       num_filters * 8 * 2, num_filters * 8)    
        self.dec4 = make_unet_deconv_layers(self.filters_dict[model][2] + num_filters * 8, 
                                       num_filters * 8 * 2, num_filters * 8)
        self.dec3 = make_unet_deconv_layers(self.filters_dict[model][3] + num_filters * 8, 
                                       num_filters * 4 * 2, num_filters * 2)
        self.dec2 = make_unet_deconv_layers(self.filters_dict[model][4] + num_filters * 2, 
                                       num_filters * 2 * 2, num_filters * 2 * 2)
        
        self.dec1 = make_unet_deconv_layers(num_filters * 2 * 2, num_filters * 2 * 2, num_filters)
        self.dec0 = nn.Sequential(nn.Conv2d(num_filters, num_filters, 3, padding=1),
                                  nn.ReLU(inplace=True))
        
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        #self.dropout_2d = nn.Dropout2d(p=self.Dropout)
        

    def forward(self, feat_pyramid):
        conv1 = feat_pyramid['stride2']
        conv2 = feat_pyramid['stride4']
        conv3 = feat_pyramid['stride8']
        conv4 = feat_pyramid['stride16']
        conv5 = feat_pyramid['stride32']

        center = self.center(self.pool(conv5))
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        #dec2 = self.dropout_2d(dec2)
            
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        return self.final(dec0)
        
