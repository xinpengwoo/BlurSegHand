import copy
import math
import torch
import torch.nn as nn

from losses import CoordLoss, ParamLoss, CoordLossOrderInvariant, DiceBCELoss
from models.modules.ktformer import KTFormer
from models.modules.regressor import Regressor
from models.modules.resnetbackbone import ResNetBackbone
from models.modules.unfolder import Unfolder
from models.modules.layer_utils import init_weights
from utils.MANO import mano
from models.modules.unetDecoder import UnetDecoder

class BlurHandNet(nn.Module):
    def __init__(self, opt, weight_init=True):
        super().__init__()
        # define trainable module
        opt_net = opt['network']
        self.img_backbone = ResNetBackbone()  # img backbone
        self.seg_unetDecoder = UnetDecoder() # seg decoder
        
        # weight initialization
        if weight_init:
            self.img_backbone.init_weights()
            self.seg_unetDecoder.apply(init_weights)
        
        # losses
        self.coord_loss = CoordLoss()
        
    def forward(self, inputs, targets, meta_info, mode):
        # extract image feature from backbone
        feat_blur_img, feat_pyramid_img = self.img_backbone(inputs['img'])
        # segmentation
        seg_mask = self.seg_unetDecoder(feat_pyramid_img)
    
        if mode == 'train':
            loss = {}
            
            # losses on segegmentation mask
            loss['seg_mask'] = self.opt_loss['lambda_seg_mask'] * self.seg_loss(seg_mask, targets['seg_mask'])

            return loss
            
        else:
            out = {}
            out['img'] = inputs['img']
            
            # our model predictions
            out['seg_mask'] = seg_mask

            return out
   