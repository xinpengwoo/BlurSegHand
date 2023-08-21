import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        # Comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        
        # Flatten label and prediction tensors
        batch_size = inputs.size(0)
        
        inputs = inputs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)
        
        intersection = (inputs * targets).sum(dim=1)
        dice_loss = 1 - (2.*intersection + smooth) / (inputs.sum(dim=1) + targets.sum(dim=1) + smooth)
        
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
                
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

    
class CoordLoss(nn.Module):
    def __init__(self):
        super(CoordLoss, self).__init__()

    def forward(self, coord_out, coord_gt, valid, is_3D=None):
        loss = torch.abs(coord_out - coord_gt) * valid
        if is_3D is not None:
            loss_z = loss[:,:,2:] * is_3D[:,None,None].float()
            loss = torch.cat((loss[:,:,:2], loss_z),2)
        return loss

class CoordLossOrderInvariant(nn.Module):
    def __init__(self):
        super(CoordLossOrderInvariant, self).__init__()

    def forward(self, coord_out_e1, coord_out_e2, coord_gt_e1, coord_gt_e2, valid_e1, valid_e2, is_3D=None, return_order=False):
        # hand-wise minimize
        loss1 = (torch.abs(coord_out_e1 - coord_gt_e1) * valid_e1 + torch.abs(coord_out_e2 - coord_gt_e2) * valid_e2).mean(dim=(1,2))
        loss2 = (torch.abs(coord_out_e1 - coord_gt_e2) * valid_e2 + torch.abs(coord_out_e2 - coord_gt_e1) * valid_e1).mean(dim=(1,2))
        loss_pf = torch.min(loss1, loss2)

        if return_order:
            # 1 if e1 -> e2 else e2 -> e1
            pred_order = (loss1 < loss2).type(torch.FloatTensor).detach().to(coord_out_e1.device)
            return loss_pf, pred_order
        else:
            return loss_pf

class ParamLoss(nn.Module):
    def __init__(self):
        super(ParamLoss, self).__init__()

    def forward(self, param_out, param_gt, valid):
        loss = torch.abs(param_out - param_gt) * valid
        return loss
