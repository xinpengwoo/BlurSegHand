import torch
import torch.nn as nn
import numpy as np

def dice_coef(y_true, y_pred, eps=1e-15, smooth=1.):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return (2. * intersection + smooth) / (union + smooth + eps)

def jaccard(intersection, union, eps=1e-15):
    return (intersection) / (union - intersection + eps)

def dice(intersection, union, eps=1e-15, smooth=1.):
    return (2. * intersection + smooth) / (union + smooth + eps)

class DiceBCELoss(nn.Module):

    def __init__(self, bce_weight=0.5, mode="dice", eps=1e-7, weight='none', smooth=1.):
        super(DiceBCELoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction=weight)
        self.bce_weight = bce_weight
        self.eps = eps
        self.mode = mode
        self.smooth = smooth

    def forward(self, outputs, targets):
        batch_size = outputs.size(0)
        outputs = outputs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)
        outputs = torch.sigmoid(outputs)    
        loss = self.bce_weight * torch.mean(self.bce_loss(outputs, targets),1)

        if self.bce_weight < 1.:
            intersection = (outputs * targets).sum(dim=1)
            union = outputs.sum(dim=1) + targets.sum(dim=1)
            if self.mode == "dice":
                score = dice(intersection, union, self.eps, self.smooth)
            elif self.mode == "jaccard":
                score = jaccard(intersection, union, self.eps)
            loss -= (1 - self.bce_weight) * torch.log(score)
        return loss

    
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
