"""Losses
    * https://github.com/JunMa11/SegLoss
"""


from torch.nn import functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable

def get_loss_function(loss_function_str: str):

    if loss_function_str == 'MeanCCELoss':

        return CCE

    elif loss_function_str == 'GDLoss':

        return GeneralizedDiceLoss
    
    elif loss_function_str == 'IoULoss':
        return IoULoss
    
    elif loss_function_str == 'FocalLoss':
        return FocalLoss
    
    elif loss_function_str == 'FocalLoss2d':
        return FocalLoss2d

class CCE(nn.Module):

    def __init__(self, weight, **kwargs):
        super(CCE, self).__init__()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight = torch.Tensor(weight).to(device)

    def forward(self, inputs, targets):
        
        loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        unique_values, unique_counts = torch.unique(targets, return_counts=True)
        selected_weight = torch.index_select(input=self.weight, dim=0, index=unique_values)

        numerator = loss.sum()                               # weighted losses
        denominator = (unique_counts*selected_weight).sum()  # weigthed counts

        loss = numerator/denominator

        return loss


class GeneralizedDiceLoss(nn.Module):
    
    def __init__(self, **kwargs):
        super(GeneralizedDiceLoss, self).__init__()
        self.scaler = nn.Softmax(dim=1)  # Softmax for loss

    def forward(self, inputs, targets):

        targets = targets.contiguous()
        targets = torch.nn.functional.one_hot(targets.to(torch.int64), inputs.size()[1])  # B, H, W, C

        inputs = inputs.contiguous()
        inputs = self.scaler(inputs)
        inputs = inputs.permute(0, 2, 3, 1)  # B, H, W, C

        w = 1. / (torch.sum(targets, (0, 1, 2)) ** 2 + 1e-9)

        numerator = targets * inputs
        numerator = w * torch.sum(numerator, (0, 1, 2))
        numerator = torch.sum(numerator)

        denominator = targets + inputs
        denominator = w * torch.sum(denominator, (0, 1, 2))
        denominator = torch.sum(denominator)

        dice = 2. * (numerator + 1e-9) / (denominator + 1e-9)

        return 1. - dice

#Jaccard/Intersection over Union (IoU) Loss
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

#Focal Loss
#PyTorch
ALPHA = 0.8
GAMMA = 2

#discard
class FocalLoss(nn.Module):
    def __init__(self, weight, size_average=True):
        super(FocalLoss, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight = torch.Tensor(weight).to(device)

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        #inputs = inputs.view(-1)
        #targets = targets.view(-1)
        if inputs.dim()>2:
            inputs = inputs.contiguous().view(inputs.size(0), inputs.size(1), -1)
            inputs = inputs.transpose(1,2)
            inputs = inputs.contiguous().view(-1, inputs.size(2)).squeeze()
        if targets.dim()==4:
            targets = targets.contiguous().view(targets.size(0), targets.size(1), -1)
            targets = targets.transpose(1,2)
            targets = targets.contiguous().view(-1, targets.size(2)).squeeze()
        elif targets.dim()==3:
            targets = targets.view(-1)
        else:
            targets = targets.view(-1, 1)
        
        #first compute binary cross-entropy 
        #BCE = F.binary_cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

#https://github.com/doiken23/focal_segmentation/blob/master/focalloss2d.py    
class FocalLoss2d(nn.Module):

    def __init__(self, gamma=1, weight=None, size_average=True):
        super(FocalLoss2d, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight = torch.Tensor(weight).to(device)
        self.gamma = gamma
        
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        # compute the negative likelyhood
        weight = Variable(self.weight)
        logpt = -F.cross_entropy(input, target)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1-pt)**self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
