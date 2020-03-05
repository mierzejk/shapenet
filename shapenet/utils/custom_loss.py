# Thanks to https://discuss.pytorch.org/t/rmse-loss-function/16540
import torch
import torch.nn.functional as F


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = torch.nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


class L1Loss_IOD(torch.nn.L1Loss):
    __constants__ = ['right_eye_outer_corner_idx', 'left_eye_outer_corner_idx']

    def __init__(self, right_eye_outer_corner_idx=36, left_eye_outer_corner_idx=45, size_average=None, reduce=None, reduction='mean'):
        super(L1Loss_IOD, self).__init__(size_average, reduce, reduction)
        self.right_eye_outer_corner_idx = right_eye_outer_corner_idx
        self.left_eye_outer_corner_idx = left_eye_outer_corner_idx

    def forward(self, input, target):
        return torch.mean(F.l1_loss(input,
                                    target,
                                    reduction='none').mean((1,2)) /
                          torch.cdist(target[:,self.right_eye_outer_corner_idx],
                                      target[:,self.left_eye_outer_corner_idx]).diag())
