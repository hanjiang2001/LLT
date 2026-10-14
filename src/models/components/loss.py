import torch.nn as nn


class SeparationLoss(nn.Module):

    def __init__(self, smoothing=0.2):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        pred = pred / pred.norm(dim=-1, keepdim=True)
        target = target / target.norm(dim=-1, keepdim=True)
        smoothed_target = (1 - self.smoothing) * target + self.smoothing / target.size(-1)
        loss = 2 - 2 * (pred * smoothed_target.detach()).sum(dim=0).mean()
        return loss


class LookupAlignmentLoss(nn.Module):

    def __init__(self, smoothing=0.2):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, targets):
        pred = pred / pred.norm(dim=-1, keepdim=True)
        targets = targets / targets.norm(dim=-1, keepdim=True)
        smoothed_target = (1 - self.smoothing) * targets + self.smoothing / targets.size(-1)
        pred_expanded = pred.expand_as(smoothed_target)
        loss = 2 - 2 * (pred_expanded * smoothed_target.detach()).sum(dim=0).mean()
        return loss
