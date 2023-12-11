import torch
import torch.nn as nn

from open_sam.registry import MODELS


@MODELS.register_module()
class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=0.25, loss_weight=1.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.loss_weight = loss_weight

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        num_pos = torch.sum(mask)
        num_neg = mask.numel() - num_pos
        w_pos = (1 - p)**self.gamma
        w_neg = p**self.gamma

        loss_pos = -self.alpha * mask * w_pos * torch.log(p + 1e-12)
        loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - p +
                                                                      1e-12)

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos +
                                                              num_neg + 1e-12)

        return loss * self.loss_weight
