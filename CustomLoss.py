import torch
import torch.nn as nn

class CustomLoss(torch.nn.Module):
    def __init__(self,alpha):
        super(CustomLoss, self).__init__()
        self.alpha=alpha
        self.bce = nn.BCELoss()

    def forward(self, y_pred, y_true):
        bce_loss = self.bce(y_pred,y_true)
        soft_dice_loss = self.sdl(y_pred,y_true)
        return (1.0-self.alpha)*soft_dice_loss  + self.alpha*bce_loss

    def sdl(self,y_pred,y_true):
        intersection = torch.sum(y_pred * y_true)
        sum_probs = torch.sum(y_pred**2) + torch.sum(y_true**2)
        soft_dice_coefficient = 1 - (2.0 * intersection) / (sum_probs + 1e-7)  # Adding a small epsilon to avoid division by zero
        loss = soft_dice_coefficient
        return loss
