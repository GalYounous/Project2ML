import torch

class SoftDiceLoss(torch.nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, y_pred, y_true):
        intersection = torch.sum(y_pred * y_true)
        sum_probs = torch.sum(y_pred**2) + torch.sum(y_true**2)
        soft_dice_coefficient = 1 - (2.0 * intersection) / (sum_probs + 1e-7)  # Adding a small epsilon to avoid division by zero
        loss = soft_dice_coefficient
        return loss