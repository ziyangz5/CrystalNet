import os
import torch
import torch.nn as nn
from ext import pytorch_ssim

def save_checkpoint(state, checkpoint_dir="checkpoints", filename="model_checkpoint.pth"):
    """Saves the model and optimizer states at a given checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved at {filepath}")

def load_checkpoint(checkpoint_path, model, optimizer):
    """Loads the model and optimizer states from a checkpoint file."""
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint.get('loss', None)
        print(f"Checkpoint loaded from {checkpoint_path} at epoch {epoch}")
        return epoch, loss
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return None, None

def tv_loss(img, tv_weight):
    """Computes total variation loss for regularization."""
    return tv_weight * (((img[:, :, :, :-1] - img[:, :, :, 1:]) ** 2).sum()
                        + ((img[:, :, :-1, :] - img[:, :, 1:, :]) ** 2).sum())

class DssimL1Loss(torch.nn.Module):

    def __init__(self,weight_L1=2.0):
        super(DssimL1Loss, self).__init__()
        self.loss_map = torch.zeros([0])
        self.weight_L1 = weight_L1

    def forward(self, pred, gt):
        self.loss_map = self.weight_L1 * torch.abs(pred-gt) + (1.0 - pytorch_ssim.ssim(pred, gt, size_average=False))

        return self.loss_map.mean()

class DL1Combine(torch.nn.Module):
    """Custom loss combining cross-entropy and L1 losses."""
    def __init__(self, weight_L1=2.0):
        super(DL1Combine, self).__init__()
        self.cls = nn.CrossEntropyLoss()
        self.rec = nn.L1Loss()

    def forward(self, pred_sn, pred_oi, pred_uv, sn, oi, uv):
        return self.cls(pred_oi, oi) + self.rec(pred_uv, uv) * 5.0 + self.rec(pred_sn, sn) * 2.0

