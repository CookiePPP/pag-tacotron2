from torch import nn
import torch

class Tacotron2Loss(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2Loss, self).__init__()
        self.pos_weight = torch.tensor(hparams.gate_positive_weight)
        self.alignment_loss_scaler = hparams.alignment_loss_weight
    
    def forward(self, model_output, targets):
        mel_target, gate_target, alignment_target = targets
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, alignment_out = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        alignment_loss = nn.MSELoss()(alignment_out, alignment_target)
        gate_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(gate_out, gate_target)
        return mel_loss + gate_loss + (self.alignment_loss_scaler * alignment_loss), alignment_loss.item()
