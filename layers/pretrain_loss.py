import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft


class Contrasive_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def convert_coeff(self, x, eps=1e-6):
        amp = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))
        phase = torch.atan2(x.imag, x.real + eps)
        return amp, phase

    def instance_contrastive_loss(self, z1, z2):
        B, T = z1.size(0), z1.size(1)                          
        z = torch.cat([z1, z2], dim=0)  # 2B x T x C
        z = z.transpose(0, 1)  # T x 2B x C
        sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)

        i = torch.arange(B, device=z1.device)
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        return loss

    def forward(self, x_q, x_k):

        q_s = F.normalize(x_q, dim=-1)
        k_s = F.normalize(x_k, dim=-1)

        q_s_freq = fft.rfft(q_s, dim=1)
        k_s_freq = fft.rfft(k_s, dim=1)
        
        q_s_amp, q_s_phase = self.convert_coeff(q_s_freq)
        k_s_amp, k_s_phase = self.convert_coeff(k_s_freq)

        seasonal_loss = self.instance_contrastive_loss(q_s_amp, k_s_amp) + \
                        self.instance_contrastive_loss(q_s_phase,k_s_phase)
    
        return seasonal_loss
