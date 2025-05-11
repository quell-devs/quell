import math
import torch
from torch import nn
import torch.nn.functional as F
from torch import fft
from pytorch_msssim import MS_SSIM
import pytorch_msssim


# Notes about variable names
# high_dose = 24mGy scan
# low_dose = 4mGy scan
# residual = high_dose - low_dose
# [predicted_residual is the output of the model]
# prediction = high_dose - residual + predicted_residual
# prediction = low_dose + predicted_residual

def get_denoised_image(residual_prediction, high_dose, residual):
    return high_dose - residual + residual_prediction

def masked_metric(x, y, metric_func, **kwargs):
    result = metric_func(x, y, reduction='none', **kwargs)
    mask = y.isnan()
    result[ mask ] = 0.0

    result = result.nan_to_num(nan=0.0)
    return result.sum()/(~mask).sum()


def masked_smooth_l1_loss(residual_prediction, high_dose, residual):
    return masked_metric(residual_prediction, residual, F.smooth_l1_loss)

def masked_mse_loss(residual_prediction, high_dose, residual):
    return masked_metric(residual_prediction, residual, F.mse_loss)

def L1_full(residual_prediction, high_dose, residual):
    denoised = get_denoised_image(residual_prediction, high_dose, residual)
    return masked_metric(denoised, high_dose, F.l1_loss)

def L2_full(residual_prediction, high_dose, residual):
    denoised = get_denoised_image(residual_prediction, high_dose, residual)
    return masked_metric(denoised, high_dose, F.mse_loss)

def masked_psnr_full(residual_prediction, high_dose, residual):
    max_range = 10.0 # scale factor of breast tissue range
    mse = L2_full(residual_prediction, high_dose, residual)
    return 10 * math.log10(max_range**2/mse)

def masked_psnr(residual_prediction, high_dose, residual):
    max_range = 10.0 # scale factor of breast tissue range
    mse = masked_mse_loss(residual_prediction, high_dose, residual)
    return 10 * math.log10(max_range**2/mse)

def log_residual_mean_ratio(residual_prediction, high_dose, residual):
    log_residual_prediction_mean = residual_prediction.abs().mean().log()
    log_residual_mean = residual.abs().mean().log()

    result = torch.abs(log_residual_prediction_mean - log_residual_mean)

    return result

class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim_module = pytorch_msssim.SSIM(data_range=1.0, channel=1, spatial_dims=3, nonnegative_ssim=True)

    def forward(self, residual_prediction, high_dose, residual):
        # compute loss
        loss = 1 - self.ssim_module(torch.clamp(residual_prediction/10.0, min=0.0, max=1.0), 
                                    torch.clamp(residual/10.0, min=0.0, max=1.0))
        return loss

def ssim(residual_prediction, high_dose, residual):
    # ensure images are floats and normalised to range [0,1]
    return pytorch_msssim.ssim(torch.clamp(residual_prediction/10.0, min=0.0, max=1.0).float(), 
                              torch.clamp(residual/10.0, min=0.0, max=1.0).float(), 
                              data_range=1)

def ssim_full(residual_prediction, high_dose, residual):
    denoised = get_denoised_image(residual_prediction, high_dose, residual)
    # ensure images are floats and normalised to range [0,1]
    return pytorch_msssim.ssim(torch.clamp(denoised/10.0, min=0.0, max=1.0).float(), 
                              torch.clamp(high_dose/10.0, min=0.0, max=1.0).float(), 
                              data_range=1)


def znormalize(img):
    # znormalise wrt individual sample inside batch
    b = img.shape[0]
    im_flat = img.reshape(b,-1)
    mean = im_flat.mean(dim=1).unsqueeze(1)
    std  = im_flat.std(dim=1).unsqueeze(1)
    return ((im_flat - mean) / std).reshape(*img.shape)


class QuellLoss(nn.Module):
    def __init__(
        self, 
        smooth_l1_factor:float=1.0, 
        l2_loss_factor:float=0.0, 
        residual_ratio_factor:float=0.05, 
   ):
        super().__init__()
        self.smooth_l1_factor = float(smooth_l1_factor)
        self.l2_loss_factor = float(l2_loss_factor)
        self.residual_ratio_factor = float(residual_ratio_factor)
        
    def forward(self, residual_prediction, high_dose, residual):
        loss = 0.0

        if self.smooth_l1_factor > 0.0:
            loss += self.smooth_l1_factor * masked_smooth_l1_loss(residual_prediction, high_dose, residual)
        
        if self.l2_loss_factor > 0.0:
            loss += self.l2_loss_factor * masked_mse_loss(residual_prediction, high_dose, residual)

        if self.residual_ratio_factor > 0.0:
            loss += self.residual_ratio_factor * torch.clamp(log_residual_mean_ratio(residual_prediction, high_dose, residual), max=10)

        return loss
