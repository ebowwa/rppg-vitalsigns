import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def calculate_rppg_metrics(predictions, targets):
    metrics = {}
    
    hr_mae = torch.mean(torch.abs(predictions['heart_rate'] - targets['heart_rate']))
    metrics['hr_mae'] = hr_mae.item()
    
    rr_mae = torch.mean(torch.abs(predictions['resp_rate'] - targets['resp_rate']))
    metrics['rr_mae'] = rr_mae.item()
    
    pulse_snr = calculate_snr(predictions['pulse_waveform'])
    metrics['pulse_snr'] = pulse_snr.item()
    
    resp_snr = calculate_snr(predictions['resp_waveform'])
    metrics['resp_snr'] = resp_snr.item()
    
    pulse_corr = calculate_correlation(predictions['pulse_waveform'], targets['pulse_waveform'])
    resp_corr = calculate_correlation(predictions['resp_waveform'], targets['resp_waveform'])
    metrics['pulse_cor'] = pulse_corr.item()
    metrics['resp_cor'] = resp_corr.item()
    
    return metrics

def calculate_emotion_metrics(predictions, targets):
    pred_labels = torch.argmax(predictions['emotion_logits'], dim=1)
    true_labels = targets['emotion_labels'].squeeze()
    
    accuracy = (pred_labels == true_labels).float().mean()
    
    pred_np = pred_labels.cpu().numpy()
    true_np = true_labels.cpu().numpy()
    
    return {
        'emotion_accuracy': accuracy.item(),
        'emotion_pred': pred_np,
        'emotion_true': true_np
    }

def calculate_snr(waveform):
    fft = torch.fft.fft(waveform, dim=-1)
    power = torch.abs(fft) ** 2
    
    peak_power = torch.max(power, dim=-1)[0]
    noise_power = torch.mean(power, dim=-1) - peak_power / power.shape[-1]
    
    snr_db = 10 * torch.log10(peak_power / (noise_power + 1e-8))
    return torch.mean(snr_db)

def calculate_correlation(pred, target):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    pred_mean = torch.mean(pred_flat)
    target_mean = torch.mean(target_flat)
    
    numerator = torch.sum((pred_flat - pred_mean) * (target_flat - target_mean))
    denominator = torch.sqrt(torch.sum((pred_flat - pred_mean) ** 2) * torch.sum((target_flat - target_mean) ** 2))
    
    correlation = numerator / (denominator + 1e-8)
    return correlation
