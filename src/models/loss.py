import torch
import torch.nn as nn
import torch.nn.functional as F

class VitalLensEmotionLoss(nn.Module):
    def __init__(self, pulse_weight=1.0, resp_weight=1.0, hr_weight=10.0, 
                 rr_weight=10.0, emotion_weight=5.0, audio_emotion_weight=3.0,
                 eyetrack_weight=2.0, fusion_weight=4.0):
        super(VitalLensEmotionLoss, self).__init__()
        self.pulse_weight = pulse_weight
        self.resp_weight = resp_weight
        self.hr_weight = hr_weight
        self.rr_weight = rr_weight
        self.emotion_weight = emotion_weight
        self.audio_emotion_weight = audio_emotion_weight
        self.eyetrack_weight = eyetrack_weight
        self.fusion_weight = fusion_weight
        
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        pulse_loss = self.mse_loss(predictions['pulse_waveform'], targets['pulse_waveform'])
        resp_loss = self.mse_loss(predictions['resp_waveform'], targets['resp_waveform'])
        
        hr_loss = self.mae_loss(predictions['heart_rate'], targets['heart_rate'])
        rr_loss = self.mae_loss(predictions['resp_rate'], targets['resp_rate'])
        
        emotion_loss = self.ce_loss(predictions['emotion_logits'], targets['emotion_labels'].squeeze(-1))
        
        pulse_snr_loss = self._snr_loss(predictions['pulse_waveform'])
        resp_snr_loss = self._snr_loss(predictions['resp_waveform'])
        
        total_loss = (
            self.pulse_weight * pulse_loss +
            self.resp_weight * resp_loss +
            self.hr_weight * hr_loss +
            self.rr_weight * rr_loss +
            self.emotion_weight * emotion_loss +
            0.1 * (pulse_snr_loss + resp_snr_loss)
        )
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'pulse_loss': pulse_loss.item(),
            'resp_loss': resp_loss.item(),
            'hr_loss': hr_loss.item(),
            'rr_loss': rr_loss.item(),
            'emotion_loss': emotion_loss.item(),
            'pulse_snr_loss': pulse_snr_loss.item(),
            'resp_snr_loss': resp_snr_loss.item()
        }
        
        if 'audio_emotion_logits' in predictions and 'audio_emotion_labels' in targets:
            audio_emotion_loss = self.ce_loss(predictions['audio_emotion_logits'], 
                                            targets['audio_emotion_labels'].squeeze(-1))
            total_loss += self.audio_emotion_weight * audio_emotion_loss
            loss_dict['audio_emotion_loss'] = audio_emotion_loss.item()
        
        if 'eyetrack_coordinates' in predictions and 'eyetrack_targets' in targets:
            eyetrack_loss = self.mse_loss(predictions['eyetrack_coordinates'], targets['eyetrack_targets'])
            total_loss += self.eyetrack_weight * eyetrack_loss
            loss_dict['eyetrack_loss'] = eyetrack_loss.item()
        
        if 'fused_emotion_logits' in predictions and 'emotion_labels' in targets:
            fused_emotion_loss = self.ce_loss(predictions['fused_emotion_logits'], 
                                            targets['emotion_labels'].squeeze(-1))
            total_loss += self.fusion_weight * fused_emotion_loss
            loss_dict['fused_emotion_loss'] = fused_emotion_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def _snr_loss(self, waveform):
        fft = torch.fft.fft(waveform, dim=-1)
        power = torch.abs(fft) ** 2
        
        peak_power = torch.max(power, dim=-1)[0]
        noise_power = torch.mean(power, dim=-1) - peak_power / power.shape[-1]
        
        snr = peak_power / (noise_power + 1e-8)
        snr_loss = -torch.log(snr + 1e-8).mean()
        
        return snr_loss
