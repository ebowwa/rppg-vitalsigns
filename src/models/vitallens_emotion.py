import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from typing import Dict, Tuple, Optional

class VitalLensEmotionModel(nn.Module):
    def __init__(self, sequence_length=150, num_emotions=7, dropout_rate=0.3,
                 enable_audio=False, enable_eyetracking=False):
        super(VitalLensEmotionModel, self).__init__()
        
        self.sequence_length = sequence_length
        self.num_emotions = num_emotions
        self.enable_audio = enable_audio
        self.enable_eyetracking = enable_eyetracking
        
        self.backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Identity()
        backbone_features = 1280
        
        self.temporal_conv1 = nn.Conv1d(backbone_features, 512, kernel_size=3, padding=1)
        self.temporal_conv2 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.temporal_conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        
        self.lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.attention_weights = nn.Linear(128, 1)
        self.attention_dropout = nn.Dropout(dropout_rate)
        
        self.pulse_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, sequence_length)
        )
        
        self.respiration_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, sequence_length)
        )
        
        self.hr_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )
        
        self.rr_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )
        
        self.emotion_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_emotions)
        )
        
        if self.enable_audio:
            self.audio_conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            self.audio_pool = nn.AdaptiveAvgPool2d((8, 8))
            self.audio_head = nn.Sequential(
                nn.Linear(64 * 8 * 8, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, num_emotions)
            )
        
        if self.enable_eyetracking:
            self.eyetrack_head = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(32, 2)
            )
        
        if self.enable_audio or self.enable_eyetracking:
            fusion_input_size = 128
            if self.enable_audio:
                fusion_input_size += 64 * 8 * 8  # 4096 from audio_flat
            if self.enable_eyetracking:
                fusion_input_size += 32  # from eyetrack_head[:-1] output
            
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_input_size, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 128)
            )
            
            self.fused_emotion_head = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, num_emotions)
            )
        
    def forward(self, x, audio_features=None, eyetrack_features=None):
        batch_size, seq_len, channels, height, width = x.shape
        
        x = x.view(batch_size * seq_len, channels, height, width)
        features = self.backbone(x)
        features = features.view(batch_size, seq_len, -1)
        
        x = features.transpose(1, 2)
        x = F.relu(self.temporal_conv1(x))
        x = F.relu(self.temporal_conv2(x))
        x = F.relu(self.temporal_conv3(x))
        x = x.transpose(1, 2)
        
        lstm_out, _ = self.lstm(x)
        attention_scores = torch.softmax(self.attention_weights(lstm_out).squeeze(-1), dim=1)
        attention_scores = self.attention_dropout(attention_scores)
        attn_out = torch.sum(lstm_out * attention_scores.unsqueeze(-1), dim=1)
        global_features = attn_out
        
        pulse_waveform = self.pulse_head(global_features)
        resp_waveform = self.respiration_head(global_features)
        heart_rate = self.hr_head(global_features)
        resp_rate = self.rr_head(global_features)
        emotion_logits = self.emotion_head(global_features)
        
        outputs = {
            'pulse_waveform': pulse_waveform,
            'resp_waveform': resp_waveform,
            'heart_rate': heart_rate,
            'resp_rate': resp_rate,
            'emotion_logits': emotion_logits
        }
        
        if self.enable_audio and audio_features is not None:
            audio_conv_out = self.audio_conv(audio_features)
            audio_pooled = self.audio_pool(audio_conv_out)
            audio_flat = audio_pooled.view(audio_pooled.size(0), -1)
            audio_emotion = self.audio_head(audio_flat)
            outputs['audio_emotion_logits'] = audio_emotion
        
        if self.enable_eyetracking and eyetrack_features is not None:
            eyetrack_pred = self.eyetrack_head(global_features)
            outputs['eyetrack_coordinates'] = eyetrack_pred
        
        if (self.enable_audio or self.enable_eyetracking) and hasattr(self, 'fusion_layer'):
            fusion_inputs = [global_features]
            
            if self.enable_audio and audio_features is not None:
                fusion_inputs.append(audio_flat)
            
            if self.enable_eyetracking:
                eyetrack_features_extracted = self.eyetrack_head[:-1](global_features)
                fusion_inputs.append(eyetrack_features_extracted)
            
            if len(fusion_inputs) > 1:
                fused_features = torch.cat(fusion_inputs, dim=1)
                fused_output = self.fusion_layer(fused_features)
                fused_emotion = self.fused_emotion_head(fused_output)
                outputs['fused_emotion_logits'] = fused_emotion
        
        return outputs
