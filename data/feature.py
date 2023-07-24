import torch
import torch.nn as nn
import torchaudio
import config
import torch.nn.functional as F
    
class FeatureExtractor(nn.Module):
    
    def __init__(self, sys_config = config.SysConfig(),exp_config = config.ExpConfig()):
        super(FeatureExtractor, self).__init__()
        
        self.pre_emphasis_filter = torch.FloatTensor([[[-exp_config.pre_emphasis, 1.]]]).to(sys_config.device)
        
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate     =   exp_config.sample_rate,
            n_fft           =   exp_config.n_fft,
            win_length      =   exp_config.win_length,
            hop_length      =   exp_config.hop_length,
            window_fn       =   torch.hamming_window,
            n_mels          =   exp_config.n_mels
        ).to(sys_config.device)
        
    def forward(self, x):
        
        #with torch.no_grad():
            # input shape == (batch, length of utterance)
            # input shape of conv1d should be (batch, 1, length of utterance)
        x = x.unsqueeze(1)  
        x = F.pad(input=x, pad=(1, 0), mode='reflect')
        x = F.conv1d(input=x, weight=self.pre_emphasis_filter)
        x = self.mel_spec(x)
        x = torch.log(x + 1e-6)

        x = x.squeeze()
            # return shape == (batch, n_mel_filter, frames)
        return x
            