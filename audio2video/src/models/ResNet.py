import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ResNet_Audio2FL(nn.Module):
    def __init__(self, num_output=68*3, num_seq=32):
        super(ResNet_Audio2FL, self).__init__()
        self.n_out = num_output
        self.n_seq = num_seq
        # shared layers
        self.audio_layer = AudioLayer()
        self.stage2 = self._make_layers(64, 128, 5, 2)
        self.stage3 = self._make_layers(128, 256, 4, 2)
        # high frequence branch
        self.stage_H = self._make_layers(256, 512, 3, 1)
        self.down_H = nn.Conv1d(512, self.n_out, 1, 1, padding='same')
        # low frequence branch
        self.stage_L = self._make_layers(256, 512, 3, 1)
        self.down_L = nn.Conv1d(512, self.n_out, 1, 1, padding='same')

    def _make_layers(self, n_in, n_out, n_blocks, stride):
        h_out = n_out // 4
        layers = [CONV_Block(n_in, h_out, n_out, stride)]
        for i in range(n_blocks-1):
            layers.append(ID_Block(n_out, h_out, n_out))
        return nn.Sequential(*layers)

    def forward(self, audio_input, energy_input):
        '''
            tensor size: [batch_size, sequence_length, channels]
        '''
        # shared layers
        out = self.audio_layer(audio_input.transpose(1,2).contiguous(), energy_input.transpose(1,2).contiguous())
        out = self.stage2(out)
        out = self.stage3(out)
        # high frequence branch
        H = self.stage_H(out)
        H = F.interpolate(H, size=self.n_seq, mode='linear', align_corners=False)
        H = self.down_H(H)
        H = H.transpose(1,2).contiguous()
        # low frequence branch
        L = self.stage_L(out)
        L = F.interpolate(L, size=self.n_seq, mode='linear', align_corners=False)
        L = self.down_L(L)
        L = L.transpose(1,2).contiguous()
        # Concatenate
        out = H + L
        return out, H, L


class AudioLayer(nn.Module):
    def __init__(self, input_dims=[256, 1], output_dims=[48, 16]):
        super(AudioLayer, self).__init__()
        
        self.audio_sample = nn.Sequential(
            nn.Conv1d(input_dims[0], output_dims[0], 3, 1, 1),
            nn.BatchNorm1d(output_dims[0]),
            nn.ReLU(),
            nn.Conv1d(output_dims[0], output_dims[0], 3, 1, 1),
        )
        self.energy_sample = nn.Sequential(
            nn.Conv1d(input_dims[1], output_dims[1], 3, 1, 1),
            nn.BatchNorm1d(output_dims[1]),
            nn.ReLU(),
            nn.Conv1d(output_dims[1], output_dims[1], 3, 1, 1),
        )
    
    def forward(self, audio, energy):
        out_audio = self.audio_sample(audio)
        out_energy = self.energy_sample(energy)
        return torch.cat([out_audio, out_energy], dim=1)


class CONV_Block(nn.Module):
    def __init__(self, n_in, h_out, n_out, stride):
        super(CONV_Block, self).__init__()

        self.residule = nn.Sequential(
            nn.Conv1d(n_in, h_out, 1, stride),
            nn.BatchNorm1d(h_out),
            nn.ReLU(),
            nn.Conv1d(h_out, h_out, 3, 1, padding='same'),
            nn.BatchNorm1d(h_out),
            nn.ReLU(),
            nn.Conv1d(h_out, n_out, 1, 1, padding='same'),
            nn.BatchNorm1d(n_out),
        )
        self.shortcut = nn.Sequential(
            nn.Conv1d(n_in, n_out, 1, stride),
            nn.BatchNorm1d(n_out),
        )
        self.relu = nn.ReLU()
    
    def forward(self, inputs):
        h = self.residule(inputs)
        s = self.shortcut(inputs)
        return self.relu(h + s)

class ID_Block(nn.Module):
    def __init__(self, n_in, h_out, n_out):
        super(ID_Block, self).__init__()

        self.residule = nn.Sequential(
            nn.Conv1d(n_in, h_out, 1, 1, padding='same'),
            nn.BatchNorm1d(h_out),
            nn.ReLU(),
            nn.Conv1d(h_out, h_out, 3, 1, padding='same'),
            nn.BatchNorm1d(h_out),
            nn.ReLU(),
            nn.Conv1d(h_out, n_out, 1, 1, padding='same'),
            nn.BatchNorm1d(n_out),
        )
        self.relu = nn.ReLU()
    
    def forward(self, inputs):
        h = self.residule(inputs)
        return self.relu(h + inputs)
