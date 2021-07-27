import torch
from torch.utils.data import Dataset, dataset
import numpy as np
import os, random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class AudioFeatureSet(Dataset):
    def __init__(self, dataset_dir, speech_set, energy_set, keypoint_set, FREQ_SPLIT=4, ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.speech_set = speech_set
        self.energy_set = energy_set
        self.keypoint_set = keypoint_set
        self.name_list = []
        for root, _, files in os.walk(self.dataset_dir):
            for f in files:
                file_name = os.path.join(root, f)
                if file_name[-3:] != 'wav':
                    continue
                self.name_list.append(f[:-4])
        self.len = len(self.name_list)
        self.FREQ_SPLIT = FREQ_SPLIT
    
    def __getitem__(self, index):
        file_name = self.name_list[index]+'.npy'
        prob = np.load(self.speech_set + file_name)
        audio_energy = np.load(self.energy_set + file_name)
        feature = np.load(self.keypoint_set + file_name)

        CUR_FRAME = random.randint(4, feature.shape[0]-32)
        while (CUR_FRAME-4)*2+80 > prob.shape[0] or (CUR_FRAME-4)*2+80 > audio_energy.shape[0]:
            CUR_FRAME = random.randint(4, feature.shape[0]-32)
        audio_feature = torch.from_numpy(prob[(CUR_FRAME-4)*2:(CUR_FRAME-4)*2+80,])
        energy_feature = torch.from_numpy(audio_energy[(CUR_FRAME-4)*2:(CUR_FRAME-4)*2+80,])
        keypoint_feature = torch.from_numpy(feature[(CUR_FRAME):(CUR_FRAME+32),]).float()
        if torch.cuda.is_available():
            audio_feature = audio_feature.cuda()
            energy_feature = energy_feature.cuda()
            keypoint_feature = keypoint_feature.cuda()
        F = torch.fft.fft(torch.transpose(keypoint_feature, 1, 0), 32)
        _split = self.FREQ_SPLIT
        H = torch.cat([torch.zeros_like(F[:,:_split]), F[:,_split:]], axis=1)
        L = torch.cat([F[:,:_split], torch.zeros_like(F[:,_split:])], axis=1)
        h = torch.fft.ifft(H).float()
        l = torch.fft.ifft(L).float()
        h = torch.transpose(h, 1, 0)
        l = torch.transpose(l, 1, 0)
        return energy_feature, audio_feature, keypoint_feature, h, l # h: high frequency l: low frequency
    
    def __len__(self):
        return self.len
