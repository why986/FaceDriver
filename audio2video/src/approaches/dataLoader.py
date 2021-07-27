from .AudioFeatureSet import AudioFeatureSet
from torch.utils.data import DataLoader

def build_dataset(dataset_dir='data/', 
                    speech_set='', 
                    energy_set='energy/', 
                    keypoint_set='face_npy_delta/',
                    FREQ_SPLIT=4, batch_size=128):
    audioFeatureSet = AudioFeatureSet(dataset_dir, dataset_dir+speech_set, dataset_dir+energy_set, dataset_dir+keypoint_set, FREQ_SPLIT)
    return DataLoader(dataset=audioFeatureSet,
                        batch_size=batch_size,
                        shuffle=True)
