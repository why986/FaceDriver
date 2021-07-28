import os, math, cv2, librosa, torch
import numpy as np
from deepspeech.evaluate import get_prob
from utils import *
from src.approaches.train_audio2fl import Audio_Landmarks_block

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 25
colors = [(255, 144, 25), (50, 205, 50), (50, 205, 50), (208, 224, 63), (71, 99, 255), (71, 99, 255), (238, 130, 238), (238, 130, 238)]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def audio_to_landmarks(audio_path, video_size, base_landmarks, model_name, model_dir, tmp_dir='audio2landmarks_tmp'):
    '''
    Function: generate a video of the motion of facial landmarks according the audio
    Input:
        audio_path -- the path of audio file
        video_size -- the size of frame in video, a tuple (width, height)
        base_landmarks -- the facial landmarks of initial face portrait, 
                          each landmarks = base_landmarks + predict difference
        tmp_dir -- the directory which the tempory audio files are saved in
        model_name -- the weights used
    Output:
        the sequence of landmarks image
    '''
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    file_name = os.path.basename(audio_path).split('.')[0]
    # convert audio into format .wav and downsample
    tmp_audio_path = os.path.join(tmp_dir, f'{file_name}.wav')
    os.system(f'ffmpeg -y -i {audio_path} -ar 16000 {tmp_audio_path}')
    # extract audio semantic features
    prob = get_prob(tmp_audio_path)
    # ectract audio energy features
    audio, _ = librosa.load(tmp_audio_path, sr=None)
    audio_energy = librosa.feature.rms(y = audio, frame_length=512, hop_length=320, center=False)
    audio_energy = audio_energy.reshape((-1, 1))
    energy_mean = np.average(audio_energy)
    energy_std = np.std(audio_energy)
    energy = (audio_energy - energy_mean) / energy_std

    # split audio semantic and energy features
    audio_inputs, energy_inputs = audio_split(prob, energy)
    audio_inputs = torch.as_tensor(audio_inputs, dtype=torch.float32, device=device)
    energy_inputs = torch.as_tensor(energy_inputs, dtype=torch.float32, device=device)
    # predict facial landmarks
    model = Audio_Landmarks_block(model_save_dir=os.path.join(model_dir, model_name))
    model.load_model(epoch=200)
    landmarks = model.single_eval(audio_inputs, energy_inputs).cpu().detach().numpy().reshape(-1, 204)
    landmarks = np.concatenate([np.repeat(landmarks[0:1], 4, axis=0), landmarks], axis=0)
    # connect facial landmarks with different lines in each frame and concatenate these frames as a video
    # global fourcc, fps, colors
    global colors
    time = librosa.get_duration(filename=audio_path)
    length = min(int(math.ceil(time * fps)), landmarks.shape[0])
    landmarks = landmarks_restore(landmark=landmarks[:length], average=base_landmarks)
    landmarks[:, :, 0] += video_size[0]//2
    landmarks[:, :, 1] += video_size[1]//2
    output = []
    for i in range(length):
        img = landmarks_to_image([landmarks[i]], video_size, colors)
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        output.append(img)
    return np.array(output)
    # output_path = os.path.join(tmp_dir, f'{file_name}.avi')
    # out = cv2.VideoWriter(output_path, fourcc, fps, video_size)
    # for i in range(length):
    #     img = landmarks_to_image([landmarks[i]], video_size, colors)
    #     out.write(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
    # return output_path
