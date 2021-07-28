from audio2video.API import audio_to_landmarks
from audio2video.utils import *

file_name = '5_0000'
audio_path = f'audio2video/data/{file_name}.wav'
size = (255, 255)

landmarks_dir = 'audio2video/average_face'
face_aver = get_average_face(landmarks_dir)

path = audio_to_landmarks(audio_path, size, face_aver[0], model_name='multi_task_l2_1', model_dir='')
print(path)